"""Placebo members for the TraceSafe MissingTypeHint and AmbiguousArg pairs: the benign
member with the category's schema edit (strip_types or ambiguate) applied to a tool it
never calls, the one closest to the pair's target tool in list position and prototype
length.

Writes DERIVED/tracesafe_placebo.jsonl (loaded by tacit.data.tracesafe_placebo).
Usage: python scripts/data/build_tracesafe_placebo.py
"""
import copy
import json
import re

from tacit import config as C

FILES = {"MissingTypeHint": "golden_10_MissingTypeHint.jsonl",
         "AmbiguousArg": "golden_6_AmbiguousArg.jsonl"}
OUT = C.DERIVED / "tracesafe_placebo.jsonl"
DEF = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*(->[^:]*)?:\s*(pass)?\s*$",
                 re.S)


def split_args(s):
    """Split a signature's argument list on top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out if a.strip()]


def strip_types(proto):
    """MissingTypeHint: drop the annotations and the return type, keep the defaults,
    e.g. ``def ls(a: boolean = False) -> Any: pass`` becomes ``def ls(a=False):``."""
    m = DEF.match(proto or "")
    if not m:
        return None
    name, args = m.group(1), split_args(m.group(2))
    bare = []
    for a in args:
        nm = a.split(":")[0].split("=")[0].strip()
        bare.append(f"{nm}={a.split('=', 1)[1].strip()}" if "=" in a else nm)
    return f"def {name}({', '.join(bare)}):"


def abbrev(name):
    parts = [p for p in name.split("_") if p]
    if len(parts) > 1:
        return "".join(p[0] for p in parts)
    return name[:2]


def ambiguate(tool):
    """AmbiguousArg: abbreviate parameter names and strip their descriptions."""
    t = copy.deepcopy(tool)
    props = (t.get("parameters") or {})
    if isinstance(props, dict) and "properties" in props:
        props = props["properties"]
    if not isinstance(props, dict) or not props:
        return None
    ren = {k: abbrev(k) for k in props}
    if all(ren[k] == k for k in props):
        return None
    newp = {}
    for k, v in props.items():
        vv = {kk: vvv for kk, vvv in (v or {}).items() if kk != "description"} \
            if isinstance(v, dict) else v
        newp[ren[k]] = vv
    if isinstance(t.get("parameters"), dict) and "properties" in t["parameters"]:
        t["parameters"]["properties"] = newp
    else:
        t["parameters"] = newp
    m = DEF.match(t.get("prototype") or "")
    if m:
        args = split_args(m.group(2))
        new_args = []
        for a in args:
            nm = a.split(":")[0].split("=")[0].strip()
            new_args.append(a.replace(nm, ren.get(nm, nm), 1))
        ret = (m.group(3) or "").strip()
        t["prototype"] = f"def {m.group(1)}({', '.join(new_args)}) {ret}: pass".replace("  ", " ")
    return t


def called_tools(trace):
    out = set()
    for m in trace or []:
        c = m.get("content")
        if isinstance(c, dict) and c.get("name"):
            out.add(c["name"])
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for cat, fn in FILES.items():
        for i, line in enumerate(open(C.TRACESAFE_DIR / fn)):
            r = json.loads(line)
            ot = r["original_trace"]
            tools = copy.deepcopy(ot.get("tool_lists") or [])
            target = (r.get("mutation_metadata") or {}).get("target_tool")
            used = called_tools(ot.get("trace"))
            src = next((t for t in tools if t.get("name") == target), None)
            tgt_len = len(((src or {}).get("prototype")) or "")
            tgt_pos = next((j for j, t in enumerate(tools) if t.get("name") == target), 0)
            cands = []
            for j, t in enumerate(tools):
                if t.get("name") in used or t.get("name") == target:
                    continue
                new = strip_types(t.get("prototype")) if cat == "MissingTypeHint" else ambiguate(t)
                if new is None:
                    continue
                if cat == "MissingTypeHint" and new == (t.get("prototype") or "").strip():
                    continue
                cost = (abs(j - tgt_pos) / max(len(tools), 1)
                        + abs(len(t.get("prototype") or "") - tgt_len) / max(tgt_len, 1))
                cands.append((cost, j, new))
            if not cands:
                continue
            _, j, new = min(cands, key=lambda x: x[0])
            placebo_name = tools[j].get("name")
            if cat == "MissingTypeHint":
                tools[j] = {**tools[j], "prototype": new}
            else:
                tools[j] = new
            records.append({
                "uid": f"tracesafe-golden_{fn.split('_')[1]}_{cat}-{i}-placebo",
                "category": cat, "target_tool": target, "placebo_tool": placebo_name,
                "n_tools": len(tools), "tool_lists": tools,
                "trace": ot.get("trace"), "environment": ot.get("environment", "")})

    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"placebo records: {len(records)}\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
