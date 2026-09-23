"""TraceSafe loader (data/golden_*.jsonl). Each attacked record yields its mutated trace
(unsafe) and its original trace (benign, uid suffix ``-orig``)."""
from __future__ import annotations
import glob
import json
from pathlib import Path

from .schema import Trace, Turn
from .. import config as C


def _s(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _turns(trace_msgs: list[dict]) -> list[Turn]:
    out: list[Turn] = []
    for m in trace_msgs:
        role, content = m.get("role"), m.get("content")
        if role == "user":
            out.append(Turn("user", content=_s(content)))
        elif role in ("agent", "assistant"):
            if isinstance(content, dict) and "name" in content:
                args = content.get("arguments")
                call = {k: v for k, v in content.items() if k != "reasoning"}
                out.append(Turn("agent", action=_s(call),
                                tool_name=content.get("name"),
                                tool_args=args if isinstance(args, dict) else None))
            else:
                out.append(Turn("agent", action=_s(content)))
        elif role in ("tool", "environment"):
            out.append(Turn("environment", content=_s(content)))
        else:
            out.append(Turn("user", content=_s(content)))
    return out


def load(data_dir: str | None = None, include_original: bool = True) -> list[Trace]:
    root = Path(data_dir or C.TRACESAFE_DIR)
    files = sorted(glob.glob(str(root / "golden_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no golden_*.jsonl under {root}")
    traces: list[Trace] = []
    for fp in files:
        stem = Path(fp).stem                      # golden_1_PromptInjectionIn
        is_benign = "benign" in stem.lower()
        for i, line in enumerate(open(fp)):
            r = json.loads(line)
            gm = r.get("golden_meta", {})
            cat = gm.get("category") or r.get("mutation_category", "")
            nt = r.get("new_trace") or {}
            traces.append(Trace(
                uid=f"tracesafe-{stem}-{i}",
                source="tracesafe",
                label=0 if is_benign else 1,
                turns=_turns(nt.get("trace", [])),
                attack_type="benign" if is_benign else cat,
                tools=nt.get("tool_lists", []) or [],
                meta={"file": Path(fp).name, "env": nt.get("environment", "")},
            ))
            if include_original and not is_benign:
                ot = r.get("original_trace") or {}
                if ot.get("trace"):
                    traces.append(Trace(
                        uid=f"tracesafe-{stem}-{i}-orig",
                        source="tracesafe", label=0,
                        turns=_turns(ot.get("trace", [])),
                        attack_type="benign_matched",
                        tools=ot.get("tool_lists", []) or [],
                        meta={"file": Path(fp).name, "matched_to": f"{stem}-{i}"},
                    ))
    return traces
