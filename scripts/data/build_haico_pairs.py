"""HAICOSYSTEM pairs: per scenario profile, the unsafe and safe (label 0) episodes with
the highest word-set Jaccard overlap, where the unsafe member has
content_safety_risks < 0 (``content``) or is any label-1 episode (``any``).

Writes DERIVED/haico_pairs.jsonl (the trajectories of every selected pair) and
DERIVED/haico_pairs_manifest.json (the pairing).
Usage: python scripts/data/build_haico_pairs.py
"""
import json
from collections import Counter, defaultdict

from tacit import config as C
from tacit.data import haicosystem
from tacit.utils.modeling import trace_to_plaintext

RISK = ("targeted_safety_risks", "system_and_operational_risks", "content_safety_risks",
        "societal_risks", "legal_and_rights_related_risks")
OUT = C.DERIVED / "haico_pairs.jsonl"
MAN = C.DERIVED / "haico_pairs_manifest.json"


def episode_dims():
    d = {}
    for line in open(C.HAICOSYSTEM_DIR / "all_episodes.jsonl"):
        ep = json.loads(line)
        try:
            d[f"haicosystem-{ep['pk']}"] = ep["rewards"][1][1]
        except (KeyError, IndexError, TypeError):
            continue
    return d


def select(groups, txt, dims, require):
    pairs = []
    for prof, members in groups.items():
        pos = [t for t in members if t.label == 1
               and (require is None or dims.get(t.uid, {}).get(require, 0) < 0)]
        saf = [t for t in members if t.label == 0]
        if not pos or not saf:
            continue
        best = None
        for a in pos:
            wa = set(txt[a.uid].lower().split())
            for b in saf:
                wb = set(txt[b.uid].lower().split())
                j = len(wa & wb) / max(len(wa | wb), 1)
                if best is None or j > best[0]:
                    best = (j, a.uid, b.uid)
        pairs.append({"unsafe": best[1], "safe": best[2], "jaccard": round(best[0], 4),
                      "scenario": prof[:120]})
    return pairs


def main():
    dims = episode_dims()
    traces = haicosystem.load()
    by_uid = {t.uid: t for t in traces}
    txt = {t.uid: trace_to_plaintext(t) for t in traces}
    groups = defaultdict(list)
    for t in traces:
        groups[(t.profile or "")[:400]].append(t)

    sel = {"content": select(groups, txt, dims, "content_safety_risks"),
           "any": select(groups, txt, dims, None)}
    keep = sorted({u for v in sel.values() for p in v for u in (p["unsafe"], p["safe"])})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for u in keep:
            t = by_uid[u]
            f.write(json.dumps({
                "uid": u, "label": t.label, "profile": t.profile,
                "domain": t.attack_type,
                "dims": {k: dims.get(u, {}).get(k) for k in RISK},
                "turns": [{"role": x.role, "content": x.content, "action": x.action,
                           "thought": x.thought, "tool_name": x.tool_name}
                          for x in t.turns]}, ensure_ascii=False) + "\n")
    json.dump(sel, open(MAN, "w"), indent=1)

    for k, v in sel.items():
        dom = Counter(by_uid[p["unsafe"]].attack_type for p in v)
        print(f"[{k}] {len(v)} pairs | median jaccard "
              f"{sorted(p['jaccard'] for p in v)[len(v) // 2]:.3f} | domains {dom.most_common(4)}",
              flush=True)
    print(f"trajectories written: {len(keep)}\nsaved {OUT} and {MAN}", flush=True)


if __name__ == "__main__":
    main()
