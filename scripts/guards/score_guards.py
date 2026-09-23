"""Test macro-F1 per benchmark of the guards' per-row predictions
(run_moderation_guard.py, run_agentdog.py), and the mean over the six.

Usage: python scripts/guards/score_guards.py
"""
from __future__ import annotations
import json

import numpy as np
from sklearn.metrics import f1_score

from tacit import config as C
from tacit.features import keyed
from tacit.suite import BENCHES, COLS, COLUMN_OF, FOLD_SEEDS, load_split

GUARDS = {"Qwen3Guard-4B": "qwen3guard-4b", "LlamaGuard3-8B": "llamaguard3-8b",
          "AgentDoG-4B": "agentdog"}


def load_guards():
    arts = {g: json.load(open(C.RUNS / "guards" / k / "all.json")) for g, k in GUARDS.items()}
    for g, d in arts.items():
        for b in BENCHES:
            y, p = np.array(d[b]["labels"]), np.array(d[b]["preds"])
            got = f1_score(y, p, average="macro")
            assert abs(got - d[b]["macro_f1"]) < 1e-3, (g, b, got, d[b]["macro_f1"])
    return arts


def test_rows(split, b, d):
    """Labels and predictions of source b's test rows, aligned by uid."""
    pred = dict(zip(keyed(d[b]["uids"]), d[b]["preds"]))
    lab = dict(zip(keyed(d[b]["uids"]), d[b]["labels"]))
    sb = split["benches"][b]
    te = [u for u, k in zip(keyed(sb["uids"]), sb["split"]) if k == "test"]
    y = np.array([lab[u] for u in te])
    assert y.tolist() == [l for l, k in zip(sb["labels"], sb["split"]) if k == "test"], b
    return y, np.array([pred[u] for u in te])


def main():
    arts = load_guards()
    split = load_split(FOLD_SEEDS[0])        # the test side is the same in every file
    out = {}
    for g, d in arts.items():
        per = {}
        for col in COLS:
            ys, ps = zip(*[test_rows(split, b, d) for b in BENCHES if COLUMN_OF[b] == col])
            per[col] = round(float(f1_score(np.concatenate(ys), np.concatenate(ps),
                                            average="macro")), 4)
        per["mean6"] = round(float(np.mean([per[c] for c in COLS])), 4)
        out[g] = per
        print(f"| {g} | " + " | ".join(f"{100 * per[c]:.1f}" for c in COLS)
              + f" | {100 * per['mean6']:.1f} |", flush=True)
    p = C.run_dir("guards") / "table1_guards.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
