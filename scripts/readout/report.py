from __future__ import annotations
import json

import numpy as np

from tacit.suite import COLS
from scripts.readout.readout import BACKBONES, D, TIERS

B = 2000


def macro_f1_counts(tp, fp, fn, tn):
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1)
            + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2


def half(v):
    return round(float(np.diff(np.percentile(v, [2.5, 97.5]))[0] / 2), 4)


def main():
    ros = {bb: {t: json.load(open(D / f"readout_{t}_{bb}.json")) for t in TIERS
                if (D / f"readout_{t}_{bb}.json").exists()} for bb in BACKBONES}
    ref = next(d for ro in ros.values() for d in ro.values())["test_rows"]
    y = np.array(ref["labels"])
    col = np.array(ref["cols"])
    rng = np.random.default_rng(0)
    for bb, ro in ros.items():
        W = {c: rng.multinomial((col == c).sum(), np.full((col == c).sum(),
                                                          1 / (col == c).sum()), size=B)
             for c in COLS}
        if not ro:
            continue
        res = {}
        for tier, d in ro.items():
            tr = d["test_rows"]
            assert tr["rows"] == ref["rows"]
            pr = (np.array(tr["probs"]) >= 0.5).astype(int)
            f1 = np.zeros(len(COLS))
            bcol = np.zeros((len(COLS), B))
            for j, c in enumerate(COLS):
                m = col == c
                p, yc, w = pr[m], y[m], W[c]
                f1[j] = macro_f1_counts((p & yc).sum(), (p & (1 - yc)).sum(),
                                        ((1 - p) & yc).sum(), ((1 - p) & (1 - yc)).sum())
                bcol[j] = macro_f1_counts(w @ (p & yc), w @ (p & (1 - yc)),
                                          w @ ((1 - p) & yc), w @ ((1 - p) & (1 - yc)))
            res[tier] = {"f1": dict(zip(COLS, f1.round(4).tolist())),
                         "mean6": round(float(f1.mean()), 4),
                         "ci_half": dict(zip(COLS, [half(v) for v in bcol])),
                         "mean6_ci_half": half(bcol.mean(0))}
        print(f"\n### {bb}\n")
        print("| tier | " + " | ".join(COLS) + " | mean |")
        print("|---" * (len(COLS) + 2) + "|")
        for tier, r in res.items():
            print(f"| {tier} | " + " | ".join(
                f"{100 * r['f1'][c]:.1f}±{100 * r['ci_half'][c]:.1f}" for c in COLS)
                + f" | {100 * r['mean6']:.1f}±{100 * r['mean6_ci_half']:.1f} |")
        json.dump(res, open(D / f"report_{bb}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
