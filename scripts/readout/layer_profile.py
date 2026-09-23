from __future__ import annotations
import argparse
import json

import numpy as np

from tacit.suite import COLS, FOLD_SEEDS
from scripts.readout import readout as R

REPS = 5000


def mean6(y, cols, pred, reps=None):
    rng = np.random.default_rng(0)
    out = np.zeros((reps or 1, pred.shape[0]))
    for c in COLS:
        rows = np.flatnonzero(cols == c)
        idx = rows[None] if reps is None else rng.choice(rows, size=(reps, len(rows)))
        yy, pp = y[idx][None], pred[:, idx]
        tp, fp = (pp & yy).sum(2), (pp & ~yy).sum(2)
        fn, tn = (~pp & yy).sum(2), (~pp & ~yy).sum(2)
        out += ((2 * tp / np.maximum(2 * tp + fp + fn, 1)
                 + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2).T / len(COLS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(R.BACKBONES), default="qwen3-4b")
    a = ap.parse_args()
    units = R.load_units("probe", a.backbone)
    keys = units[FOLD_SEEDS[0]]["keys"]
    layer_of = np.array([int(k.split("|")[1]) for k in keys])
    y = np.concatenate([units[s]["yva"] for s in FOLD_SEEDS]).astype(bool)
    cols = np.concatenate([units[s]["cva"] for s in FOLD_SEEDS])
    pred, picks = [], {}
    for l in range(layer_of.max() + 1):
        sel = np.flatnonzero(layer_of == l)
        sub = {s: dict(u, keys=u["keys"][sel], pva=u["pva"][sel], pte=u["pte"][sel],
                       ptv=u["ptv"][sel]) for s, u in units.items()}
        R._ST.clear()
        R._GR.clear()
        tb = R.tiebreak(list(sub[FOLD_SEEDS[0]]["keys"]))
        best, m, _ = R.choose(sub, tb, R.PROBE_RULES)
        p = []
        for j in FOLD_SEEDS:
            mj = R.select(best, R.selection_stats(sub, [s for s in FOLD_SEEDS if s != j]), tb)
            p.append(sub[j]["pva"][mj].mean(0) >= 0.5)
        pred.append(np.concatenate(p))
        picks[l] = {"rule": R.rname(best), "member": str(sub[FOLD_SEEDS[0]]["keys"][m[0]])}
    pred = np.array(pred)
    val = mean6(y, cols, pred)[0]
    lo, hi = np.percentile(mean6(y, cols, pred, REPS), [2.5, 97.5], axis=0)
    out = {int(l): {**picks[l], "val_mean6": round(float(val[l]), 4),
                    "ci95": [round(float(lo[l]), 4), round(float(hi[l]), 4)]}
           for l in picks}
    json.dump(out, open(R.D / f"layer_profile_{a.backbone}.json", "w"), indent=1)
    for l, r in out.items():
        print(f"layer {l:2d} {r['member']:18s} val {r['val_mean6']:.4f} {r['ci95']}")


if __name__ == "__main__":
    main()
