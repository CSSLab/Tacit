import json

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from tacit import config as C
from tacit.suite import BENCHES, COLS, COLUMN_OF, FOLD_SEEDS, load_split
from scripts.guards.score_guards import GUARDS, load_guards, test_rows

SEED = FOLD_SEEDS[0]


def pr(y, pred):
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "n": int(len(y)), "n_unsafe": int(np.sum(y))}


def ours():
    d = json.load(open(C.RUNS / "readout" / "readout_probe_qwen3-4b.json"))
    tr = d["test_rows"]
    cols = np.array(tr["cols"])
    y = np.array(tr["labels"], dtype=int)
    pred = (np.array(tr["probs"]) >= 0.5).astype(int)
    out = {}
    for col in COLS:
        m = cols == col
        out[col] = pr(y[m], pred[m])
        assert abs(out[col]["macro_f1"] - d["test_f1"][col]) < 5e-4
    return out


def guards():
    arts = load_guards()
    split = load_split(SEED)
    out = {}
    for g, d in arts.items():
        per = {}
        for col in COLS:
            ys, ps = zip(*[test_rows(split, b, d) for b in BENCHES if COLUMN_OF[b] == col])
            per[col] = pr(np.concatenate(ys), np.concatenate(ps))
        out[g] = per
    return out


def main():
    sysd = {"TACIT probe (Qwen3-4B)": {"pr": ours()}}
    for g, per in guards().items():
        sysd[g] = {"pr": per}
    for s, v in sysd.items():
        gaps = [abs(v["pr"][b]["precision"] - v["pr"][b]["recall"]) for b in COLS]
        v["mean_abs_pr"] = round(float(np.mean(gaps)), 4)
        v["max_abs_pr"] = round(float(np.max(gaps)), 4)
        print(f"  {s:24s} mean|P-R|={v['mean_abs_pr']:.3f} max={v['max_abs_pr']:.3f} "
              + " ".join(f"{b[:6]}={v['pr'][b]['precision']-v['pr'][b]['recall']:+.2f}"
                         for b in COLS))
    p = C.run_dir("guards") / "policy_consistency.json"
    json.dump({"benches": COLS, "systems": sysd}, open(p, "w"), indent=1)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
