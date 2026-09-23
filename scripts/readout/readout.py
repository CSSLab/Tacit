from __future__ import annotations
import argparse
import json

import numpy as np
from scipy.stats import rankdata

from tacit import config as C
from tacit.suite import COLS, FOLD_SEEDS

D = C.RUNS / "readout"
CACHE = D / "cache"
BACKBONES = {"qwen3-4b": "qwen3-4b-instruct", "llama3.1-8b": "llama3.1-8b-instruct"}
TIERS = {"probe": "probe", "ensemble": "probe", "multilayer": "multilayer"}
POOLS = ("last", "mean", "both")
CRITERIA = ["f1", "auc"]
BUDGETS = [5, 10, 20]
PROBE_RULES = [(c,) for c in CRITERIA]
ENSEMBLE_RULES = [(m, c) for m in BUDGETS for c in CRITERIA]
TV_SEED = FOLD_SEEDS[0]


def rname(r):
    return "/".join(map(str, r))


def load_units(kind, backbone, pools=POOLS):
    stem = f"{kind}_{BACKBONES[backbone]}"
    ts = [np.load(CACHE / f"{stem}_tv{TV_SEED}_{p}.npz") for p in pools]
    tkeys = np.concatenate([t["keys"] for t in ts])
    ptv = np.concatenate([t["pte_tv"] for t in ts])
    units = {}
    for s in FOLD_SEEDS:
        fs = [np.load(CACHE / f"{stem}_fold{s}_{p}.npz") for p in pools]
        keys = np.concatenate([f["keys"] for f in fs])
        assert (keys == tkeys).all()
        f0 = fs[0]
        units[s] = dict(keys=keys.astype(str), pva=np.concatenate([f["pva"] for f in fs]),
                        pte=np.concatenate([f["pte"] for f in fs]), ptv=ptv,
                        yva=f0["yva"].astype(int), cva=f0["cva"].astype(str),
                        iva=f0["iva"], yte=f0["yte"].astype(int),
                        cte=f0["cte"].astype(str), ute=f0["ute"].astype(str),
                        ite=f0["ite"])
    k0 = units[FOLD_SEEDS[0]]
    for u in units.values():
        assert (u["keys"] == k0["keys"]).all() and (u["ite"] == k0["ite"]).all()
    allval = np.concatenate([u["iva"] for u in units.values()])
    assert len(np.unique(allval)) == len(allval)
    assert not np.intersect1d(allval, k0["ite"]).size
    return units


def tiebreak(keys):
    def one(k):
        f = k.split("|")
        if len(f) != 3:
            return (-1, 0.0, 0.0, 0, "")
        return (0, float(f[2]), 0.0, int(f[1]), f[0])
    order = sorted(range(len(keys)), key=lambda i: one(keys[i]) + (i,))
    tb = np.empty(len(keys), dtype=int)
    tb[order] = np.arange(len(keys))
    return tb


def f1_rows(y, E):
    yy = y.astype(bool)
    npos, nneg = int(yy.sum()), int((~yy).sum())
    pred = E >= 0.5
    tp = (pred & yy).sum(1)
    fp = pred.sum(1) - tp
    fn, tn = npos - tp, nneg - fp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1)
            + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2


def auc_rows(y, P):
    r = rankdata(P, axis=1)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return (r[:, pos].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)


def f1(y, p):
    return float(f1_rows(y, p[None, :])[0])


SCORE = {"f1": f1_rows, "auc": auc_rows}
_ST = {}
_GR = {}


def selection_stats(units, folds):
    key = tuple(folds)
    if key in _ST:
        return _ST[key]
    st = {}
    for c in COLS:
        P = np.concatenate([units[s]["pva"][:, units[s]["cva"] == c] for s in folds], axis=1)
        y = np.concatenate([units[s]["yva"][units[s]["cva"] == c] for s in folds])
        st[c] = {"P": P, "y": y, **{k: SCORE[k](y, P) for k in CRITERIA}}
    st["_G"] = {k: np.mean([st[c][k] for c in COLS], axis=0) for k in CRITERIA}
    _ST[key] = st
    return st


def pick_one(st, criterion, tb):
    return int(np.lexsort((tb, -np.round(st["_G"][criterion], 6)))[0])


def greedy_path(st, criterion, tb):
    key = (id(st), criterion)
    if key in _GR:
        return _GR[key]
    pri = np.argsort(np.argsort(tb))
    sums = {c: np.zeros(st[c]["P"].shape[1]) for c in COLS}
    chosen, hist = [], []
    for k in range(1, max(BUDGETS) + 1):
        sc = sum(SCORE[criterion](st[c]["y"], (sums[c][None, :] + st[c]["P"]) / k)
                 for c in COLS) / len(COLS)
        best = int(np.lexsort((pri, -np.round(sc, 6)))[0])
        chosen.append(best)
        for c in COLS:
            sums[c] += st[c]["P"][best]
        hist.append(float(sc[best]))
    _GR[key] = (chosen, hist)
    return _GR[key]


def select(rule, st, tb):
    if len(rule) == 1:
        return np.array([pick_one(st, rule[0], tb)])
    m, criterion = rule
    chosen, hist = greedy_path(st, criterion, tb)
    return np.array(chosen[:int(np.argmax(np.round(hist[:m], 6))) + 1])


def nested(units, tb, rules):
    sc = {r: {c: [] for c in COLS} for r in rules}
    for j in FOLD_SEEDS:
        st = selection_stats(units, [s for s in FOLD_SEEDS if s != j])
        u = units[j]
        for r in rules:
            m = select(r, st, tb)
            for c in COLS:
                mask = u["cva"] == c
                sc[r][c].append(f1(u["yva"][mask], u["pva"][m][:, mask].mean(0)))
    return {r: float(np.mean([np.mean(v) for v in d.values()])) for r, d in sc.items()}


def choose(units, tb, rules):
    ns = nested(units, tb, rules)
    best = max(rules, key=lambda r: (round(ns[r], 6), -rules.index(r)))
    return best, select(best, selection_stats(units, FOLD_SEEDS), tb), ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), required=True)
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    a = ap.parse_args()
    units = load_units(TIERS[a.tier], a.backbone)
    keys = list(units[FOLD_SEEDS[0]]["keys"])
    best, m, ns = choose(units, tiebreak(keys), PROBE_RULES if a.tier == "probe"
                         else ENSEMBLE_RULES)
    u = units[TV_SEED]
    p = u["ptv"][m].mean(0)
    scores = {c: f1(u["yte"][u["cte"] == c], p[u["cte"] == c]) for c in COLS}
    out = {"tier": a.tier, "backbone": a.backbone, "rule": rname(best),
           "nested_mean6": {rname(r): round(v, 4) for r, v in ns.items()},
           "members": [keys[i] for i in m],
           "test_f1": {c: round(v, 4) for c, v in scores.items()},
           "test_mean6": round(float(np.mean(list(scores.values()))), 4),
           "test_rows": {"uids": u["ute"].tolist(), "cols": u["cte"].tolist(),
                         "rows": u["ite"].tolist(), "labels": u["yte"].tolist(),
                         "probs": [round(float(v), 6) for v in p]}}
    path = D / f"readout_{a.tier}_{a.backbone}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"[readout] {a.tier} {a.backbone}: {out['rule']} n={len(set(out['members']))} "
          f"test {out['test_mean6']} " + " ".join(f"{c[:6]}={v:.3f}" for c, v in scores.items()))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
