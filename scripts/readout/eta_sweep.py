from __future__ import annotations
import argparse
import json
import time

import numpy as np
from joblib import Parallel, delayed

from tacit.features import load_feats, labels_and_sources
from tacit.probing.multilayer import select_salient
from tacit.suite import load_split
from scripts.readout import readout as R
from scripts.readout import multilayer_cache as S

ETAS = [0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
SEED = 42


def global_config(bb):
    units = R.load_units("multilayer", bb)
    R._ST.clear()
    keys = list(units[R.FOLD_SEEDS[0]]["keys"])
    i = R.pick_one(R.selection_stats(units, R.FOLD_SEEDS), "f1", R.tiebreak(keys))
    return keys[i], units[R.FOLD_SEEDS[0]]["ptv"][i]


def sweep(bb, njobs):
    key, cached = global_config(bb)
    pool, lc, eta0, lo, alpha_mode, head, hp = key.split("|")
    lc, eta0, lo = float(lc), float(eta0), float(lo)
    if head == "lr":
        hp_ = float(hp)
    else:
        h, a = hp.split("-")
        hp_ = ((int(h),), float(a))
    print(f"[sweep] {bb}: global config {key}", flush=True)
    split = load_split(SEED)
    feats = load_feats(R.BACKBONES[bb], split)
    ytr, _, _ = labels_and_sources(feats, split, "train")
    yva, _, _ = labels_and_sources(feats, split, "val")
    yte, cte, _ = labels_and_sources(feats, split, "test")
    cte = np.array(cte)
    Z = S.standardized(feats, split, pool, fit_on=("train", "val"))
    del feats
    Ztv = np.concatenate([Z.pop("train"), Z.pop("val")], axis=0)
    Zte = Z.pop("test")
    ytv = np.concatenate([ytr, yva])
    L, H = Ztv.shape[1], Ztv.shape[2]
    S._G.update(Ztr=Ztv, Zva=Zte, ytr=ytv, yva=yte)
    t0 = time.time()
    res = Parallel(n_jobs=njobs, backend="multiprocessing")(
        delayed(S.layer_probe)(l, lc, SEED) for l in range(L))
    coefs = [r[0] for r in res]
    train_f1 = np.array([r[1] for r in res])
    print(f"[sweep] {bb}: {L} stage-1 probes in {time.time() - t0:.0f}s", flush=True)
    layers = S.kept_layers(L, lo)
    f = train_f1[layers]
    alpha = (((f - f.min()) / (f.max() - f.min())).tolist() if alpha_mode == "train"
             and f.max() > f.min() else [1.0] * len(layers))
    rows = {}
    for eta in ETAS + ["none"]:
        t0 = time.time()
        salient = ([np.arange(H)] * len(layers) if eta == "none"
                   else [select_salient(coefs[l], eta) for l in layers])
        Atv, Ate = S.aggregate(Ztv, layers, salient, alpha), S.aggregate(Zte, layers, salient, alpha)
        mu, sd = Atv.mean(0, keepdims=True), Atv.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        Atv, Ate = (Atv - mu) / sd, (Ate - mu) / sd
        clf = S.make_head(head, hp_, SEED)
        clf.fit(Atv, ytv)
        p = clf.predict_proba(Ate)[:, 1].astype(np.float32)
        f1 = {c: R.f1(yte[cte == c], p[cte == c]) for c in R.COLS}
        n = int(Atv.shape[1])
        rows[str(eta)] = {"n_features": n, "share": n / (len(layers) * H),
                          "test_f1": {c: round(v, 4) for c, v in f1.items()},
                          "test_mean6": round(float(np.mean(list(f1.values()))), 4)}
        if eta == eta0:
            rows[str(eta)]["max_abs_diff_vs_cache"] = float(np.abs(p - cached).max())
            fc = np.mean([R.f1(yte[cte == c], cached[cte == c]) for c in R.COLS])
            rows[str(eta)]["cache_test_mean6"] = round(float(fc), 4)
        print(f"[sweep] {bb} eta={eta}: {n} dims ({rows[str(eta)]['share']:.1%}) "
              f"test mean {rows[str(eta)]['test_mean6']} ({time.time() - t0:.0f}s)",
              flush=True)
        del Atv, Ate
    S._G.clear()
    return {"config": key, "eta0": eta0, "layers": [int(x) for x in layers], "L": L,
            "H": H, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(R.BACKBONES))
    ap.add_argument("--njobs", type=int, default=16)
    a = ap.parse_args()
    out = {}
    p = R.D / "eta_sweep.json"
    for bb in a.backbones:
        out[bb] = sweep(bb, a.njobs)
        json.dump(out, open(p, "w"), indent=1)
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
