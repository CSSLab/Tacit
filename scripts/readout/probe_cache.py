from __future__ import annotations
import argparse
import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tacit import config as C
from tacit.features import labels_and_sources, load_feats, split_mask, stack
from tacit.suite import BENCHES, FOLD_SEEDS, load_split

OUT = C.RUNS / "readout" / "cache"
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
_G = {}


def row_ids(split, part):
    ids, off = [], 0
    for b in BENCHES:
        m = split_mask(split, b, part)
        ids.append(off + np.flatnonzero(m))
        off += len(m)
    return np.concatenate(ids)


def weights(y, cols):
    cells = {}
    for c, k in zip(cols, y):
        cells[(c, k)] = cells.get((c, k), 0) + 1
    n, m = len(y), len(cells)
    return np.array([n / (m * cells[(c, k)]) for c, k in zip(cols, y)])


def fit_probe(Z, y, w, c, seed):
    clf = LogisticRegression(penalty="l2", C=c, solver="liblinear", max_iter=4000,
                             random_state=seed)
    return clf.fit(Z, y, sample_weight=w)


def layer_task(pool, l, seed, tv):
    feats, split = _G["feats"], _G["split"]
    parts = [stack(feats, split, pool, l, p) for p in (("train", "val") if tv else ("train",))]
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = weights(y, np.concatenate([p[2] for p in parts]))
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    Zev = [sc.transform(stack(feats, split, pool, l, p)[0])
           for p in (("test",) if tv else ("val", "test"))]
    return [[fit_probe(Z, y, w, c, seed).predict_proba(E)[:, 1].astype(np.float32)
             for E in Zev] for c in CS]


def save(path, **arrays):
    tmp = str(path)[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=FOLD_SEEDS)
    ap.add_argument("--trainval", type=int, nargs="*", default=[FOLD_SEEDS[0]])
    ap.add_argument("--pools", nargs="+", default=["last", "mean", "both"])
    ap.add_argument("--njobs", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"probe_{a.model}"
    feats = load_feats(a.model, load_split(a.seeds[0]))
    L = feats[BENCHES[0]]["X"]["last"].shape[1]
    jobs = [(s, False) for s in a.seeds] + [(s, True) for s in a.trainval]
    for s, tv in jobs:
        split = load_split(s)
        yva, cva, _ = labels_and_sources(feats, split, "val")
        yte, cte, ute = labels_and_sources(feats, split, "test")
        meta = dict(yva=yva.astype(np.int8), cva=cva, iva=row_ids(split, "val"),
                    yte=yte.astype(np.int8), cte=cte, ite=row_ids(split, "test"),
                    ute=np.array(ute))
        _G.clear()
        _G.update(feats=feats, split=split)
        for pool in a.pools:
            t0 = time.time()
            res = Parallel(n_jobs=a.njobs, backend="multiprocessing")(
                delayed(layer_task)(pool, l, s, tv) for l in range(L))
            keys = [f"{pool}|{l}|{c}" for l in range(L) for c in CS]
            probs = [p for per_c in res for p in per_c]
            if tv:
                save(OUT / f"{stem}_tv{s}_{pool}.npz", keys=np.array(keys),
                     pte_tv=np.array([p[0] for p in probs]), **meta)
            else:
                save(OUT / f"{stem}_fold{s}_{pool}.npz", keys=np.array(keys),
                     pva=np.array([p[0] for p in probs]),
                     pte=np.array([p[1] for p in probs]), **meta)
            print(f"[cache] {stem} {'tv' if tv else 'fold'}{s} {pool}: {len(keys)} configs "
                  f"in {time.time() - t0:.0f}s", flush=True)
    print("PROBE CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
