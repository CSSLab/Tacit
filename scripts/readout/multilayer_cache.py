"""Cached probabilities of the Multi-layer (cross-layer aggregation) grid.

  stage 1  per pooling and L1 strength, one L1-regularized logistic probe per layer on
           the standardized training rows
  stage 2  for every (eta, first layer, layer weighting) the salient dimensions of every
           kept layer are concatenated, re-standardized, and a head is fitted: an L2
           logistic regression at several C, or a one-hidden-layer MLP
Layer weighting is either uniform ("none") or each layer's stage-1 training macro-F1,
min-max scaled over the kept layers ("train").

As in probe_cache.py, each fold's grid is fitted on its train split (probabilities on
validation and test), and with --trainval the whole pipeline is refitted on train+val
(probabilities on test). One file per pooling:
  RUNS/readout/cache/multilayer_<model>_{fold<k>,tv<k>}_<pool>.npz

Usage: python scripts/readout/multilayer_cache.py --model qwen3-4b-instruct --seeds 42 \
           --pools last --njobs 8
       python scripts/readout/multilayer_cache.py --model qwen3-4b-instruct --seeds 42 \
           --pools last --njobs 8 --trainval
"""
from __future__ import annotations
import argparse
import itertools
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from tacit import config as C
from tacit.features import load_feats, labels_and_sources, split_mask
from tacit.probing.multilayer import select_salient
from tacit.suite import BENCHES, FOLD_SEEDS, load_split
from scripts.readout.probe_cache import row_ids

OUT = C.RUNS / "readout" / "cache"
POOLS = ["last", "mean", "both"]
LAYER_CS = [0.03, 0.1, 0.3]
ETAS = [0.6, 0.9, 0.99]
LAYER_LOS = [0.0, 0.5]
ALPHAS = ["train", "none"]
HEADS = ([("lr", c) for c in (0.03, 0.1, 0.3, 1.0, 3.0)]
         + [("mlp", (h, a)) for h in ((64,), (256,)) for a in (1e-3, 1e-1)])
# relative fit cost per aggregated feature of each head; orders the task queue only
HEAD_COST = [1.0, 1.15, 1.22, 1.26, 1.32, 0.1, 0.1, 0.48, 0.39]

# Worker state inherited through fork: the standardized stacks of the current pooling
# and the stage-1 probes; tasks receive indices only.
_G = {}


def standardized(feats, split, pool, fit_on=("train",)):
    """Per-layer standardization of one pooling, fitted on `fit_on` and applied to all
    three parts."""
    parts = {}
    for part in ("train", "val", "test"):
        Xs = []
        for b in BENCHES:
            m = split_mask(split, b, part)
            X = feats[b]["X"]
            if pool == "both":
                Xs.append(np.concatenate([X["mean"][m], X["last"][m]], axis=2))
            else:
                Xs.append(X[pool][m])
        parts[part] = np.concatenate(Xs, axis=0).astype(np.float32)   # (N, L, H')
    ref = np.concatenate([parts[p] for p in fit_on], axis=0) if len(fit_on) > 1 \
        else parts[fit_on[0]]
    mu = ref.mean(0, keepdims=True)
    sd = ref.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    for part in parts:
        parts[part] = (parts[part] - mu) / sd
    return parts


def layer_probe(l, layer_c, seed):
    """Stage 1 at one layer: coefficients and training / validation macro-F1."""
    Ztr, Zva, ytr, yva = _G["Ztr"], _G["Zva"], _G["ytr"], _G["yva"]
    Xl = np.ascontiguousarray(Ztr[:, l, :])
    p = LogisticRegression(penalty="l1", C=layer_c, solver="liblinear", max_iter=400,
                           class_weight="balanced", random_state=seed)
    p.fit(Xl, ytr)
    f_tr = f1_score(ytr, p.predict(Xl), average="macro")
    f_va = f1_score(yva, p.predict(np.ascontiguousarray(Zva[:, l, :])), average="macro")
    return p.coef_[0].astype(np.float32), float(f_tr), float(f_va)


def aggregate(Z, layers, salient, alpha):
    return np.concatenate([Z[:, l, idx] * a for l, idx, a in zip(layers, salient, alpha)],
                          axis=1)


def kept_layers(L, lo):
    return [l for l in range(L) if l >= int(round(lo * (L - 1)))]


def agg_arrays(layer_c, eta, lo, alpha_mode):
    """Train / val / test features of one aggregation, re-standardized on train."""
    Ztr, Zva, Zte = _G["Ztr"], _G["Zva"], _G["Zte"]
    coefs, train_f1 = _G["probes"][layer_c]["coef"], _G["probes"][layer_c]["train_f1"]
    layers = kept_layers(Ztr.shape[1], lo)
    salient = [select_salient(coefs[l], eta) for l in layers]
    if alpha_mode == "train":
        f = np.array([train_f1[l] for l in layers])
        alpha = ((f - f.min()) / (f.max() - f.min())).tolist() if f.max() > f.min() \
            else [1.0] * len(layers)
    else:
        alpha = [1.0] * len(layers)
    Atr = aggregate(Ztr, layers, salient, alpha)
    Ava = aggregate(Zva, layers, salient, alpha)
    Ate = aggregate(Zte, layers, salient, alpha)
    mu, sd = Atr.mean(0, keepdims=True), Atr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (Atr - mu) / sd, (Ava - mu) / sd, (Ate - mu) / sd, len(layers)


def make_head(kind, hp, seed):
    if kind == "lr":
        return LogisticRegression(penalty="l2", C=hp, solver="liblinear", max_iter=4000,
                                  class_weight="balanced", random_state=seed)
    return MLPClassifier(hidden_layer_sizes=hp[0], alpha=hp[1], max_iter=800,
                         random_state=seed, early_stopping=True)


def head_one(layer_c, eta, lo, alpha_mode, h, seed):
    """One head of one aggregation: probabilities on validation and test."""
    Atr, Ava, Ate, _ = agg_arrays(layer_c, eta, lo, alpha_mode)
    clf = make_head(*HEADS[h], seed)
    clf.fit(Atr, _G["ytr"])
    return (clf.predict_proba(Ava)[:, 1].astype(np.float32),
            clf.predict_proba(Ate)[:, 1].astype(np.float32))


def n_features(layer_c, eta, lo):
    coefs = _G["probes"][layer_c]["coef"]
    return sum(len(select_salient(coefs[l], eta)) for l in kept_layers(_G["Ztr"].shape[1], lo))


def mem_left():
    """Bytes of anonymous memory this process may still take under its cgroup limits,
    or None when no limit is set."""
    left = None
    try:
        p = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().split("::")[-1].strip()
    except OSError:
        return None
    while len(p) > len("/sys/fs/cgroup"):
        try:
            mx = open(f"{p}/memory.max").read().strip()
            if mx != "max":
                stat = dict(line.split() for line in open(f"{p}/memory.stat"))
                free = int(mx) - int(stat["anon"]) - int(stat.get("shmem", 0))
                left = free if left is None else min(left, free)
        except (OSError, KeyError, ValueError):
            pass
        p = os.path.dirname(p)
    return left


def run_budgeted(fn, tasks, cost, mem, njobs, budget):
    """fn(*task) for every task in forked workers, costliest first; a task starts only
    while the estimated memory of the running tasks plus its own stays within budget
    (one task always runs). Results come back in task order."""
    todo = sorted(range(len(tasks)), key=lambda i: -cost[i])
    out, running = [None] * len(tasks), {}
    with ProcessPoolExecutor(njobs, mp_context=multiprocessing.get_context("fork")) as ex:
        while todo or running:
            used = sum(mem[i] for i in running.values())
            for i in list(todo):
                if len(running) >= njobs:
                    break
                if not running or used + mem[i] <= budget:
                    todo.remove(i)
                    running[ex.submit(fn, *tasks[i])] = i
                    used += mem[i]
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                out[running.pop(f)] = f.result()
    return out


def cfg_key(pool, r, h):
    hp = h["hp"]
    hps = str(hp) if h["head"] == "lr" else f"{hp[0][0]}-{hp[1]}"
    return f"{pool}|{r['layer_c']}|{r['eta']}|{r['lo']}|{r['alpha']}|{h['head']}|{hps}"


def cache_grid(model, feats, seed, pools, njobs, trainval, out, mem_gb=None):
    """Probabilities of the whole grid. trainval=False: fitted on the fold's train split,
    probabilities on validation and test. trainval=True: standardisation, stage-1
    probes, salient dimensions, layer weights and head all fitted on train+val,
    probabilities on test."""
    split = load_split(seed)
    ytr, _, _ = labels_and_sources(feats, split, "train")
    yva, src_va, _ = labels_and_sources(feats, split, "val")
    yte, src_te, ute = labels_and_sources(feats, split, "test")
    print(f"[multilayer-cache] {model} fold {seed} {'train+val' if trainval else 'train'} "
          f"fit: train {len(ytr)} val {len(yva)} test {len(yte)}", flush=True)
    keys, pv, pt = [], [], []
    for pool in pools:
        t0 = time.time()
        if trainval:
            Z = standardized(feats, split, pool, fit_on=("train", "val"))
            Ztv = np.concatenate([Z.pop("train"), Z.pop("val")], axis=0)
            # this fit has no validation rows; the val slots point at test so that the
            # same functions run unchanged, and nothing computed there is kept
            _G.update(Ztr=Ztv, Zva=Z["test"], Zte=Z["test"], ytr=np.concatenate([ytr, yva]),
                      yva=yte, yte=yte, src_va=src_te, src_te=src_te)
        else:
            Z = standardized(feats, split, pool)
            _G.update(Ztr=Z["train"], Zva=Z["val"], Zte=Z["test"], ytr=ytr, yva=yva,
                      yte=yte, src_va=src_va, src_te=src_te)
        L = _G["Ztr"].shape[1]
        res = Parallel(n_jobs=njobs, backend="multiprocessing")(
            delayed(layer_probe)(l, lc, seed) for lc in LAYER_CS for l in range(L))
        probes = {}
        for (lc, l), (coef, ftr, _) in zip(itertools.product(LAYER_CS, range(L)), res):
            probes.setdefault(lc, {"coef": {}, "train_f1": {}})
            probes[lc]["coef"][l] = coef
            probes[lc]["train_f1"][l] = ftr
        print(f"  [{pool}] stage 1: {len(LAYER_CS)} x {L} L1 probes in "
              f"{time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        cfgs = list(itertools.product(LAYER_CS, ETAS, LAYER_LOS, ALPHAS))
        _G["probes"] = probes
        # one task per (aggregation, head), costliest first, within a memory budget;
        # estimated peak bytes per aggregated feature: the raw and standardized float32
        # parts plus the head's working copy of the training rows
        tasks = [(*c, h, seed) for c in cfgs for h in range(len(HEADS))]
        n_tr = len(_G["ytr"])
        n_all = n_tr + len(_G["Zva"]) + len(_G["Zte"])
        nf = {c: n_features(*c[:3]) for c in cfgs}
        mem = [nf[t[:4]] * max(8 * n_all + 4 * n_tr,
                               4 * n_all + (16 if HEADS[t[4]][0] == "lr" else 8) * n_tr)
               + 2 ** 28 for t in tasks]
        cost = [nf[t[:4]] * HEAD_COST[t[4]] for t in tasks]
        left = mem_left()
        budget = 0.8 * (left if left is not None
                        else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
        if mem_gb:
            budget = min(budget, mem_gb * 2 ** 30)
        print(f"  [{pool}] stage 2: {len(tasks)} fits, budget {budget / 2 ** 30:.0f}G, "
              f"largest fit {max(mem) / 2 ** 30:.1f}G", flush=True)
        res = run_budgeted(head_one, tasks, cost, mem, njobs, budget)
        for (lc, eta, lo, am, h, _), (pva, pte) in zip(tasks, res):
            kind, hp = HEADS[h]
            keys.append(cfg_key(pool, {"layer_c": lc, "eta": eta, "lo": lo, "alpha": am},
                                {"head": kind,
                                 "hp": hp if kind == "lr" else [list(hp[0]), hp[1]]}))
            pv.append(pva)
            pt.append(pte)
        print(f"  [{pool}] stage 2: {len(cfgs)} aggregations x {len(HEADS)} heads in "
              f"{time.time() - t0:.0f}s", flush=True)
        del Z
        _G.clear()
    meta = dict(keys=np.array(keys), yva=yva.astype(np.int8), cva=src_va,
                iva=row_ids(split, "val"), yte=yte.astype(np.int8), cte=src_te,
                ite=row_ids(split, "test"), ute=np.array(ute))
    tmp = str(out)[:-4] + f".tmp{os.getpid()}.npz"      # atomic: a reader never sees half
    if trainval:
        np.savez_compressed(tmp, pte_tv=np.array(pt, dtype=np.float32), **meta)
    else:
        np.savez_compressed(tmp, pva=np.array(pv, dtype=np.float32),
                            pte=np.array(pt, dtype=np.float32), **meta)
    os.replace(tmp, out)
    print(f"[multilayer-cache] saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b-instruct")
    ap.add_argument("--seeds", type=int, nargs="+", default=FOLD_SEEDS)
    ap.add_argument("--pools", nargs="+", default=POOLS)
    ap.add_argument("--njobs", type=int, default=8)
    ap.add_argument("--trainval", action="store_true",
                    help="the train+val fit of the grid, test probabilities only")
    ap.add_argument("--mem-gb", type=float, default=None,
                    help="cap on the stage-2 memory budget")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    feats = load_feats(a.model, load_split(a.seeds[0]))
    for s in a.seeds:
        for pool in a.pools:
            name = f"multilayer_{a.model}_{'tv' if a.trainval else 'fold'}{s}_{pool}.npz"
            cache_grid(a.model, feats, s, [pool], a.njobs, a.trainval, OUT / name, a.mem_gb)
    print("MULTILAYER CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
