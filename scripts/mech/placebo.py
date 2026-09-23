"""Placebo control on Qwen3-4B: an out-of-fold difference-in-means direction fitted on
the TraceSafe pairs scores the unsafe, benign and placebo members
(scripts/data/build_tracesafe_placebo.py).

Usage: python scripts/mech/placebo.py
Writes RUNS/mech/placebo.json.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import GroupKFold

from tacit import config as C
from tacit.probing.bundles import load_bundle

CATS = {"MissingTypeHint", "AmbiguousArg"}
PAIR_RENDER = "text"


def category(uid):
    return uid.split("tracesafe-golden_", 1)[1].split("_", 1)[1] \
              .rsplit("-", 1)[0].removesuffix("-orig").removesuffix("-placebo")


def rank_acc(sa, sb):
    v = np.sign(np.asarray(sa, float) - np.asarray(sb, float))
    return float((v > 0).mean() + 0.5 * (v == 0).mean())


def main():
    b = load_bundle("qwen3-4b-instruct", "tracesafe", render=PAIR_RENDER)
    uids = list(b["uids"])
    pos = {u: i for i, u in enumerate(uids)}
    X = b["feats"]["last"].float().numpy()
    L = X.shape[1]

    p = load_bundle("qwen3-4b-instruct", "tracesafe-placebo", render=PAIR_RENDER)
    puid = list(p["uids"])
    P = p["feats"]["last"].float().numpy()

    # (unsafe, benign, placebo built from that benign member, category)
    rows = []
    for u in uids:
        if u.endswith("-orig") or u + "-orig" not in pos or category(u) not in CATS:
            continue
        pu = u + "-placebo"
        if pu in puid:
            rows.append((pos[u], pos[u + "-orig"], puid.index(pu), category(u)))
    A = np.array([r[0] for r in rows])
    B = np.array([r[1] for r in rows])
    Q = np.array([r[2] for r in rows])
    cats = np.array([r[3] for r in rows])
    n = len(rows)
    print(f"aligned members: {n} (unsafe / benign / placebo)", flush=True)

    folds = list(GroupKFold(5).split(np.arange(n), groups=np.arange(n)))
    out = {"n": n, "layers": {}}
    for l in range(L):
        sd = X[:, l, :].astype(np.float64).std(0) + 1e-6
        H = X[:, l, :].astype(np.float64) / sd
        G = P[:, l, :].astype(np.float64) / sd
        sa, sb, sq = np.zeros(n), np.zeros(n), np.zeros(n)
        for tr, te in folds:
            v = (H[A[tr]] - H[B[tr]]).mean(0)
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            v /= nv
            sa[te], sb[te], sq[te] = H[A[te]] @ v, H[B[te]] @ v, G[Q[te]] @ v
        out["layers"][l] = {
            "attacked_vs_benign": round(rank_acc(sa, sb), 4),
            "placebo_vs_benign": round(rank_acc(sq, sb), 4),
            "attacked_vs_placebo": round(rank_acc(sa, sq), 4)}

    pk = max(out["layers"], key=lambda l: out["layers"][l]["attacked_vs_benign"])
    out["peak_layer"] = pk
    r = out["layers"][pk]
    print(f"peak layer {pk}: unsafe>benign {r['attacked_vs_benign']:.3f} | "
          f"placebo>benign {r['placebo_vs_benign']:.3f} | "
          f"unsafe>placebo {r['attacked_vs_placebo']:.3f}", flush=True)
    for c in sorted(CATS):
        s = cats == c
        sd = X[:, pk, :].astype(np.float64).std(0) + 1e-6
        H = X[:, pk, :].astype(np.float64) / sd
        G = P[:, pk, :].astype(np.float64) / sd
        sa, sb, sq = np.zeros(n), np.zeros(n), np.zeros(n)
        for tr, te in folds:
            v = (H[A[tr]] - H[B[tr]]).mean(0)
            v /= (np.linalg.norm(v) + 1e-12)
            sa[te], sb[te], sq[te] = H[A[te]] @ v, H[B[te]] @ v, G[Q[te]] @ v
        out.setdefault("per_category", {})[c] = {
            "n": int(s.sum()),
            "attacked_vs_benign": round(rank_acc(sa[s], sb[s]), 4),
            "placebo_vs_benign": round(rank_acc(sq[s], sb[s]), 4),
            "attacked_vs_placebo": round(rank_acc(sa[s], sq[s]), 4)}
        print(f"  [{c}] {out['per_category'][c]}", flush=True)

    fp = C.run_dir("mech") / "placebo.json"
    json.dump(out, open(fp, "w"), indent=1)
    print(f"saved {fp}", flush=True)


if __name__ == "__main__":
    main()
