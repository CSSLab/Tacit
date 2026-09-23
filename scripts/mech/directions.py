"""Direction geometry of the pair sets for two backbones and three guards.

Per layer: the cosine between the TraceSafe and HAICOSYSTEM difference-in-means
directions with a pair bootstrap, split-half cosines within each set, and each
direction's rank accuracy on every pair set. Both sets are scaled by one per-dimension
standard deviation over their union. Also records each set's out-of-fold rank-accuracy
curve and peak layer (read by within_benchmark.py).

Usage: python scripts/mech/directions.py
Writes RUNS/mech/directions.json.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import GroupKFold

from tacit import config as C
from tacit.probing.bundles import load_bundle

CONTEXT_CATS = {"MissingTypeHint", "AmbiguousArg"}
PAIR_RENDER = "text"
MODELS = [
    "guard_internals_qwen3guard-4b",
    "guard_internals_llamaguard3-8b",
    "guard_internals_agentdog",
    "qwen3-4b-instruct",
    "llama3.1-8b-instruct",
]


def category(uid):
    return uid.split("tracesafe-golden_", 1)[1].split("_", 1)[1] \
              .rsplit("-", 1)[0].removesuffix("-orig")


def rank_acc(sa, sb):
    v = np.sign(np.asarray(sa, float) - np.asarray(sb, float))
    return float((v > 0).mean() + 0.5 * (v == 0).mean())


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    man = json.load(open(C.DERIVED / "haico_pairs_manifest.json"))
    out = {}
    for model in MODELS:
        ts = load_bundle(model, "tracesafe", render=PAIR_RENDER)
        hp = load_bundle(model, "haico-pairs", render=PAIR_RENDER)
        tu = {u: i for i, u in enumerate(ts["uids"])}
        hu = {u: i for i, u in enumerate(hp["uids"])}
        Xt = ts["feats"]["last"].float().numpy()
        Xh = hp["feats"]["last"].float().numpy()

        ctx = [(tu[u], tu[u + "-orig"]) for u in ts["uids"]
               if not u.endswith("-orig") and u + "-orig" in tu
               and category(u) in CONTEXT_CATS]
        At = np.array([a for a, _ in ctx])
        Bt = np.array([b for _, b in ctx])
        arms = {"context": ("ts", At, Bt)}
        for key, sel in (("content_strict", "content"), ("content_broad", "any")):
            ok = [p for p in man[sel] if p["unsafe"] in hu and p["safe"] in hu]
            arms[key] = ("hp", np.array([hu[p["unsafe"]] for p in ok]),
                         np.array([hu[p["safe"]] for p in ok]))
        # pair-level bootstrap weights for the cross-risk cosine, shared across layers
        boot_rng = np.random.default_rng(2026)
        n_boot = 300
        boot_w = {
            k: boot_rng.poisson(1.0, size=(n_boot, len(arms[k][1]))).astype(np.float32)
            for k in ("context", "content_strict")
        }
        # five fixed disjoint half splits per risk for the same-risk control
        split_rng = np.random.default_rng(31415)
        split_idx = {}
        for k in ("context", "content_strict"):
            n = len(arms[k][1])
            split_idx[k] = []
            for _ in range(5):
                idx = split_rng.permutation(n)
                split_idx[k].append((idx[:n // 2], idx[n // 2:]))
        L = Xt.shape[1]
        rec = {k: {"n": len(v[1])} for k, v in arms.items()}
        rec["layers"] = {}
        for l in range(L):
            a = Xt[:, l, :].astype(np.float64)
            b = Xh[:, l, :].astype(np.float64)
            sd = np.concatenate([a, b], 0).std(0) + 1e-6      # common scaling
            H = {"ts": a / sd, "hp": b / sd}
            dirs, boot_dirs, diffs = {}, {}, {}
            for k, (src, A, B) in arms.items():
                D = (H[src][A] - H[src][B]).astype(np.float32)
                diffs[k] = D
                dirs[k] = unit(D.mean(0))
                if k in boot_w:
                    bd = boot_w[k] @ D
                    boot_dirs[k] = bd / (np.linalg.norm(bd, axis=1, keepdims=True) + 1e-12)
            split_cos = {}
            for k in ("context", "content_strict"):
                split_cos[k] = [
                    float(unit(diffs[k][i].mean(0)) @ unit(diffs[k][j].mean(0)))
                    for i, j in split_idx[k]
                ]
            split_mean = [
                (split_cos["context"][i] + split_cos["content_strict"][i]) / 2
                for i in range(5)
            ]
            row = {
                "reliability": {k: round(float(np.mean(v)), 4)
                                for k, v in split_cos.items()},
                "same_risk_split": {
                    "context": [round(x, 4) for x in split_cos["context"]],
                    "content_strict": [round(x, 4) for x in split_cos["content_strict"]],
                    "mean": [round(x, 4) for x in split_mean],
                },
                "cos": {}, "cos_boot_ci95": {}, "transfer": {}
            }
            for k1 in arms:
                for k2 in arms:
                    if k1 >= k2:
                        continue
                    row["cos"][f"{k1}|{k2}"] = round(float(dirs[k1] @ dirs[k2]), 4)
            boot_cos = np.sum(boot_dirs["context"] * boot_dirs["content_strict"], axis=1)
            row["cos_boot_ci95"]["content_strict|context"] = [
                round(float(np.quantile(boot_cos, 0.025)), 4),
                round(float(np.quantile(boot_cos, 0.975)), 4),
            ]
            for kd, v in dirs.items():                          # direction fitted on kd
                for kt, (src, A, B) in arms.items():            # scored on kt's pairs
                    row["transfer"][f"{kd}->{kt}"] = round(
                        rank_acc(H[src][A] @ v, H[src][B] @ v), 4)
            rec["layers"][l] = row

        # out-of-fold rank accuracy per risk, under the same common scaling
        for k, (src, A, B) in arms.items():
            folds = list(GroupKFold(5).split(np.arange(len(A)), groups=np.arange(len(A))))
            prof = []
            for l in range(L):
                src_x = Xt if src == "ts" else Xh
                a = Xt[:, l, :].astype(np.float64)
                b = Xh[:, l, :].astype(np.float64)
                sd = np.concatenate([a, b], 0).std(0) + 1e-6
                Hl = (src_x[:, l, :].astype(np.float64)) / sd
                oof = np.zeros(len(A))
                for tr, te in folds:
                    v = unit((Hl[A[tr]] - Hl[B[tr]]).mean(0))
                    oof[te] = Hl[A[te]] @ v - Hl[B[te]] @ v
                prof.append(round(rank_acc(oof, np.zeros(len(A))), 4))
            pk = int(np.argmax(prof))
            rec[k].update({"oof_curve": prof, "oof_peak": prof[pk], "oof_peak_layer": pk})

        out[model] = rec
        for k in arms:
            print(f"[{model}] {k:15s} n={rec[k]['n']:4d} oof peak {rec[k]['oof_peak']:.3f}"
                  f"@{rec[k]['oof_peak_layer']}", flush=True)

    fp = C.run_dir("mech") / "directions.json"
    json.dump(out, open(fp, "w"), indent=1)
    print(f"saved {fp}", flush=True)


if __name__ == "__main__":
    main()
