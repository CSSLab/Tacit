"""Cosine between each guard's output direction W_U[unsafe] - W_U[safe] and the
final-layer difference-in-means direction of each pair set, in standardized
coordinates and in multiples of 1/sqrt(d), with a pair bootstrap, split-half
reliability and a sign-flip null.

Usage: python scripts/mech/alignment.py      (CPU)
Writes RUNS/mech/alignment.json.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tacit import config as C
from tacit.utils import modeling

CONTEXT_CATS = {"MissingTypeHint", "AmbiguousArg"}
GUARDS = ["qwen3guard-4b", "llamaguard3-8b", "agentdog"]
UNSAFE_W = {"unsafe", "controversial"}
SAFE_W = {"safe"}
B_BOOT, N_SPLIT, N_NULL = 2000, 400, 2000


def category(uid):
    return uid.split("tracesafe-golden_", 1)[1].split("_", 1)[1] \
              .rsplit("-", 1)[0].removesuffix("-orig")


def readout_params(kind):
    """Final norm weight and epsilon, unembedding matrix, and tokenizer of a guard."""
    if kind == "agentdog":
        from tacit.guards.agentdog import MODEL_ID
        model_id = MODEL_ID
        tok = AutoTokenizer.from_pretrained(model_id)
    else:
        model_id = C.MODELS[kind]
        tok = modeling.load_tokenizer(kind)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                             device_map="cpu").eval()
    gamma = m.model.norm.weight.detach().float().numpy()
    eps = float(getattr(m.model.norm, "variance_epsilon", 1e-6))
    WU = m.lm_head.weight.detach().float().numpy()
    del m
    return gamma, eps, WU, tok


def arm_pairs(uids, arm, man):
    pos = {u: i for i, u in enumerate(uids)}
    if arm == "context":
        prs = [(pos[u], pos[u + "-orig"]) for u in uids
               if not u.endswith("-orig") and u + "-orig" in pos
               and category(u) in CONTEXT_CATS]
    else:
        prs = [(pos[p["unsafe"]], pos[p["safe"]]) for p in man["content"]
               if p["unsafe"] in pos and p["safe"] in pos]
    return np.array([a for a, _ in prs]), np.array([b for _, b in prs])


def main():
    man = json.load(open(C.DERIVED / "haico_pairs_manifest.json"))
    rng = np.random.default_rng(0)
    out = {}
    for kind in GUARDS:
        gamma, eps, WU, tok = readout_params(kind)
        print(f"\n[{kind}]", flush=True)
        rec = {}
        for arm, dataset in (("context", "tracesafe"), ("content", "haico-pairs")):
            b = torch.load(C.ACTS / f"guard_internals_{kind}" / f"{dataset}.pt",
                           weights_only=False)
            uids = list(b["uids"])
            H = b["feats"]["last"].float().numpy()[:, -1, :]
            del b
            Hn2 = H / np.sqrt((H ** 2).mean(1, keepdims=True) + eps) * gamma
            hit = lambda Z: float(np.mean([tok.decode([int(i)]).strip().lower()
                                           in UNSAFE_W | SAFE_W | {"cont"}
                                           for i in (Z @ WU.T).argmax(1)]))
            Hn = H if hit(H) >= hit(Hn2) else Hn2
            Lg = Hn @ WU.T
            pick = lambda ws: max([i for i in range(WU.shape[0])
                                   if tok.decode([i]).strip().lower() in ws],
                                  key=lambda i: Lg[:, i].mean())
            w = WU[pick(UNSAFE_W)] - WU[pick(SAFE_W)]

            A, Bx = arm_pairs(uids, arm, man)
            n, d = len(A), Hn.shape[1]
            sd = Hn.std(0) + 1e-6
            D = (Hn[A] - Hn[Bx]) / sd                    # per-pair difference, standardized
            wz = (w / np.linalg.norm(w)) * sd
            wz /= np.linalg.norm(wz)
            scale = np.sqrt(d)

            def align(rows, sign=None):
                m = (D[rows] * (1 if sign is None else sign[:, None])).mean(0)
                return float(m @ wz / (np.linalg.norm(m) + 1e-12)) * scale

            point = align(np.arange(n))
            boot = np.array([align(rng.integers(0, n, n)) for _ in range(B_BOOT)])
            rel = []
            for _ in range(N_SPLIT):
                p = rng.permutation(n)
                a1 = D[p[: n // 2]].mean(0)
                a2 = D[p[n // 2:]].mean(0)
                rel.append(float(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2)
                                            + 1e-12)) * scale)
            null = np.array([align(np.arange(n), rng.choice([-1.0, 1.0], n))
                             for _ in range(N_NULL)])
            rel = np.array(rel)

            rec[arm] = {
                "n_pairs": n, "point_x_chance": round(point, 2),
                "boot_mean": round(float(boot.mean()), 2),
                "boot_ci95": [round(float(np.percentile(boot, 2.5)), 2),
                              round(float(np.percentile(boot, 97.5)), 2)],
                "split_half_reliability_x_chance": round(float(rel.mean()), 2),
                "split_half_ci95": [round(float(np.percentile(rel, 2.5)), 2),
                                    round(float(np.percentile(rel, 97.5)), 2)],
                "null_abs_p95_x_chance": round(float(np.percentile(np.abs(null), 95)), 2),
                "exceeds_null": bool(abs(point) > np.percentile(np.abs(null), 95))}
            r = rec[arm]
            print(f"  {arm:8s} n={n:4d} alignment {point:+6.2f}x  boot95 "
                  f"[{r['boot_ci95'][0]:+.2f}, {r['boot_ci95'][1]:+.2f}]  "
                  f"split-half {rel.mean():6.2f}x  "
                  f"null |.|p95 {r['null_abs_p95_x_chance']:.2f}x", flush=True)
        out[kind] = rec
        del WU

    fp = C.run_dir("mech") / "alignment.json"
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\nsaved {fp}", flush=True)


if __name__ == "__main__":
    main()
