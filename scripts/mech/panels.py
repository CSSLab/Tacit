"""Per-layer rank accuracy on the pair sets for two backbones and three guards, and the
guards' output scores on the same pairs (pair_scores.py).

Pair sets: ``context``, the TraceSafe MissingTypeHint and AmbiguousArg pairs;
``content_strict`` and ``content_broad``, the HAICOSYSTEM pairs of
scripts/data/build_haico_pairs.py (manifest keys ``content`` and ``any``). Each curve
uses the last-token state and an out-of-fold difference-in-means direction (five folds).

Usage: python scripts/mech/panels.py
Writes RUNS/mech/panels.json.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from tacit import config as C
from tacit.probing.bundles import load_bundle
from tacit.utils.modeling import trace_to_plaintext

CONTEXT_CATS = {"MissingTypeHint", "AmbiguousArg"}
PAIR_RENDER = "text"
SYSTEMS = [  # display name, activation source, output-score stem, role
    ("Qwen3-4B-Instruct", "bundle:qwen3-4b-instruct", None, "backbone"),
    ("Llama-3.1-8B-Instruct", "bundle:llama3.1-8b-instruct", None, "backbone"),
    ("Qwen3Guard-4B", "guard:qwen3guard-4b", "qwen3guard-4b", "guard"),
    ("LlamaGuard3-8B", "guard:llamaguard3-8b", "llamaguard3-8b", "guard"),
    ("AgentDoG-4B", "guard:agentdog", "agentdog", "guard"),
]


def category(uid):
    return uid.split("tracesafe-golden_", 1)[1].split("_", 1)[1] \
              .rsplit("-", 1)[0].removesuffix("-orig")


def rank_acc(sa, sb):
    v = np.sign(np.asarray(sa, float) - np.asarray(sb, float))
    return float((v > 0).mean() + 0.5 * (v == 0).mean())


def feats(src, dataset):
    kind, key = src.split(":", 1)
    if kind == "bundle":
        b = load_bundle(key, dataset, render=PAIR_RENDER)
        return list(b["uids"]), b["feats"]["last"].float().numpy()
    b = torch.load(C.ACTS / f"guard_internals_{key}" / f"{dataset}.pt", weights_only=False)
    return list(b["uids"]), b["feats"]["last"].float().numpy()


def curve(X, A, B):
    """Out-of-fold rank accuracy of the difference-in-means direction at every layer."""
    n = len(A)
    folds = list(GroupKFold(5).split(np.arange(n), groups=np.arange(n)))
    prof = []
    for l in range(X.shape[1]):
        H = X[:, l, :].astype(np.float64)
        H = H / (H.std(0) + 1e-6)
        oof = np.zeros(n)
        for tr, te in folds:
            v = (H[A[tr]] - H[B[tr]]).mean(0)
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            v /= nv
            oof[te] = H[A[te]] @ v - H[B[te]] @ v
        prof.append(round(float((oof > 0).mean() + 0.5 * (oof == 0).mean()), 4))
    return prof


def context_pairs(uids):
    pos = {u: i for i, u in enumerate(uids)}
    P = [(pos[u], pos[u + "-orig"]) for u in uids
         if not u.endswith("-orig") and u + "-orig" in pos and category(u) in CONTEXT_CATS]
    return np.array([a for a, _ in P]), np.array([b for _, b in P])


def main():
    man = json.load(open(C.DERIVED / "haico_pairs_manifest.json"))
    panels = {"context": ("tracesafe", None),
              "content_strict": ("haico-pairs", man["content"]),
              "content_broad": ("haico-pairs", man["any"])}

    out = {"panels": {}, "systems": {}}
    from tacit.data import haico_pairs as HP
    hp = {t.uid: trace_to_plaintext(t) for t in HP.load()}
    for pname, (ds, pairs) in panels.items():
        if pairs is None:
            continue
        jac, rat = [], []
        for p in pairs:
            wa, wb = set(hp[p["unsafe"]].lower().split()), set(hp[p["safe"]].lower().split())
            jac.append(len(wa & wb) / max(len(wa | wb), 1))
            rat.append(len(hp[p["unsafe"]]) / max(len(hp[p["safe"]]), 1))
        out["panels"][pname] = {"dataset": ds, "n": len(pairs),
                                "word_jaccard_median": round(float(np.median(jac)), 4),
                                "char_ratio_median": round(float(np.median(rat)), 4)}

    for name, src, vstem, role in SYSTEMS:
        rec = {"role": role, "curves": {}, "verdicts": {}}
        for pname, (ds, pairs) in panels.items():
            try:
                uids, X = feats(src, ds)
            except FileNotFoundError:
                print(f"  [{name}] missing bundle for {ds}, skipped", flush=True)
                continue
            if pairs is None:
                A, B = context_pairs(uids)
            else:
                pos = {u: i for i, u in enumerate(uids)}
                ok = [p for p in pairs if p["unsafe"] in pos and p["safe"] in pos]
                A = np.array([pos[p["unsafe"]] for p in ok])
                B = np.array([pos[p["safe"]] for p in ok])
            rec["curves"][pname] = curve(X, A, B)
            out["panels"].setdefault(pname, {})["n"] = len(A)
            if vstem:
                fp = C.RUNS / "mech" / f"pair_scores_{ds}_{vstem}.json"
                if fp.exists():
                    d = json.load(open(fp))
                    sc = dict(zip(d["uids"], d["scores"]))
                    rec["verdicts"][pname] = round(
                        rank_acc([sc[uids[i]] for i in A], [sc[uids[i]] for i in B]), 4)
            pr = rec["curves"][pname]
            pk = int(np.argmax(pr))
            print(f"{name:24s} {pname:15s} n={len(A):4d} "
                  f"peak={pr[pk]:.3f}@{pk}/{len(pr)-1} final={pr[-1]:.3f}"
                  + (f" output={rec['verdicts'][pname]:.3f}"
                     if pname in rec["verdicts"] else ""), flush=True)
        out["systems"][name] = rec

    fp = C.run_dir("mech") / "panels.json"
    json.dump(out, open(fp, "w"), indent=1)
    print(f"saved {fp}", flush=True)


if __name__ == "__main__":
    main()
