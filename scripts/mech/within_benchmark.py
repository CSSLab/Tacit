"""Cross-risk transfer (AUC) inside ATBench for the three guards, over ten random half
splits: one logistic readout per risk group (CTX_AT; CNT_AT or CNT_FM) against the
benign rows, each scored on both groups.

Usage: python scripts/mech/within_benchmark.py
Writes RUNS/mech/within_benchmark.json.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tacit import config as C
from tacit.data import atbench as atbench_data
from tacit.probing.bundles import load_bundle

CTX_AT = {"tool_description_injection", "indirect_prompt_injection",
          "direct_prompt_injection", "corrupted_tool_feedback",
          "malicious_tool_execution", "unreliable_or_misinformation"}
CNT_AT = {"malicious_user_instruction_or_jailbreak"}
CNT_FM = {"instruction_for_harmful_illegal_activity", "generation_of_malicious_executables",
          "generation_of_harmful_offensive_content",
          "provide_inaccurate_misleading_or_unverified_information"}
MODELS = ["guard_internals_qwen3guard-4b", "guard_internals_llamaguard3-8b",
          "guard_internals_agentdog"]
N_SPLITS = 10


def form_groups(uids, y):
    traces = {t.uid: t for t in atbench_data.load()}

    def fms(t):
        v = t.meta["failure_mode"]
        return set(v if isinstance(v, (list, tuple)) else [v])

    ctx, cnt = [], []
    for i, u in enumerate(uids):
        if y[i] != 1 or u not in traces:
            continue
        t = traces[u]
        is_ctx = t.attack_type in CTX_AT
        is_cnt = t.attack_type in CNT_AT or bool(fms(t) & CNT_FM)
        if is_ctx and is_cnt:
            continue
        (ctx if is_ctx else cnt if is_cnt else []).append(i)
    return np.array(ctx), np.array(cnt)


def main():
    geometry = json.load(open(C.RUNS / "mech" / "directions.json"))
    out = {}
    for model in MODELS:
        bundle = load_bundle(model, "atbench")
        uids = list(bundle["uids"])
        y = np.asarray(bundle["labels"])
        X = bundle["feats"]["last"].float().numpy().astype(np.float64)
        ctx, cnt = form_groups(uids, y)
        groups = {"context": ctx, "content": cnt,
                  "benign": np.array([i for i in range(len(uids)) if y[i] == 0])}
        layers = {"context": geometry[model]["context"]["oof_peak_layer"],
                  "content": geometry[model]["content_strict"]["oof_peak_layer"]}
        runs = []
        for seed in range(N_SPLITS):
            rng = np.random.default_rng(seed)
            halves = {}
            for key, idx in groups.items():
                p = rng.permutation(idx)
                halves[key] = (p[:len(p) // 2], p[len(p) // 2:])
            scores = {}
            for axis in ("context", "content"):
                H = X[:, layers[axis], :]
                fit_rows = np.concatenate([halves[k][0] for k in
                                           ("context", "content", "benign")])
                H = H / (H[fit_rows].std(0) + 1e-6)
                unsafe_train = np.concatenate([halves["context"][0], halves["content"][0]])
                g = H[unsafe_train].mean(0) - H[halves["benign"][0]].mean(0)
                g /= np.linalg.norm(g) + 1e-12
                H = H - np.outer(H @ g, g)
                train = np.concatenate([halves[axis][0], halves["benign"][0]])
                train_y = np.concatenate([np.ones(len(halves[axis][0])),
                                          np.zeros(len(halves["benign"][0]))])
                readout = LogisticRegression(C=1.0, max_iter=3000,
                                             class_weight="balanced").fit(H[train], train_y)
                for target in ("context", "content"):
                    test = np.concatenate([halves[target][1], halves["benign"][1]])
                    test_y = np.concatenate([np.ones(len(halves[target][1])),
                                             np.zeros(len(halves["benign"][1]))])
                    scores[f"{axis}->{target}"] = float(roc_auc_score(
                        test_y, readout.decision_function(H[test])))
            runs.append({"own": (scores["context->context"]
                                  + scores["content->content"]) / 2,
                         "cross": (scores["context->content"]
                                    + scores["content->context"]) / 2,
                         **scores})
        out[model] = {"layers": layers, "n": {k: int(len(v)) for k, v in groups.items()},
                      "runs": runs,
                      "mean_own": float(np.mean([r["own"] for r in runs])),
                      "mean_cross": float(np.mean([r["cross"] for r in runs]))}
        print(model, out[model]["layers"], out[model]["n"],
              f"own {out[model]['mean_own']:.3f} cross {out[model]['mean_cross']:.3f}",
              flush=True)
    fp = C.run_dir("mech") / "within_benchmark.json"
    json.dump(out, open(fp, "w"), indent=1)
    print(f"saved {fp}")


if __name__ == "__main__":
    main()
