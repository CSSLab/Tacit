"""Qwen3Guard-4B or LlamaGuard3-8B on every row of the seven sources, with greedy
decoding (prompts and parsing: tacit.guards.moderation).
Writes RUNS/guards/<kind>/all.json, read by score_guards.py.

Usage: python scripts/guards/run_moderation_guard.py --kind qwen3guard-4b
       python scripts/guards/run_moderation_guard.py --kind llamaguard3-8b
"""
import argparse
import json

from sklearn.metrics import f1_score

from tacit import config as C
from tacit.activations.extract import _iter_dataset
from tacit.guards import moderation as M
from tacit.suite import BENCHES
from tacit.utils import modeling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["qwen3guard-4b", "llamaguard3-8b"])
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--budget-split", default="need", choices=["need", "even"],
                    help="see tacit.guards.moderation.make_prompt")
    ap.add_argument("--datasets", nargs="+", default=BENCHES)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    tok = modeling.load_tokenizer(a.kind)
    llm = LLM(model=C.MODELS[a.kind], dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=a.max_len,
              tensor_parallel_size=a.tp, enforce_eager=True)
    sp = SamplingParams(max_tokens=16, temperature=0.0)

    results = {}
    for ds in a.datasets:
        tr = list(_iter_dataset(ds))
        stats = {}
        prompts = [M.make_prompt(t, tok, a.max_len - 2048, a.budget_split, stats)
                   for t in tr]
        out = llm.generate(prompts, sp)
        preds = [M.parse_verdict(a.kind, o.outputs[0].text) for o in out]
        y = [t.label for t in tr]
        macro = round(float(f1_score(y, preds, average="macro")), 4)
        results[ds] = {"macro_f1": macro, "n": len(y), "n_unsafe": int(sum(y)),
                       "preds": [int(p) for p in preds], "labels": [int(v) for v in y],
                       "uids": [t.uid for t in tr], "max_len": a.max_len,
                       "budget_split": a.budget_split,
                       "n_truncated": stats.get("truncated", 0)}
        print(f"[{a.kind}/{ds}] macro-F1={macro:.4f} (n={len(y)})", flush=True)

    out_p = C.run_dir("guards", a.kind) / "all.json"
    json.dump(results, open(out_p, "w"), indent=1)
    print(f"saved {out_p}", flush=True)


if __name__ == "__main__":
    main()
