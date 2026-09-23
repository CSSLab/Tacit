"""AgentDoG-4B on every row of the seven sources, with greedy decoding (prompt:
tacit.guards.agentdog). The score is the unsafe share of the first generated token's
probability mass.
Writes RUNS/guards/agentdog/all.json, read by score_guards.py.

Usage: python scripts/guards/run_agentdog.py
"""
import argparse
import json
import math

import numpy as np
from sklearn.metrics import f1_score

from tacit import config as C
from tacit.activations.extract import _iter_dataset
from tacit.guards.agentdog import MODEL_ID, build_prompt
from tacit.suite import BENCHES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--datasets", nargs="+", default=BENCHES)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=a.max_len, enforce_eager=True)
    sp = SamplingParams(max_tokens=8, temperature=0.0, logprobs=20)

    results = {}
    for ds in a.datasets:
        tr = list(_iter_dataset(ds))
        prompts, n_trunc = [], 0
        for t in tr:
            text, cut = build_prompt(tok, t, a.max_len)
            prompts.append(text)
            n_trunc += int(cut)
        outs = llm.generate(prompts, sp)
        preds, scores, n_parse_fail = [], [], 0
        for o in outs:
            txt = o.outputs[0].text.strip().lower()
            if txt.startswith("unsafe"):
                pred = 1
            elif txt.startswith("safe"):
                pred = 0
            else:
                n_parse_fail += 1
                pred = 1 if "unsafe" in txt else 0
            preds.append(pred)
            lps = o.outputs[0].logprobs
            um = sm = 0.0
            if lps:
                for lp in lps[0].values():
                    t = (getattr(lp, "decoded_token", "") or "").strip().lower()
                    if not t:
                        continue
                    if t.startswith("unsafe") or "unsafe".startswith(t):
                        um += math.exp(lp.logprob)
                    elif t.startswith("safe") or "safe".startswith(t):
                        sm += math.exp(lp.logprob)
            scores.append(um / (um + sm) if um + sm > 0 else float(pred))
        y = [t.label for t in tr]
        f1 = f1_score(y, preds, average="macro")
        results[ds] = {"macro_f1": round(float(f1), 4), "n": len(y),
                       "n_unsafe": int(sum(y)), "n_trunc": n_trunc,
                       "n_parse_fail": n_parse_fail,
                       "preds": preds, "labels": [int(v) for v in y],
                       "scores": [round(float(s), 6) for s in scores],
                       "uids": [t.uid for t in tr]}
        print(f"[agentdog] {ds:20s} F1={f1:.4f} n={len(y)} trunc={n_trunc} "
              f"parse_fail={n_parse_fail}", flush=True)

    od = C.run_dir("guards", "agentdog")
    json.dump(results, open(od / "all.json", "w"))
    m = float(np.mean([results[b]["macro_f1"] for b in results]))
    print(f"[agentdog] mean over sources={m:.4f}\nsaved {od / 'all.json'}", flush=True)


if __name__ == "__main__":
    main()
