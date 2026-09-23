"""Output scores of a guard on every trajectory of a pair set: the unsafe share of the
probability mass at the first generated token that is a verdict word (Controversial
counts as unsafe).

Usage: python scripts/mech/pair_scores.py --kind qwen3guard-4b --dataset tracesafe
  kinds: qwen3guard-4b, llamaguard3-8b, agentdog; datasets: tracesafe, haico-pairs
Writes RUNS/mech/pair_scores_<dataset>_<kind>.json.
"""
import argparse
import json
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from tacit import config as C
from tacit.activations.extract import _iter_dataset
from tacit.guards import moderation as M
from tacit.utils import modeling

UNSAFE_TOK = {"unsafe", "controversial"}
SAFE_TOK = {"safe"}


def verdict_p_unsafe(out, fallback: int) -> tuple[float, int]:
    """Unsafe share at the first generated position whose token is a verdict word."""
    o = out.outputs[0]
    lps, tids = o.logprobs or [], o.token_ids
    verdicts = UNSAFE_TOK | SAFE_TOK
    for i, dist in enumerate(lps):
        chosen = dist.get(tids[i])
        tok = (getattr(chosen, "decoded_token", "") or "").strip().lower()
        if tok not in verdicts:
            continue
        um = sm = 0.0
        for lp in dist.values():
            t = (getattr(lp, "decoded_token", "") or "").strip().lower()
            if t in UNSAFE_TOK:
                um += math.exp(lp.logprob)
            elif t in SAFE_TOK:
                sm += math.exp(lp.logprob)
        return (um / (um + sm) if um + sm > 0 else float(fallback)), i
    return float(fallback), -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["qwen3guard-4b", "llamaguard3-8b",
                                                      "agentdog"])
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--dataset", default="tracesafe", choices=["tracesafe", "haico-pairs"])
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    traces = _iter_dataset(a.dataset)
    y = [t.label for t in traces]
    print(f"[{a.kind}] scoring all {len(traces)} {a.dataset} trajectories", flush=True)

    if a.kind == "agentdog":
        from transformers import AutoTokenizer
        from tacit.guards.agentdog import MODEL_ID, build_prompt
        model_id, tok = MODEL_ID, AutoTokenizer.from_pretrained(MODEL_ID)
        build = lambda t: build_prompt(tok, t, a.max_len)[0]
        parse = lambda txt: (1 if txt.strip().lower().startswith("unsafe") else 0)
    else:
        model_id = C.MODELS[a.kind]
        tok = modeling.load_tokenizer(a.kind)
        build = lambda t: M.make_prompt(t, tok, a.max_len - 2048)
        parse = lambda txt: M.parse_verdict(a.kind, txt)

    # max_tokens stays at the guards' 16 so that parse_verdict sees the full response
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=a.max_len, enforce_eager=True, max_logprobs=600)
    sp = SamplingParams(max_tokens=16, temperature=0.0, logprobs=600)

    outs = llm.generate([build(t) for t in traces], sp)
    preds = [parse(o.outputs[0].text) for o in outs]
    scored = [verdict_p_unsafe(o, p) for o, p in zip(outs, preds)]
    scores = [s for s, _ in scored]
    at = [i for _, i in scored]
    n_fail = sum(i < 0 for i in at)
    # the score must agree with the parsed verdict, else the wrong position was read
    agree = float(np.mean([(s > 0.5) == bool(p) for s, p in zip(scores, preds)]))
    print(f"[{a.kind}] verdict token not found in {n_fail}/{len(at)}; "
          f"(score>0.5) agrees with the parsed verdict on {agree:.4f}", flush=True)

    rec = {"kind": a.kind, "dataset": a.dataset, "n": len(traces),
           "macro_f1": round(float(f1_score(y, preds, average="macro")), 4),
           "auc": round(float(roc_auc_score(y, scores)), 4),
           "n_no_verdict_token": int(n_fail),
           "verdict_agreement": round(agree, 4),
           "verdict_positions": [int(i) for i in at],
           "uids": [t.uid for t in traces], "labels": [int(v) for v in y],
           "preds": [int(p) for p in preds],
           "scores": [round(float(s), 6) for s in scores]}
    fp = C.run_dir("mech") / f"pair_scores_{a.dataset}_{a.kind}.json"
    json.dump(rec, open(fp, "w"), indent=1)
    print(f"saved {fp}", flush=True)


if __name__ == "__main__":
    main()
