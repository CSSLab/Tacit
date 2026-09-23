from __future__ import annotations
import argparse
import dataclasses
import json
import math

import numpy as np
from sklearn.preprocessing import StandardScaler

from tacit import config as C
from tacit.activations.extract import _iter_dataset
from tacit.features import load_feats, stack
from tacit.guards import moderation as M
from tacit.suite import BENCHES, load_split
from tacit.utils import modeling
from scripts.readout.probe_cache import fit_probe, weights

MODEL, TV_SEED, MAX_LEN = "qwen3-4b-instruct", 42, 16384
CASES = [("r-judge", "rjudge-2027", "r-judge"),
         ("assebench-safety", "assebench-safety-1945", "assebench")]


def pick():
    d = json.load(open(C.RUNS / "readout" / "readout_probe_qwen3-4b.json"))
    rows = d["test_rows"]
    pool, layer, c = d["members"][0].split("|")
    return ({"pool": pool, "layer": int(layer), "C": float(c)},
            dict(zip(rows["uids"], rows["probs"])))


def prefixes(trace):
    return [dataclasses.replace(trace, turns=trace.turns[:k])
            for k in range(1, len(trace.turns) + 1)]


def probe_scores():
    import torch
    sel, cached = pick()
    split = load_split(TV_SEED)
    model, tok = modeling.load_model_and_tokenizer(MODEL)
    model.eval()
    feats = load_feats(MODEL, split, benches=BENCHES)
    parts = [stack(feats, split, sel["pool"], sel["layer"], p) for p in ("train", "val")]
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    sc = StandardScaler().fit(X)
    clf = fit_probe(sc.transform(X), y, weights(y, np.concatenate([p[2] for p in parts])),
                    sel["C"], TV_SEED)
    out = {}
    for ds, uid, col in CASES:
        trace = {str(t.uid): t for t in _iter_dataset(ds)}[uid]
        rows = []
        for k, pre in enumerate(prefixes(trace), start=1):
            rendered = modeling.render(tok, pre, rich_tools=True)
            ids, _ = modeling.encode_truncated(tok, rendered, MAX_LEN)
            x = ids[None].to(model.device)
            with torch.no_grad():
                hs = model(x, attention_mask=torch.ones_like(x), output_hidden_states=True,
                           use_cache=False).hidden_states
            h = hs[sel["layer"]][0]
            v = {"last": h[-1], "mean": h.mean(0),
                 "both": torch.cat([h.mean(0), h[-1]])}[sel["pool"]].to(torch.float16)
            p = float(clf.predict_proba(sc.transform(v[None].float().cpu().numpy()))[0, 1])
            t = pre.turns[-1]
            rows.append({"k": k, "role": t.role, "score": round(p, 6),
                         "text": " | ".join(x for x in [t.thought, t.action, t.content] if x)})
        ref = cached[uid]
        meta = {"pick": sel, "cached_test_prob": ref, "full_prefix_score": rows[-1]["score"],
                "delta_vs_cached": rows[-1]["score"] - ref}
        if abs(meta["delta_vs_cached"]) > 0.03 or (rows[-1]["score"] >= 0.5) != (ref >= 0.5):
            raise RuntimeError(f"{uid}: prefix {rows[-1]['score']} != cached {ref}")
        out[uid] = {"dataset": ds, "column": col, "label": int(trace.label),
                    "rows": rows, **meta}
        print(f"[probe] {uid} " + " ".join(f"{r['role'][:5]}={r['score']:.3f}" for r in rows),
              flush=True)
    return out


def dog_score(o):
    lps = o.outputs[0].logprobs
    if not lps:
        return None
    u = s = 0.0
    for lp in lps[0].values():
        t = (getattr(lp, "decoded_token", "") or "").strip().lower()
        if t and "unsafe".startswith(t):
            u += math.exp(lp.logprob)
        elif t and "safe".startswith(t):
            s += math.exp(lp.logprob)
    return None if u + s == 0 else u / (u + s)


def guard_scores(kind):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from tacit.guards.agentdog import MODEL_ID as DOG_ID, PROMPT as DOG_PROMPT
    from tacit.guards.agentdog import trace_to_agentdog

    model_id = DOG_ID if kind == "agentdog" else C.MODELS[kind]
    gtok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    todo, index = [], []
    for ds, uid, _col in CASES:
        trace = {str(t.uid): t for t in _iter_dataset(ds)}[uid]
        for k, pre in enumerate(prefixes(trace), start=1):
            if kind == "agentdog":
                prompt = gtok.apply_chat_template(
                    [{"role": "user", "content": DOG_PROMPT.format(
                        trajectory=trace_to_agentdog(pre))}],
                    tokenize=False, add_generation_prompt=True)
            else:
                prompt = M.make_prompt(pre, gtok, MAX_LEN - 2048, "need", {})
            todo.append(prompt)
            index.append((uid, k, pre.turns[-1].role, len(trace.turns)))
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=MAX_LEN, enforce_eager=True)
    outs = llm.generate(todo, SamplingParams(max_tokens=(8 if kind == "agentdog" else 16),
                                             temperature=0.0, logprobs=20))
    ref = json.load(open(C.RUNS / "guards" / kind / "all.json"))
    out = {}
    for (uid, k, role, n), o in zip(index, outs):
        text = o.outputs[0].text
        if kind == "agentdog":
            pred = 1 if text.strip().lower().startswith("unsafe") else (
                0 if text.strip().lower().startswith("safe") else int("unsafe" in text.lower()))
            raw = dog_score(o)
        else:
            pred = M.parse_verdict(kind, text)
            raw = M.verdict_score(kind, o)
        score = float(pred) if raw is None else float(raw)
        r = out.setdefault(uid, {"rows": []})
        r["rows"].append({"k": k, "role": role, "score": round(score, 6), "pred": int(pred)})
        if k == n:
            ds = next(d for d, u, _ in CASES if u == uid)
            rec = dict(zip([str(x) for x in ref[ds]["uids"]], ref[ds]["preds"]))[uid]
            r["recorded_pred"] = int(rec)
            if int(pred) != int(rec):
                raise RuntimeError(f"{kind} {uid}: final {pred} != recorded {rec}")
    for uid, r in out.items():
        print(f"[{kind}] {uid} " + " ".join(f"{x['score']:.3f}" for x in r["rows"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["probe", "agentdog", "llamaguard3-8b", "qwen3guard-4b"])
    a = ap.parse_args()
    res = probe_scores() if a.kind == "probe" else guard_scores(a.kind)
    p = C.run_dir("case_study") / f"prefix_scores_{a.kind}.json"
    json.dump({"kind": a.kind, "max_len": MAX_LEN, "cases": res}, open(p, "w"), indent=1)
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
