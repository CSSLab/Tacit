"""Hidden states of a guard at the step where it decodes its verdict word: each
trajectory is decoded greedily until the generated token is a verdict word, and every
layer's state at that step is kept.

Saves ACTS/guard_internals_<kind>/<dataset>.pt with feats["last"]: float16 [N, L+1, H],
uids and labels.

Usage: python scripts/mech/guard_internals.py --kind qwen3guard-4b --dataset tracesafe
  kinds: qwen3guard-4b, llamaguard3-8b, agentdog
  datasets: tracesafe, tracesafe-placebo, haico-pairs, atbench
"""
import argparse
import collections
import warnings

warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tacit import config as C
from tacit.activations.extract import _iter_dataset
from tacit.guards import moderation as M
from tacit.utils import modeling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["qwen3guard-4b", "llamaguard3-8b",
                                                      "agentdog"])
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--dataset", default="tracesafe")
    a = ap.parse_args()

    traces = _iter_dataset(a.dataset)
    print(f"[{a.kind}] {len(traces)} {a.dataset} trajectories", flush=True)

    if a.kind == "agentdog":
        from tacit.guards.agentdog import PROMPT, MODEL_ID, trace_to_agentdog
        model_id = MODEL_ID
        tok = AutoTokenizer.from_pretrained(model_id)

        def build(t):
            body = trace_to_agentdog(t)
            ids = tok(body, add_special_tokens=False).input_ids
            if len(ids) > a.max_len - 2048:          # head+tail
                keep = a.max_len - 2048
                body = (tok.decode(ids[: keep // 2]) + "\n[... truncated ...]\n"
                        + tok.decode(ids[-keep // 2:]))
            return tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(trajectory=body)}],
                tokenize=False, add_generation_prompt=True)
    else:
        model_id = C.MODELS[a.kind]
        tok = modeling.load_tokenizer(a.kind)
        build = lambda t: M.make_prompt(t, tok, a.max_len - 2048)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"[{a.kind}] {model_id}: {n_layers} layers, hidden {hidden}", flush=True)

    VERDICT = {"safe", "unsafe", "controversial"}
    max_steps = 6

    feats = torch.zeros(len(traces), n_layers + 1, hidden, dtype=torch.float16)
    n_trunc, at = 0, []
    with torch.inference_mode():
        for i, t in enumerate(traces):
            ids = tok(build(t), return_tensors="pt",
                      add_special_tokens=False).input_ids
            if ids.shape[1] > a.max_len:            # head+tail
                half = a.max_len // 2
                ids = torch.cat([ids[:, :half], ids[:, -half:]], dim=1)
                n_trunc += 1
            ids = ids.to("cuda")
            past, step, cur = None, -1, ids
            for k in range(max_steps):
                out = model(cur, past_key_values=past, output_hidden_states=True,
                            use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax())
                # every layer's state at the position this token is decoded from
                hs = torch.stack([h[0, -1] for h in out.hidden_states])
                if tok.decode([nxt]).strip().lower() in VERDICT:
                    feats[i], step = hs.half().cpu(), k
                    break
                cur = torch.tensor([[nxt]], device="cuda")
            if step < 0:                            # no verdict word within max_steps
                feats[i] = hs.half().cpu()
            at.append(step)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(traces)}", flush=True)
    print(f"[{a.kind}] verdict decoded at step {dict(collections.Counter(at))} "
          f"(-1 = no verdict word within {max_steps})", flush=True)

    d = C.ACTS / f"guard_internals_{a.kind}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": {"last": feats},
                "labels": torch.tensor([t.label for t in traces]),
                "uids": [t.uid for t in traces], "model": a.kind,
                "dataset": a.dataset, "n_layers": n_layers, "hidden": hidden,
                "verdict_step": at, "n_trunc": n_trunc},
               d / f"{a.dataset}.pt")
    print(f"[{a.kind}] truncated {n_trunc}; saved {d / (a.dataset + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
