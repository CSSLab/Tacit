"""Activation extraction.

Each rendered trajectory goes through the model in one forward pass (no generation);
every layer's hidden states are pooled over tokens and cached as
ACTS/<model>/<dataset>[_rich].pt with
  feats[pool]: float16 [N, L+1, H]   (L+1 includes the embedding layer)
  labels:      int64  [N]
  uids:        list[str]
  n_tokens:    int64  [N]

Usage: python -m tacit.activations.extract --model qwen3-4b-instruct --dataset r-judge
       python -m tacit.activations.extract --model qwen3-4b-instruct --dataset tracesafe \
           --render rich
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from .. import config as C


def _iter_dataset(name: str):
    if name == "r-judge":
        from ..data import rjudge
        return rjudge.load()
    if name == "tracesafe":
        from ..data import tracesafe
        return tracesafe.load()
    if name == "atbench":
        from ..data import atbench
        return atbench.load()
    if name.startswith("assebench-"):
        from ..data import assebench
        return assebench.load(name.split("-", 1)[1])
    if name == "oas":
        from ..data import openagentsafety
        return openagentsafety.load()
    if name == "agentdojo":
        from ..data import agentdojo
        return agentdojo.load()
    if name == "haico-pairs":
        from ..data import haico_pairs
        return haico_pairs.load()
    if name == "tracesafe-placebo":
        from ..data import tracesafe_placebo
        return tracesafe_placebo.load()
    raise ValueError(f"unknown dataset {name!r}")


def extract(model_key: str, dataset: str, max_length: int = 16384,
            pools: tuple[str, ...] = ("mean", "last"), dtype: str = "bfloat16",
            render_mode: str = "text") -> Path:
    import torch
    from ..utils import modeling

    traces = _iter_dataset(dataset)
    model, tok = modeling.load_model_and_tokenizer(model_key, dtype=dtype)
    cfg = model.config
    n_layers = cfg.num_hidden_layers + 1
    hidden = cfg.hidden_size
    print(f"[extract] model={model_key} layers={n_layers} H={hidden} "
          f"dataset={dataset} n={len(traces)}", flush=True)

    feats = {p: torch.zeros(len(traces), n_layers, hidden, dtype=torch.float16)
             for p in pools}
    labels = torch.zeros(len(traces), dtype=torch.long)
    n_tokens = torch.zeros(len(traces), dtype=torch.long)
    uids: list[str] = []
    n_trunc = 0
    t0 = time.time()

    rich = render_mode == "rich"
    for i, tr in enumerate(traces):
        text = modeling.render(tok, tr, rich_tools=rich)
        ids, trunc = modeling.encode_truncated(tok, text, max_length)
        n_trunc += int(trunc)
        input_ids = ids.unsqueeze(0).to(model.device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple[L+1] of [1, T, H]
        for p in pools:
            if p == "mean":
                v = torch.stack([h[0].mean(0) for h in hs])
            elif p == "last":
                v = torch.stack([h[0, -1] for h in hs])
            else:
                raise ValueError(p)
            feats[p][i] = v.to(torch.float16).cpu()
        labels[i] = tr.label
        n_tokens[i] = ids.shape[0]
        uids.append(tr.uid)
        if (i + 1) % 50 == 0 or i == len(traces) - 1:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(traces)}] {dt:.0f}s "
                  f"({(i+1)/dt:.1f}/s) trunc={n_trunc}", flush=True)

    out_dir = C.ACTS / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}{'_rich' if rich else ''}.pt"
    torch.save({"feats": feats, "labels": labels, "uids": uids,
                "n_tokens": n_tokens, "model": model_key, "dataset": dataset,
                "n_layers": n_layers, "hidden": hidden, "pools": list(pools),
                "max_length": max_length}, out_path)
    print(f"[extract] saved {out_path}  ({sum(labels).item()} unsafe / "
          f"{len(labels)} total, {n_trunc} truncated)", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b-instruct")
    ap.add_argument("--dataset", default="r-judge")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--pools", nargs="+", default=["mean", "last"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--render", default="text", choices=["text", "rich"],
                    help="rich: tool schemas with parameter types, defaults and "
                         "descriptions (used for TraceSafe and ATBench in the benchmark "
                         "pipeline)")
    a = ap.parse_args()
    extract(a.model, a.dataset, a.max_length, tuple(a.pools), a.dtype, a.render)


if __name__ == "__main__":
    main()
