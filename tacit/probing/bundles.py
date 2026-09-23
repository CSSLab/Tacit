"""Loading of cached activation bundles (written by tacit.activations.extract)."""
from __future__ import annotations

import torch

from .. import config as C

_NOTED = set()


def load_bundle(model: str, dataset: str, render: str | None = None) -> dict:
    """Load ACTS/<model>/<dataset>.pt.

    render="rich" (default: config.RENDER) reads ``<dataset>_rich.pt`` when it exists,
    else the plain bundle; render="text" always reads the plain bundle.
    """
    render = C.RENDER if render is None else render
    if render not in ("text", "rich"):
        raise ValueError(f"render={render!r}; expected 'text' or 'rich'")
    if render == "rich" and not dataset.endswith("_rich"):
        twin = C.ACTS / model / f"{dataset}_rich.pt"
        if twin.exists():
            if (model, dataset) not in _NOTED:
                print(f"[rich] {model}/{dataset} -> {twin.name}", flush=True)
                _NOTED.add((model, dataset))
            dataset = twin.stem
    b = torch.load(C.ACTS / model / f"{dataset}.pt", weights_only=False)
    # bundles are stored in fp16; clamp the rare overflowing entries to its range
    for p, t in b["feats"].items():
        if torch.isinf(t).any():
            b["feats"][p] = torch.nan_to_num(t, posinf=65504.0, neginf=-65504.0)
    return b
