"""Cached activations aligned row by row with the frozen splits."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from .probing.bundles import load_bundle
from .suite import BENCHES, COLS, COLUMN_OF


def load_feats(model: str, split: dict | None = None, benches=BENCHES,
               pools=("last", "mean")) -> dict:
    """Per-source float16 (N, L, H) stacks plus labels; the row order must match the
    split file."""
    out = {}
    for b in benches:
        bd = load_bundle(model, b)
        uids = [str(u) for u in bd["uids"]]
        y = bd["labels"].numpy().astype(int)
        if split is not None:
            sb = split["benches"][b]
            if uids != [str(u) for u in sb["uids"]]:
                bad = next(i for i, (p, q) in enumerate(zip(uids, sb["uids"])) if p != str(q))
                raise AssertionError(f"{b}: bundle row order differs from split at {bad}")
            if y.tolist() != list(sb["labels"]):
                raise AssertionError(f"{b}: bundle labels differ from split labels")
        out[b] = {"X": {p: bd["feats"][p].numpy() for p in pools}, "y": y, "uids": uids}
        print(f"[feat] {b:20s} {out[b]['X'][pools[0]].shape} unsafe={int(y.sum())}",
              flush=True)
    return out


def keyed(uids) -> list[str]:
    """Row keys that stay unique if a uid repeats within a source: the k-th occurrence
    of a uid becomes "uid#k". Every artifact lists rows in loader order, so the keys
    agree across artifacts."""
    seen: dict = {}
    out = []
    for u in map(str, uids):
        k = seen.get(u, 0)
        seen[u] = k + 1
        out.append(f"{u}#{k}")
    return out


def pool_feat(f: dict, pool: str, layer: int) -> np.ndarray:
    if pool == "both":
        return np.concatenate([f["X"]["mean"][:, layer, :], f["X"]["last"][:, layer, :]],
                              axis=1).astype(np.float32)
    return f["X"][pool][:, layer, :].astype(np.float32)


def split_mask(split: dict, b: str, part: str) -> np.ndarray:
    return np.array(split["benches"][b]["split"]) == part


def stack(feats: dict, split: dict, pool: str, layer: int, part: str,
          benches=BENCHES):
    """(X, y, column) for one split part over all sources, at one layer."""
    Xs, ys, src = [], [], []
    for b in benches:
        m = split_mask(split, b, part)
        Xs.append(pool_feat(feats[b], pool, layer)[m])
        ys.append(feats[b]["y"][m])
        src += [COLUMN_OF[b]] * int(m.sum())
    return np.vstack(Xs), np.concatenate(ys), np.array(src)


def labels_and_sources(feats: dict, split: dict, part: str, benches=BENCHES):
    ys, src, uids = [], [], []
    for b in benches:
        m = split_mask(split, b, part)
        ys.append(feats[b]["y"][m])
        src += [COLUMN_OF[b]] * int(m.sum())
        uids += [u for u, k in zip(feats[b]["uids"], m) if k]
    return np.concatenate(ys), np.array(src), uids


def per_source_f1(y, pred, src, benches=COLS) -> dict:
    return {b: float(f1_score(y[src == b], pred[src == b], average="macro"))
            for b in benches}


def mean_f1(per: dict, benches=COLS) -> float:
    return float(np.mean([per[b] for b in benches]))
