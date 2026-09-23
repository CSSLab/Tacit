"""Salient-dimension selection of the Multi-layer readout."""
from __future__ import annotations

import numpy as np


def select_salient(coef: np.ndarray, eta: float) -> np.ndarray:
    """Indices of the smallest set of dimensions whose absolute coefficients carry a
    fraction ``eta`` of the probe's total coefficient mass, largest first."""
    w = np.abs(coef)
    tot = w.sum()
    if tot == 0:
        return np.arange(len(w))
    wn = w / tot
    order = np.argsort(wn)[::-1]
    cum = np.cumsum(wn[order])
    k = int(np.searchsorted(cum, eta)) + 1
    return order[:k]
