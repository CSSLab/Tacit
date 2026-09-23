"""Benchmarks, fold seeds and split files.

Seven sources form six reported columns (the two ASSEBench files share one).
data/splits/split_seed<s>.json (s = 42..45) marks every row as train, val or test; the
test rows are the same in every file.
"""
from __future__ import annotations
import json

from . import config as C

BENCHES = ["r-judge", "tracesafe", "atbench", "assebench-safety",
           "assebench-security", "oas", "agentdojo"]
FOLD_SEEDS = [42, 43, 44, 45]
# reported column of each source
COLUMN_OF = {b: b for b in BENCHES}
COLUMN_OF["assebench-safety"] = COLUMN_OF["assebench-security"] = "assebench"
COLS = list(dict.fromkeys(COLUMN_OF[b] for b in BENCHES))
SPLITS = ("train", "val", "test")


def split_path(seed: int):
    return C.SPLITS / f"split_seed{seed}.json"


def load_split(seed: int) -> dict:
    return json.loads(split_path(seed).read_text())
