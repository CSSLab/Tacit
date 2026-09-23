"""Trajectories of the HAICOSYSTEM pairs (written by scripts/data/build_haico_pairs.py);
per-dimension risk scores are kept in ``meta["dims"]``."""
from __future__ import annotations
import json

from .schema import Trace, Turn
from .. import config as C


def load(path: str | None = None) -> list[Trace]:
    fp = path or (C.DERIVED / "haico_pairs.jsonl")
    out: list[Trace] = []
    for line in open(fp):
        r = json.loads(line)
        turns = [Turn(role=t["role"], content=t.get("content"), action=t.get("action"),
                      thought=t.get("thought"), tool_name=t.get("tool_name"))
                 for t in r["turns"]]
        out.append(Trace(uid=r["uid"], source="haicosystem", label=int(r["label"]),
                         turns=turns, profile=r.get("profile"),
                         attack_type=str(r.get("domain", "")), tools=[],
                         meta={"dims": r.get("dims", {})}))
    return out
