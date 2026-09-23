"""Placebo trajectories (written by scripts/data/build_tracesafe_placebo.py), all with
label 0."""
from __future__ import annotations
import json

from .schema import Trace
from .tracesafe import _turns
from .. import config as C


def load(path: str | None = None) -> list[Trace]:
    fp = path or (C.DERIVED / "tracesafe_placebo.jsonl")
    traces: list[Trace] = []
    for line in open(fp):
        r = json.loads(line)
        traces.append(Trace(
            uid=r["uid"], source="tracesafe-placebo", label=0,
            turns=_turns(r.get("trace") or []),
            attack_type=f"placebo_{r['category']}",
            tools=r.get("tool_lists") or [],
            meta={"env": r.get("environment", ""), "target_tool": r.get("target_tool"),
                  "placebo_tool": r.get("placebo_tool")},
        ))
    return traces
