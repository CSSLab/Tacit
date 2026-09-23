"""ASSEBench loader (AgentJudge-{safety,security}.json)."""
from __future__ import annotations
import json
import re
from pathlib import Path

from .schema import Trace, Turn
from .. import config as C


def _turns(contents) -> list[Turn]:
    out: list[Turn] = []
    for traj in (contents or []):
        for t in traj:
            role = t.get("role")
            if role == "user":
                out.append(Turn(role="user", content=t.get("content", "") or ""))
            elif role == "agent":
                act = (t.get("action", "") or "").strip()
                tn = re.split(r"[{\n(]", act, 1)[0].strip() if act else None
                out.append(Turn(role="agent", thought=t.get("thought"),
                                action=act or None, tool_name=(tn or None)))
            elif role in ("environment", "tool"):
                out.append(Turn(role="environment", content=t.get("content", "") or ""))
    return out


def _load_file(path: Path, source: str, split: str) -> list[Trace]:
    raw = json.load(open(path))
    out: list[Trace] = []
    for e in raw:
        turns = _turns(e.get("contents", []))
        if not turns:
            continue
        out.append(Trace(
            uid=f"{source}-{split}-{e['id']}", source=source, label=int(e["label"]),
            turns=turns, profile=(e.get("profile") or ""), goal=(e.get("goal") or ""),
            risk_description=(e.get("risk_description") or ""),
            attack_type=str(e.get("risk_type", "")), tools=[]))
    return out


def load(split: str = "safety") -> list[Trace]:
    """split: safety | security."""
    return _load_file(C.ASSEBENCH_DIR / "ASSEBench" / "dataset" / f"AgentJudge-{split}.json",
                      "assebench", split)
