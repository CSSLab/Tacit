"""R-Judge loader.

Raw layout: R-Judge/data/<Category>/<scenario>.json, each a list of records
  {id, scenario, profile, goal, contents, label, risk_description, attack_type}
where ``contents`` is a list of sessions, each a list of turns
  {role: user|agent|environment, content?, thought?, action?}.
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

from .schema import Trace, Turn
from .. import config as C

# "ToolName: {json args}"
_ACTION_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*:\s*(\{.*\})\s*$", re.DOTALL)


def _parse_action(action: str | None) -> tuple[str | None, dict | None]:
    """Parse an R-Judge action string into (tool_name, args) where possible."""
    if not action or not isinstance(action, str):
        return None, None
    m = _ACTION_RE.match(action.strip())
    if not m:
        # tool name without JSON arguments, e.g. "FinishAction: ..."
        head = action.strip().split(":", 1)[0].strip()
        return (head if re.fullmatch(r"[A-Za-z_][\w.]*", head) else None), None
    name, raw = m.group(1), m.group(2)
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            args = {"_": args}
    except Exception:
        args = None
    return name, args


def _normalise_turn(raw: dict) -> Turn:
    role = raw.get("role", "")
    if role == "agent":
        name, args = _parse_action(raw.get("action"))
        return Turn(role="agent", thought=raw.get("thought"),
                    action=raw.get("action"), tool_name=name, tool_args=args)
    content = raw.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return Turn(role=role if role in ("user", "environment") else "user", content=content)


def load(rjudge_dir: str | Path | None = None) -> list[Trace]:
    """All R-Judge records as a flat list of ``Trace``."""
    root = Path(rjudge_dir or C.RJUDGE_DIR)
    files = sorted(glob.glob(str(root / "data" / "*" / "*.json")))
    if not files:
        raise FileNotFoundError(f"No R-Judge json under {root/'data'}")
    traces: list[Trace] = []
    for fp in files:
        category = Path(fp).parent.name
        for rec in json.load(open(fp)):
            turns: list[Turn] = []
            for session in rec.get("contents", []):
                for raw in session:
                    turns.append(_normalise_turn(raw))
            traces.append(Trace(
                uid=f"rjudge-{rec['id']}",
                source="r-judge",
                label=int(rec["label"]),
                turns=turns,
                profile=rec.get("profile", "") or "",
                goal=rec.get("goal", "") or "",
                risk_description=rec.get("risk_description", "") or "",
                attack_type=str(rec.get("attack_type", "")),
                meta={"category": category, "scenario": rec.get("scenario", "")},
            ))
    return traces
