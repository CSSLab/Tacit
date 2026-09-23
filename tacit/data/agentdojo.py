"""AgentDojo loader for data/agentdojo/<model>/<pipeline>/<suite>/... (written by
scripts/data/collect_agentdojo.sh). A trajectory is unsafe when ``security is False``."""
from __future__ import annotations
import ast
import glob
import json

from .schema import Trace, Turn
from .. import config as C

ROOT = C.AGENTDOJO_DIR
DEFAULT_MODELS = ("qwen3-4b-instruct", "llama3.1-8b-instruct")


def _text(blob) -> str:
    """Message content is stored as a stringified list of typed blocks."""
    if blob is None or blob == "None":
        return ""
    if isinstance(blob, str):
        try:
            blob = ast.literal_eval(blob)
        except (ValueError, SyntaxError):
            return blob
    if isinstance(blob, list):
        parts = []
        for b in blob:
            if isinstance(b, dict):
                parts.append(str(b.get("content", "")))
            else:
                parts.append(str(b))
        return "\n".join(p for p in parts if p and p != "None")
    return str(blob)


def _calls(blob) -> list[dict]:
    if not blob or blob == "None":
        return []
    if isinstance(blob, str):
        try:
            blob = ast.literal_eval(blob)
        except (ValueError, SyntaxError):
            return []
    out = []
    for c in blob if isinstance(blob, list) else []:
        if isinstance(c, dict):
            out.append({"name": c.get("function"), "arguments": c.get("args")})
    return out


def _to_trace(rec: dict, uid: str, label: int) -> Trace:
    turns, profile = [], ""
    for m in rec.get("messages", []):
        role, content = m.get("role"), _text(m.get("content"))
        if role == "system":
            profile = content
        elif role == "user":
            turns.append(Turn("user", content=content))
        elif role == "assistant":
            calls = _calls(m.get("tool_calls"))
            if calls:
                turns.append(Turn("agent", thought=content or None,
                                  action=json.dumps(calls, ensure_ascii=False,
                                                    default=str),
                                  tool_name=calls[0]["name"]))
            else:
                turns.append(Turn("agent", action=content))
        elif role == "tool":
            turns.append(Turn("environment", content=content))
    return Trace(uid=uid, source="agentdojo", label=label, turns=turns, profile=profile,
                 attack_type=rec.get("attack_type") or "none",
                 meta={"suite": rec.get("suite_name"),
                       "user_task": rec.get("user_task_id"),
                       "injection_task": rec.get("injection_task_id")})


def load(models: tuple[str, ...] = DEFAULT_MODELS) -> list[Trace]:
    out = []
    for model in models:
        base = ROOT / model
        if not base.is_dir():
            continue
        # <pipeline>/<suite>/user_task_*/<attack>/<injection task>.json
        for path in sorted(glob.glob(str(base / "*" / "*" / "user_task_*" / "*" / "*.json"))):
            rec = json.load(open(path))
            if rec.get("security") is None or rec.get("error"):
                continue
            suite = rec.get("suite_name")
            ut, it = rec.get("user_task_id"), rec.get("injection_task_id") or "none"
            uid = f"agentdojo-{model.split('-')[0]}-{suite}-{ut}-{it}"
            out.append(_to_trace(rec, uid, int(rec["security"] is False)))
    if not out:
        raise FileNotFoundError(f"no AgentDojo rollouts under {ROOT}")
    return out
