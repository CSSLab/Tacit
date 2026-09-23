"""OpenAgentSafety loader (evaluation/<agent>/{traj,eval}_<task>.json).

A trajectory is kept when its event log is non-empty and its eval file defines at least
one checkpoint; it is unsafe when final_score.result > 0. Observations are cut head+tail
at OBS_CAP characters.
"""
from __future__ import annotations
import json
import os

from .schema import Trace, Turn
from .. import config as C

OBS_CAP = 2000
SKIP_ACTIONS = {"recall", "system", "change_agent_state"}
NOISE_ARGS = {"thought", "impl_source", "image_urls", "wait_for_response",
              "view_range", "start", "end", "old_str", "new_str",
              "insert_line", "outputs", "task_completed", "translated_ipython_code"}


def _cap(s: str, n: int = OBS_CAP) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n // 2] + "\n...[truncated]...\n" + s[-n // 2:]


def _task_text(task: str) -> str:
    for p in (C.OAS_DIR / "workspaces" / "tasks" / task / "task.md",
              C.OAS_DIR / "tasks" / task / "task.md"):
        if p.exists():
            return p.read_text(errors="ignore").strip()
    return ""


def _events_to_turns(events: list[dict]) -> list[Turn]:
    turns: list[Turn] = []
    for e in events:
        src = e.get("source")
        act = e.get("action")
        obs = e.get("observation")
        if act in SKIP_ACTIONS or obs in SKIP_ACTIONS:
            continue
        if obs:
            content = _cap(str(e.get("content") or e.get("message") or ""))
            if content.strip():
                turns.append(Turn(role="environment", content=content))
            continue
        if src == "user" and act == "message":
            turns.append(Turn(role="user",
                              content=str(e.get("message") or "")))
            continue
        if src == "agent" and act:
            args = e.get("args") or {}
            thought = (args.get("thought") or args.get("final_thought")
                       or e.get("message") or "")
            payload = {k: v for k, v in args.items()
                       if k not in NOISE_ARGS and v not in (None, "", "None")}
            fn = ((e.get("tool_call_metadata") or {}).get("function_name")
                  or act)
            action = json.dumps(payload, ensure_ascii=False, default=str)
            turns.append(Turn(role="agent",
                              thought=_cap(str(thought), 800) or None,
                              action=_cap(action, 4000), tool_name=str(fn)))
    return turns


def load(models: list[str] | None = None) -> list[Trace]:
    ev_root = C.OAS_DIR / "evaluation"
    out: list[Trace] = []
    for model in sorted(os.listdir(ev_root)):
        mdir = ev_root / model
        if not mdir.is_dir() or (models and model not in models):
            continue
        for tp in sorted(mdir.glob("traj_*.json")):
            task = tp.name[len("traj_"):-len(".json")]
            ep = mdir / f"eval_{task}.json"
            if not ep.exists():
                continue
            try:
                ev = json.load(open(ep))
                fs = ev["final_score"]
                if int(fs.get("total", 0)) <= 0:
                    continue
                label = int(int(fs["result"]) > 0)
                events = json.load(open(tp))
            except Exception:
                continue
            if not isinstance(events, list) or not events:
                continue
            turns = _events_to_turns(events)
            if not turns:
                continue
            goal = _task_text(task)
            out.append(Trace(
                uid=f"oas-{model}-{task}", source="openagentsafety",
                label=label, turns=turns,
                profile=("You are an AI agent working in a software company's "
                         "environment (terminal, file system, GitLab, chat). "
                         "Task:\n" + goal) if goal else "",
                attack_type=model, tools=[]))
    return out
