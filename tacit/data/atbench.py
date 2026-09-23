"""ATBench loader (ATBench/test.json).

Record fields: tool_used (schemas), contents (list of sessions of {role, content}),
label (0/1), risk_source, failure_mode, real_world_harm, reason.
"""
from __future__ import annotations
import json
from pathlib import Path

from .schema import Trace, Turn
from .. import config as C


def _s(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _turns(sessions: list) -> list[Turn]:
    out: list[Turn] = []
    for sess in sessions:
        seq = sess if isinstance(sess, list) else [sess]
        for m in seq:
            if not isinstance(m, dict):
                continue
            role, content = m.get("role"), m.get("content")
            if role == "user":
                out.append(Turn("user", content=_s(content)))
            elif role in ("agent", "assistant"):
                # agent messages carry the call in `action` (with `thought` alongside)
                act = m.get("action")
                if act in (None, "") and isinstance(content, dict) and "name" in content:
                    act = content
                if act in (None, ""):
                    act = content
                tool_name = tool_args = None
                if isinstance(act, dict):
                    tool_name = act.get("name")
                    tool_args = act.get("arguments") if isinstance(act.get("arguments"),
                                                                   dict) else None
                elif isinstance(act, str):
                    # 'Complete{...}' or a bare '{"name": ..., "arguments": ...}'
                    brace = act.find("{")
                    if brace >= 0:
                        try:
                            p = json.loads(act[brace:])
                            if isinstance(p, dict) and "name" in p:
                                tool_name = p.get("name")
                                tool_args = p.get("arguments") if isinstance(
                                    p.get("arguments"), dict) else None
                        except Exception:
                            pass
                out.append(Turn("agent", action=_s(act), thought=(m.get("thought") or None),
                                tool_name=tool_name, tool_args=tool_args))
            elif role in ("tool", "environment"):
                out.append(Turn("environment", content=_s(content)))
            else:
                out.append(Turn("user", content=_s(content)))
    return out


def load(path: str | None = None) -> list[Trace]:
    path = Path(path) if path else C.ATBENCH_DIR / "ATBench" / "test.json"
    d = json.load(open(path))
    traces: list[Trace] = []
    for r in d:
        contents = r.get("contents") or r.get("content") or []
        traces.append(Trace(
            uid=f"atbench-{r['id']}",
            source="atbench", label=int(r["label"]),
            turns=_turns(contents),
            attack_type=str(r.get("risk_source", "")),
            risk_description=str(r.get("reason", "")),
            tools=r.get("tool_used", []) or [],
            meta={"failure_mode": r.get("failure_mode"),
                  "harm": r.get("real_world_harm")},
        ))
    return traces
