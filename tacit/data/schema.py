"""Trajectory representation shared by every dataset loader.

Each loader turns its raw release into ``list[Trace]``; rendering, activation
extraction and the guard baselines only ever see ``Trace``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Turn:
    """One step of a trajectory.

    role: 'user' | 'agent' | 'environment'
      user         content is the human message
      agent        thought (reasoning) and action (the tool call as a string);
                   tool_name / tool_args hold the parsed call when available
      environment  content is the tool output or observation returned to the agent
    """
    role: str
    content: Optional[str] = None
    thought: Optional[str] = None
    action: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None


@dataclass
class Trace:
    uid: str
    source: str                       # dataset name, e.g. 'r-judge'
    label: int                        # 1 = unsafe, 0 = safe
    turns: list[Turn]
    profile: str = ""                 # agent role / system prompt
    goal: str = ""                    # task description
    risk_description: str = ""        # gold rationale, if any
    attack_type: str = ""             # dataset-specific risk category
    tools: list[dict[str, Any]] = field(default_factory=list)  # tool schemas in scope
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def tool_names(self) -> list[str]:
        seen: list[str] = []
        for t in self.turns:
            if t.tool_name and t.tool_name not in seen:
                seen.append(t.tool_name)
        return seen
