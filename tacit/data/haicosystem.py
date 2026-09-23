"""HAICOSYSTEM loader (all_episodes.jsonl, all_environments.jsonl), used by
scripts/data/build_haico_pairs.py. A trajectory is unsafe if any risk dimension is
negative.

  profile      the turn-0 Environment context addressed to the AI agent
  user         utterances of the human participant
  agent        AI utterances (content) and [action] tool calls (action)
  environment  the Observation blocks relayed by the Environment
"""
from __future__ import annotations
import json
import re

from .schema import Trace, Turn
from .. import config as C

RISK_DIMS = ("targeted_safety_risks", "system_and_operational_risks",
             "content_safety_risks", "societal_risks",
             "legal_and_rights_related_risks")


def _strip_said(c: str) -> str:
    m = re.match(r'\s*said:\s*"(.*)"\s*$', c, re.S)
    return m.group(1) if m else c.strip()


def load() -> list[Trace]:
    envs = {}
    with open(C.HAICOSYSTEM_DIR / "all_environments.jsonl") as f:
        for line in f:
            e = json.loads(line)
            envs[e.get("pk")] = e
    traces = []
    with open(C.HAICOSYSTEM_DIR / "all_episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            try:
                dims = ep["rewards"][1][1]
            except (KeyError, IndexError, TypeError):
                continue
            if not any(k in dims for k in RISK_DIMS):
                continue
            label = int(any(dims.get(k, 0) < 0 for k in RISK_DIMS))

            msgs = ep.get("messages") or []
            if not msgs:
                continue
            # participants: receivers of the turn-0 Environment context; the AI agent is
            # the second (rewards[1] carries its risk dimensions)
            ctx = [(r, c) for s, r, c in msgs[0] if s == "Environment"]
            if len(ctx) < 2:
                continue
            human, ai = ctx[0][0], ctx[1][0]
            profile = ctx[1][1].strip()

            turns: list[Turn] = []
            for turn in msgs[1:]:
                for s, r, c in turn:
                    c = str(c)
                    if s == human and r == "Environment":
                        if c.strip() in ("did nothing", "left the conversation"):
                            continue
                        turns.append(Turn(role="user", content=_strip_said(c)))
                    elif s == ai and r == "Environment":
                        if c.strip() == "did nothing":
                            continue
                        if c.lstrip().startswith("[action]"):
                            act = c.lstrip()[len("[action]"):].strip()
                            tn = None
                            try:
                                tn = json.loads(act).get("tool")
                            except Exception:
                                pass
                            turns.append(Turn(role="agent", action=act,
                                              tool_name=tn))
                        else:
                            turns.append(Turn(role="agent",
                                              content=_strip_said(c)))
                    elif s == "Environment" and r == ai and "\nObservation:" in c:
                        obs = c.split("\nObservation:", 1)[1].strip()
                        if obs:
                            turns.append(Turn(role="environment", content=obs))
            if not turns:
                continue
            env = envs.get(ep.get("environment"), {})
            traces.append(Trace(
                uid=f"haicosystem-{ep['pk']}", source="haicosystem",
                label=label, turns=turns, profile=profile,
                attack_type=str(env.get("domain", "")), tools=[]))
    return traces
