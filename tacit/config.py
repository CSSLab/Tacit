"""Paths and model registry. Each path can be overridden by an environment variable:

  TACIT_DATA   raw benchmark corpora (see data/README.md)   default <repo>/data/raw
  TACIT_RUNS   experiment outputs                            default <repo>/runs
  TACIT_ACTS   cached activations                            default <repo>/runs/acts

Model weights resolve through the Hugging Face cache (HF_HOME).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dir(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


DATA_RAW = _dir("TACIT_DATA", PROJECT_ROOT / "data" / "raw")
RUNS = _dir("TACIT_RUNS", PROJECT_ROOT / "runs")
ACTS = _dir("TACIT_ACTS", RUNS / "acts")
#: frozen train/validation/test assignment of every benchmark row
SPLITS = PROJECT_ROOT / "data" / "splits"
#: our AgentDojo rollouts (scripts/data/collect_agentdojo.sh)
AGENTDOJO_DIR = PROJECT_ROOT / "data" / "agentdojo"
#: pair sets built from the raw corpora by scripts/data/
DERIVED = RUNS / "derived"

#: tool-schema rendering read by the benchmark pipeline: "rich" writes parameter types,
#: defaults and descriptions; "text" writes parameter names only
RENDER = "rich"

MODELS = {
    # backbones
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    # released guard checkpoints
    "qwen3guard-4b": "Qwen/Qwen3Guard-Gen-4B",
    "llamaguard3-8b": "meta-llama/Llama-Guard-3-8B",
    "agentdog": "AI45Research/AgentDoG-Qwen3-4B",
}

RJUDGE_DIR = DATA_RAW / "R-Judge"
TRACESAFE_DIR = DATA_RAW / "TraceSafe" / "data"
ATBENCH_DIR = DATA_RAW / "ATBench"
ASSEBENCH_DIR = DATA_RAW / "AgentAuditor-ASSEBench"
OAS_DIR = DATA_RAW / "OpenAgentSafety"
HAICOSYSTEM_DIR = DATA_RAW / "HAICOSYSTEM"


def run_dir(*parts: str) -> Path:
    """RUNS/<parts...>, created on first use."""
    p = RUNS.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
