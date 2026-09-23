# Data

## Shipped with the code

- `splits/split_seed{42,43,44,45}.json`: the frozen assignment of every row of the seven
  sources. The test side (20%) is identical in all four files; the remaining 80% forms
  four validation folds, and each file marks one of them as `val` and the other three as
  `train`. Rows are listed in loader order, with their labels.
- `agentdojo/`: our AgentDojo trajectories (Qwen3-4B and Llama-3.1-8B as agents), as
  written by `scripts/data/collect_agentdojo.sh`.

## Raw benchmarks

Fetch each benchmark from its own release and place it under `$TACIT_DATA`
(default `data/raw/`):

| Directory | Dataset key | Expected contents |
|---|---|---|
| `R-Judge/` | `r-judge` | `data/<Category>/*.json` (R-Judge release) |
| `TraceSafe/` | `tracesafe` | `data/golden_*.jsonl` (github.com/cycraft-corp/TraceSafe) |
| `ATBench/` | `atbench` | `ATBench/test.json`, the 1,000-row release (huggingface.co/collections/AI45Research/agentdog) |
| `AgentAuditor-ASSEBench/` | `assebench-safety`, `assebench-security` | `ASSEBench/dataset/AgentJudge-{safety,security}.json` (github.com/Astarojth/AgentAuditor-ASSEBench) |
| `OpenAgentSafety/` | `oas` | `evaluation/<agent>/{traj,eval}_<task>.json`, `workspaces/tasks/<task>/task.md` (github.com/Open-Agent-Safety/OpenAgentSafety) |
| `HAICOSYSTEM/` | pair sets only | `all_episodes.jsonl`, `all_environments.jsonl` (HAICOSYSTEM dataset release) |

Check the placement (run from the repository root with `PYTHONPATH=$PWD`):

```bash
python - <<'PY'
from tacit.activations.extract import _iter_dataset
for d in ["r-judge", "tracesafe", "atbench", "assebench-safety",
          "assebench-security", "oas", "agentdojo"]:
    t = _iter_dataset(d)
    print(f"{d:20s} n={len(t):5d} unsafe={sum(x.label for x in t):5d}")
PY
```

Expected:

```
r-judge              n=  571 unsafe=  301
tracesafe            n= 2250 unsafe= 1080
atbench              n= 1000 unsafe=  497
assebench-safety     n= 1476 unsafe=  806
assebench-security   n=  817 unsafe=  409
oas                  n= 1298 unsafe=  534
agentdojo            n= 2075 unsafe= 1706
```

Every loader lists its rows in the order of the split files; `tacit.features.load_feats`
checks this row by row.
