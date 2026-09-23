#!/usr/bin/env bash
# Collect AgentDojo trajectories with a local agent served by vLLM, without an attack and
# under the important_instructions attack. Rollouts are written under
# data/agentdojo/<model_key>/.
#
# Usage: bash scripts/data/collect_agentdojo.sh <model_key> [port] [tool_parser] [provider]
#   Qwen3-4B:     bash scripts/data/collect_agentdojo.sh qwen3-4b-instruct 8123 hermes VLLM_PARSED
#   Llama-3.1-8B: bash scripts/data/collect_agentdojo.sh llama3.1-8b-instruct 8123 llama3_json LOCAL
# provider VLLM_PARSED uses vLLM's tool-call parser; LOCAL uses AgentDojo's own prompting
# and parsing (for tool parsers that emit a single call per turn).
# Requires agentdojo==0.1.35 (on PATH as `python`) and vllm (on PATH as `vllm`).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_KEY=${1:-qwen3-4b-instruct}
PORT=${2:-8123}
PARSER=${3:-hermes}
PROVIDER=${4:-VLLM_PARSED}
HF_ID=$(cd "$REPO" && python -c "from tacit import config as C; print(C.MODELS['$MODEL_KEY'])")
OUT=$REPO/data/agentdojo/$MODEL_KEY
mkdir -p "$OUT"

vllm serve "$HF_ID" --port "$PORT" --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser "$PARSER" \
  --gpu-memory-utilization 0.90 > "$OUT/vllm_server.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1 && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited"; tail -30 "$OUT/vllm_server.log"; exit 1; }
  sleep 5
done

export LOCAL_LLM_PORT=$PORT
export PYTHONPATH=$REPO/scripts/data${PYTHONPATH:+:$PYTHONPATH}
# one benchmark process per suite, all sharing the vLLM server
for ATTACK in none important_instructions; do
  PIDS=()
  for S in workspace travel banking slack; do
    ARGS=(--model "$PROVIDER" -ml agentdojo_compat --benchmark-version v1.2.2
          --logdir "$OUT" -s "$S")
    [ "$ATTACK" != none ] && ARGS+=(--attack "$ATTACK")
    python -m agentdojo.scripts.benchmark "${ARGS[@]}" > "$OUT/${ATTACK}_$S.log" 2>&1 &
    PIDS+=($!)
  done
  for P in "${PIDS[@]}"; do wait "$P"; done
done
