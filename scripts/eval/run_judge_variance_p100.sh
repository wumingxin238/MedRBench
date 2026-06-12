#!/bin/bash
# P100: Judge variance probe (hit-check stability test)
# Usage:
#   Terminal 1: bash scripts/eval/run_judge_variance_p100.sh judge
#   Terminal 2: bash scripts/eval/run_judge_variance_p100.sh probe
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# --- CUDA for bitsandbytes / PyTorch on P100 ---
if command -v module >/dev/null 2>&1; then
  module unload cuda/10.1 2>/dev/null || true
  module load cuda/11.7 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda/12.0 2>/dev/null || true
fi
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

MODE="${1:-probe}"

if [[ "$MODE" == "judge" ]]; then
  echo "Starting Qwen2.5-7B judge on :8000 (4-bit; use --no-4bit if bitsandbytes fails)"
  exec python scripts/server/judge/judge_server_transformers.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --seed 42
fi

if [[ "$MODE" == "judge-fp16" ]]; then
  echo "Starting judge fp16 (no bitsandbytes) — needs ~16GB VRAM"
  exec python scripts/server/judge/judge_server_transformers.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --no-4bit \
    --seed 42
fi

if [[ "$MODE" == "probe" ]]; then
  if [[ -f scripts/server/config/eval_config.env ]]; then
    # shellcheck disable=SC1091
    source scripts/server/config/eval_config.env
  else
    export EVAL_BACKEND=vllm
    export EVAL_BASE_URL=http://127.0.0.1:8000/v1
    export EVAL_API_KEY=local
    export EVAL_MODEL=Qwen/Qwen2.5-7B-Instruct
    export EVAL_TEMPERATURE=0
    export EVAL_DISABLE_WEB_SEARCH=1
  fi

  # Wait for judge if not up
  for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      break
    fi
    echo "Waiting for judge on :8000 ($i/30)..."
    sleep 2
  done

  exec python scripts/eval/judge_completeness_variance.py \
    --cases PMC11321471 PMC11625232 PMC11375620 PMC11417919 PMC11609106 \
    --models gemini2-ft qwq deepseek-r1 \
    --repeats 10 \
    --mode hit_only \
    --temperature 0 \
    --seed 42
fi

echo "Usage: $0 {judge|judge-fp16|probe}"
exit 1
