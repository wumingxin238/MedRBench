#!/bin/bash
# Qwen judge on P100 16GB — use 14B only
set -eu

MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"

echo "==> Model: $MODEL (P100 16GB)"
echo "==> API: http://127.0.0.1:$PORT/v1"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2>/dev/null || true

exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size 1 \
  --enforce-eager
