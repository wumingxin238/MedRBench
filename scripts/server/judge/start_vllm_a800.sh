#!/bin/bash
# Qwen judge on A800 80GB
set -eu

MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.92}"

echo "==> Model: $MODEL"
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
