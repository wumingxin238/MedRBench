#!/usr/bin/env bash
# Optional deps for Stage-1 reasoning_eval with --judge server (OpenAI-compatible API).
# --judge gemma-local does NOT require openai after metrics/utils lazy-import fix.
set -euo pipefail

conda activate gemma_scope

pip install --only-binary :all: 'openai>=1.0.0' 2>/dev/null || pip install 'openai>=1.0.0'

echo "OK. Gemma-local judge: no extra deps."
echo "Server judge: set EVAL_BACKEND=vllm EVAL_BASE_URL=http://127.0.0.1:8000/v1 EVAL_MODEL=..."
