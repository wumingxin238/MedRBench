#!/bin/bash
# A800 80GB: vLLM + Qwen2.5-32B judge
set -eu

export USE_OLLAMA=0
export QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/server_setup.sh"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

pip install torch 2>/dev/null || true
pip install vllm || { echo "vLLM install failed"; exit 1; }

chmod +x "$SCRIPT_DIR/../judge/start_vllm_a800.sh" "$SCRIPT_DIR/../../eval/run_eval_gemini35.sh" 2>/dev/null || true

cat > "$PROJECT_ROOT/scripts/server/config/eval_config.env" <<EOF
export EVAL_BACKEND=vllm
export EVAL_BASE_URL=http://127.0.0.1:8000/v1
export EVAL_API_KEY=local
export EVAL_MODEL=$QWEN_VLLM_MODEL
export EVAL_TEMPERATURE=0
export EVAL_DISABLE_WEB_SEARCH=1
EOF

echo ""
echo "==> A800 profile ready (32B). Start: bash scripts/server/judge/start_vllm_a800.sh"
