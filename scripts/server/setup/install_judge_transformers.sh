#!/bin/bash
# P100: local judge via Transformers (no Ollama). Run inside conda env medrbench.
set -eu

echo "==> Installing judge server deps (Python $(python --version 2>&1))"
pip install -U pip
pip install torch transformers accelerate bitsandbytes fastapi uvicorn

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

cat > "$PROJECT_ROOT/scripts/server/config/eval_config.env" <<EOF
export EVAL_BACKEND=vllm
export EVAL_BASE_URL=http://127.0.0.1:8000/v1
export EVAL_API_KEY=local
export EVAL_MODEL=$MODEL
export EVAL_TEMPERATURE=0
export EVAL_DISABLE_WEB_SEARCH=1
EOF

echo "Wrote scripts/server/config/eval_config.env"
echo ""
echo "Terminal 1 (tmux):"
echo "  conda activate medrbench"
echo "  cd $PROJECT_ROOT"
echo "  python scripts/server/judge/judge_server_transformers.py --model $MODEL --port 8000"
echo ""
echo "Terminal 2:"
echo "  conda activate medrbench"
echo "  source scripts/server/config/eval_config.env"
echo "  bash scripts/eval/run_eval_gemini35.sh"
