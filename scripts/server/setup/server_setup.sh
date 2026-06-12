#!/bin/bash
# MedRBench server bootstrap: local Qwen evaluator + evaluation deps
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ENV_NAME="${ENV_NAME:-medrbench}"
QWEN_OLLAMA_MODEL="${QWEN_OLLAMA_MODEL:-qwen2.5:14b-instruct}"
QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
USE_OLLAMA="${USE_OLLAMA:-1}"

echo "==> Project root: $PROJECT_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> GPU:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo "==> No nvidia-smi (CPU-only Ollama is OK for smaller Qwen models)."
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.11 -y
  fi
  conda activate "$ENV_NAME"
else
  echo "Conda not found; using current Python: $(python --version)"
fi

pip install -U pip
pip install openai tqdm requests numpy
pip install selenium beautifulsoup4 fake-useragent duckduckgo-search

if [ "$USE_OLLAMA" = "1" ]; then
  echo ""
  echo "==> Ollama"
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Install: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
  fi
  ollama pull "$QWEN_OLLAMA_MODEL" || true
  cat > "$PROJECT_ROOT/scripts/server/config/eval_config.env" <<EOF
export EVAL_BACKEND=ollama
export EVAL_BASE_URL=http://127.0.0.1:11434/v1
export EVAL_API_KEY=ollama
export EVAL_MODEL=$QWEN_OLLAMA_MODEL
export EVAL_TEMPERATURE=0
export EVAL_DISABLE_WEB_SEARCH=1
EOF
else
  echo ""
  echo "==> vLLM"
  pip install torch transformers accelerate || true
  pip install vllm || echo "vLLM install failed — install manually or use USE_OLLAMA=1"
  cat > "$PROJECT_ROOT/scripts/server/config/eval_config.env" <<EOF
export EVAL_BACKEND=vllm
export EVAL_BASE_URL=http://127.0.0.1:$VLLM_PORT/v1
export EVAL_API_KEY=local
export EVAL_MODEL=$QWEN_VLLM_MODEL
export EVAL_TEMPERATURE=0
export EVAL_DISABLE_WEB_SEARCH=1
EOF
fi

echo "Wrote $PROJECT_ROOT/scripts/server/config/eval_config.env"
echo "==> Setup complete."
