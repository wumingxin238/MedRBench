#!/bin/bash
# P100 16GB: vLLM is NOT supported (GPU sm_60). Use Ollama + Qwen2.5-14B.
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_NAME="${ENV_NAME:-medrbench}"
QWEN_OLLAMA_MODEL="${QWEN_OLLAMA_MODEL:-qwen2.5:14b-instruct}"

echo "==> P100 setup: Ollama judge (vLLM requires Volta/A100+, Python>=3.8)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda first."
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Recreate env with Python 3.10 (was likely 3.6): conda remove -n $ENV_NAME --all"
fi
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

echo "==> Python: $(which python) ($(python --version))"
pip install -U pip
pip install openai tqdm requests numpy

cat > "$PROJECT_ROOT/scripts/server/config/eval_config.env" <<EOF
export EVAL_BACKEND=ollama
export EVAL_BASE_URL=http://127.0.0.1:11434/v1
export EVAL_API_KEY=ollama
export EVAL_MODEL=$QWEN_OLLAMA_MODEL
export EVAL_TEMPERATURE=0
export EVAL_DISABLE_WEB_SEARCH=1
EOF

chmod +x "$SCRIPT_DIR/../judge/start_ollama_judge.sh" "$SCRIPT_DIR/../../eval/run_eval_gemini35.sh" 2>/dev/null || true

echo ""
echo "Wrote scripts/server/config/eval_config.env (Ollama)"
echo "Next:"
echo "  1) Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
echo "  2) bash scripts/server/judge/start_ollama_judge.sh"
echo "  3) conda activate $ENV_NAME && source scripts/server/config/eval_config.env && bash scripts/eval/run_eval_gemini35.sh"
