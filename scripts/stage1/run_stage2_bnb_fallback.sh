#!/usr/bin/env bash
# Quick path: skip AWQ/autoawq — use bitsandbytes 4bit on base Qwen3-14B (GPU1).
# Run on server when conda/pip AWQ deps are too painful.
#
#   bash scripts/stage1/run_stage2_bnb_fallback.sh
#
set -eu
cd "$(dirname "$0")/../.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3_infer

if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
fi

echo "==> Testing bitsandbytes..."
if ! python -c "import bitsandbytes; print('bitsandbytes OK')" 2>&1; then
  echo "bitsandbytes still broken. Need LD_LIBRARY_PATH + nvidia-cusparse pip packages."
  echo "  pip install nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cusparse-cu11"
  exit 1
fi

export QWEN_QUANT_MODE=4bit
echo "Using QWEN_QUANT_MODE=4bit (base Qwen3-14B, not AWQ checkpoint)"

tmux kill-session -t stage2 2>/dev/null || true
STAGE2_FORCE=1 bash scripts/stage1/run_stage2_tmux.sh
