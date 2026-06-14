#!/usr/bin/env bash
# Qwen3-14B on GPU1 while Gemma 9B may use GPU0.
set -euo pipefail
cd "$(dirname "$0")/../.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV:-qwen3_infer}"

python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null \
  || { echo "Fix: pip install 'numpy==1.26.4'"; exit 1; }

echo "=== GPU status (GPU1 must be ~0 MiB used) ==="
nvidia-smi

# Only physical GPU1; never touch GPU0 while Gemma runs.
export CUDA_VISIBLE_DEVICES=1

echo "=== Probing 14B load (each attempt in a fresh subprocess) ==="
export CUDA_VISIBLE_DEVICES=1

python scripts/stage1/run_qwen_inference.py \
  --task diagnosis \
  --model qwen3-14b \
  --quant-mode awq \
  --gpu-id 1
