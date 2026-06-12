#!/usr/bin/env bash
# Gemma 9B + SAE: score deepseek-r1, o3-mini, qwen3-8b (diagnosis demo 100).
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gemma_scope

unset CUDA_VISIBLE_DEVICES
export GEMMA_9B_MODE=fp16
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparse/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python scripts/stage1/prepare_subject_outputs.py --task diagnosis

for m in deepseek-r1 o3-mini qwen3-8b; do
  echo ""
  echo "=========================================="
  echo "Gemma 9B eval: ${m}"
  echo "=========================================="
  python scripts/stage1/run_gemma_scope_eval.py \
    --task diagnosis \
    --gemma-size 9b \
    --subject-model "${m}" \
    --groups direct sae_augmented
done

echo ""
echo "Done:"
ls -lh data/Stage1/gemma_scope/diagnosis_9b_*.json
