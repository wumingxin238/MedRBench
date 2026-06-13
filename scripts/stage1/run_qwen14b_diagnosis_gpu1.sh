#!/usr/bin/env bash
# Qwen3-14B diagnosis inference while Gemma 9B may occupy GPU0.
#
# Strategy A (GPU1 only, 4bit + CPU offload during load):
#   bash scripts/stage1/run_qwen14b_diagnosis_gpu1.sh
#
# Strategy B (Gemma on GPU0, 14B fp16 mostly on GPU1):
#   bash scripts/stage1/run_qwen14b_diagnosis_gpu1.sh asymmetric
#
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

ENV_NAME="${QWEN_ENV:-qwen3_infer}"
conda activate "$ENV_NAME"

MODE="${1:-gpu1}"

python -c "import numpy; assert int(numpy.__version__.split('.')[0]) < 2" 2>/dev/null \
  || { echo "Fix: pip install 'numpy==1.26.4' --only-binary :all:"; exit 1; }

OUT="data/Stage1/inference/qwen3-14b_diagnosis.json"

if [[ "$MODE" == "asymmetric" ]]; then
  echo "Mode: fp16-asymmetric (GPU0 limited, GPU1 main) — do NOT set CUDA_VISIBLE_DEVICES"
  unset CUDA_VISIBLE_DEVICES
  QUANT="--quant-mode fp16-asymmetric"
else
  echo "Mode: 4bit on GPU1 only (CUDA_VISIBLE_DEVICES=1)"
  export CUDA_VISIBLE_DEVICES=1
  QUANT="--quant-mode auto"
fi

echo "Visible GPUs:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv 2>/dev/null || true

if [[ -f "$OUT" ]]; then
  python -c "
import json
d = json.load(open('$OUT', encoding='utf-8'))
errs = sum(1 for v in d.values() if 'error' in v.get('qwen3-14b', {}))
print(f'Resuming {len(d)}/100 cases (errors={errs})')
"
fi

python scripts/stage1/run_qwen_inference.py \
  --task diagnosis \
  --model qwen3-14b \
  $QUANT

python -c "
import json
d = json.load(open('$OUT', encoding='utf-8'))
errs = sum(1 for v in d.values() if 'error' in v.get('qwen3-14b', {}))
print(f'Done: {len(d)}/100 cases, errors={errs}')
"
