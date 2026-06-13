#!/usr/bin/env bash
# Dedicated env for Qwen3 inference (transformers>=4.51).
# Keep gemma_scope at transformers 4.44.x for Gemma Scope + SAE.
#
# P100 / CentOS 7: never compile sentencepiece — use conda + binary wheels only.
set -eu

ENV_NAME="${ENV_NAME:-qwen3_infer}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Creating $ENV_NAME (python 3.10)"
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
pip install -U pip

echo "==> PyTorch (cu118 binary wheels)"
pip install torch==2.1.2 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

echo "==> sentencepiece via conda (avoid pip compile on old GCC)"
conda install -c conda-forge sentencepiece -y

echo "==> transformers + deps (4.51.x — Qwen3 needs >=4.51; avoid 5.x on torch 2.1)"
pip install \
  "numpy==1.26.4" \
  "transformers>=4.51.0,<5.0" \
  "tokenizers>=0.21,<0.22" \
  accelerate \
  "bitsandbytes==0.42.0" \
  tqdm \
  "huggingface_hub>=0.26,<1.0" \
  --only-binary :all:

# Persist bitsandbytes CUDA libs (same as gemma_scope)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<'EOF'
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparse/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
EOF

echo ""
echo "==> Ready: conda activate $ENV_NAME"
echo "    cd $PROJECT_ROOT"
echo "    export CUDA_VISIBLE_DEVICES=0"
echo "    python scripts/stage1/run_qwen_inference.py --task diagnosis --model qwen3-8b --limit 1"
python -c "import transformers, sentencepiece; print('transformers', transformers.__version__, '| sentencepiece OK')"
