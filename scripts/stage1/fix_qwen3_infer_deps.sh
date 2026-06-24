#!/usr/bin/env bash
# Fix qwen3_infer for Stage-2 Qwen3-14B AWQ on P100 (CentOS 7 — no pip source builds).
#
#   bash scripts/stage1/fix_qwen3_infer_deps.sh
#
set -eu
cd "$(dirname "$0")/../.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3_infer

pip install -U pip wheel

echo "==> NVIDIA CUDA libs (bitsandbytes fallback)"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cusparse-cu11 \
  2>/dev/null || pip install nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cusparse-cu11

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<'EOF'
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparse/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
EOF
# shellcheck source=/dev/null
source "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

echo "==> pandas + pyarrow via conda-forge (never pip-compile on GCC 4.8)"
if ! python -c "import pandas, pyarrow" 2>/dev/null; then
  conda install -y -c conda-forge "pandas==2.0.3" "pyarrow>=14,<18" || \
    conda install -y -c conda-forge pandas pyarrow
fi
python -c "import pandas; import pyarrow; print('pandas', pandas.__version__, '| pyarrow', pyarrow.__version__)"

echo "==> datasets (pip --no-deps — pandas/pyarrow already from conda)"
if ! python -c "import datasets" 2>/dev/null; then
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "multiprocess" "xxhash" "fsspec" "aiohttp" "dill" \
    || pip install multiprocess xxhash fsspec aiohttp dill
  pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple "datasets==2.19.0" \
    || pip install --no-deps "datasets==2.19.0"
fi
python -c "import datasets; print('datasets', datasets.__version__)"

echo "==> autoawq"
if ! python -c "import awq" 2>/dev/null; then
  pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple "autoawq==0.2.9" \
    || pip install --no-deps "autoawq==0.2.9"
fi

echo ""
echo "==> Verify"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -W ignore::DeprecationWarning -c "import awq; print('awq OK')" || {
  echo "awq import failed — see traceback above"
  exit 1
}

if python -c "import bitsandbytes" 2>/dev/null; then
  echo "bitsandbytes OK (4bit fallback available)"
else
  echo "bitsandbytes: optional fallback not available (AWQ is enough)"
fi

echo ""
echo "Done. Next:"
echo "  bash scripts/stage1/check_stage2_deps.sh"
echo "  STAGE2_FORCE=1 bash scripts/stage1/run_stage2_tmux.sh"
