#!/usr/bin/env bash
# Try to get bitsandbytes >= 0.43 on P100 (pip index often caps at 0.42).
set -eu

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV:-qwen3_infer}"

echo "Current bitsandbytes:"
python -c "import bitsandbytes as b; print(b.__version__)" 2>/dev/null || echo "not installed"

ver_ok() {
  python - <<'PY'
import bitsandbytes as b
parts = tuple(int(x) for x in b.__version__.split(".")[:3])
raise SystemExit(0 if parts >= (0, 43, 2) else 1)
PY
}

if ver_ok; then
  echo "Already >= 0.43.2"
  exit 0
fi

echo "==> Try conda-forge"
conda install -y -c conda-forge bitsandbytes || true
if ver_ok; then exit 0; fi

echo "==> Try pip pre-release / latest binary"
pip install -U 'bitsandbytes>=0.43.2' --only-binary :all: || true
if ver_ok; then exit 0; fi

echo "==> Try GitHub source (needs CUDA toolkit / may take several minutes)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export BNB_CUDA_VERSION=118
pip install --no-cache-dir "git+https://github.com/TimDettmers/bitsandbytes.git@v0.43.3" || true
if ver_ok; then exit 0; fi

echo "FAILED: still on bitsandbytes $(python -c 'import bitsandbytes as b; print(b.__version__)' 2>/dev/null || echo '?')"
echo "14B 4bit may fail on dispatch with bnb 0.42. Options:"
echo "  1) Wait for Gemma 9B to finish, then --quant-mode fp16-split"
echo "  2) Run 14B on another machine"
exit 1
