#!/usr/bin/env bash
# Minimal datasets stub so autoawq can import (inference-only; no calibration).
# Avoids pandas/pyarrow/conda on CentOS 7.
#
#   bash scripts/stage1/install_awq_stub_datasets.sh
#
set -eu

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3_infer

SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
STUB="$SITE/datasets"

if python -c "import datasets; print(getattr(datasets,'__file__','builtin'))" 2>/dev/null | grep -q site-packages/datasets; then
  if python -c "import datasets; import datasets.load_dataset" 2>/dev/null; then
    echo "Real datasets already installed — skip stub"
  fi
fi

if [[ -d "$STUB" ]] && [[ -f "$STUB/__init__.py" ]] && grep -q "AWQ inference stub" "$STUB/__init__.py" 2>/dev/null; then
  echo "Stub datasets already present"
else
  if [[ -d "$STUB" ]]; then
    echo "Backing up existing $STUB -> ${STUB}.bak"
    rm -rf "${STUB}.bak"
    mv "$STUB" "${STUB}.bak"
  fi
  mkdir -p "$STUB"
  cat > "$STUB/__init__.py" << 'PY'
"""AWQ inference stub — real datasets not needed to load pre-quantized AWQ weights."""

__version__ = "0.0.0-stub"


def load_dataset(*args, **kwargs):
    raise RuntimeError(
        "datasets stub: load_dataset only needed for AWQ calibration, not inference"
    )


__all__ = ["load_dataset"]
PY
  echo "Wrote stub datasets to $STUB"
fi

if ! python -c "import awq" 2>/dev/null; then
  echo "Installing autoawq (--no-deps)..."
  pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple autoawq==0.2.9 \
    || pip install --no-deps autoawq==0.2.9
fi

if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
fi

echo ""
echo "==> Verify awq import"
python -W ignore::DeprecationWarning -c "import awq; print('awq OK')"

echo "==> triton: keep torch-bundled 2.1.x (do NOT upgrade to 3.x on CentOS 7 — needs stdatomic.h)"
python -c "import triton; print('triton', triton.__version__, '(AWQ uses naive matmul, not triton)')" || true

echo ""
echo "==> Quick AWQ load + 1-token generate (may download weights ~9GB first time)"
CUDA_VISIBLE_DEVICES=1 python - << 'PY'
import torch
from transformers import AutoTokenizer
mid = "Qwen/Qwen3-14B-AWQ"

def bootstrap_awq_centos7():
    try:
        import awq.modules.linear.gemm as gemm_mod
        gemm_mod.TRITON_AVAILABLE = False
        gemm_mod.awq_ext = None
    except Exception:
        pass

bootstrap_awq_centos7()
print("Loading", mid, "(triton disabled) ...")
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized(
    mid, device_map="auto", trust_remote_code=True, fuse_layers=False
)
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
dev = next(model.parameters()).device
ids = tok("Hello", return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=4, do_sample=False)
print("AWQ generate OK:", tok.decode(out[0], skip_special_tokens=True)[:80])
del model
torch.cuda.empty_cache()
print("Ready for --quant-mode awq")
PY

echo ""
echo "Next:"
echo "  CUDA_VISIBLE_DEVICES=1 python scripts/stage1/run_qwen_inference.py \\"
echo "    --model qwen3-14b-thinking --quant-mode awq --gpu-id 1 --limit 1 ..."
