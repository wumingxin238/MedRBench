#!/usr/bin/env bash
# Install gemma_scope deps for Gemma-9B judge + SAE (A800).
#
#   bash scripts/stage1/fix_gemma_scope_a800.sh
#
set -eu

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gemma_scope

pip install -U pip wheel

echo "==> pyarrow/pandas via conda-forge (avoid pip compile)"
conda install -y -c conda-forge "pandas>=2.0" "pyarrow>=14,<18" || \
  conda install -y -c conda-forge pandas pyarrow

echo "==> sae-lens + runtime deps"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  einops jaxtyping simple-parsing tenacity typing-extensions \
  docstring-parser wadler-lindig rich safetensors huggingface_hub fsspec \
  multiprocess xxhash dill aiohttp \
  || pip install einops jaxtyping simple-parsing tenacity typing-extensions \
  docstring-parser wadler-lindig rich safetensors huggingface_hub fsspec \
  multiprocess xxhash dill aiohttp

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "datasets>=2.19,<3" \
  || pip install "datasets>=2.19,<3"

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sae-lens \
  || pip install sae-lens --no-deps

echo ""
echo "==> Verify imports"
python - <<'PY'
import torch
import transformers
from sae_lens import SAE
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("sae_lens OK")
PY

echo ""
echo "Done. Re-run Gemma:"
echo "  GEMMA_GPU=3 bash scripts/stage1/run_stage2_a800_parallel.sh gemma"
