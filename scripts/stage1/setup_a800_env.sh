#!/usr/bin/env bash
# Qwen3 + Gemma env for A800 (CUDA 12.x driver, 80GB VRAM).
# Use fp16 for Qwen3-14B — no AWQ/Triton hacks needed.
#
#   bash scripts/stage1/setup_a800_env.sh
#
# If synced from Windows and you see $'\r': command not found, run once:
#   sed -i 's/\r$//' scripts/stage1/setup_a800_env.sh scripts/stage1/run_stage2_a800_*.sh
#
set -eu

QWEN_ENV="${QWEN_ENV:-qwen3_infer}"
GEMMA_ENV="${GEMMA_ENV:-gemma_scope}"

_init_conda() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    return 0
  fi
  for d in "${CONDA_ROOT:-}" "$HOME/miniconda3" "$HOME/anaconda3" "/export/home/$(whoami)/miniconda3"; do
    [[ -z "${d}" ]] && continue
    if [[ -f "${d}/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "${d}/etc/profile.d/conda.sh"
      return 0
    fi
  done
  echo "ERROR: conda not found. Install miniconda or set CONDA_ROOT." >&2
  exit 1
}

_init_conda

_setup_env() {
  local name="$1"
  if ! conda env list | awk '{print $1}' | grep -qx "$name"; then
    conda create -n "$name" python=3.10 -y
  fi
  conda activate "$name"
  pip install -U pip wheel
}

echo "==> Qwen env: ${QWEN_ENV}"
_setup_env "${QWEN_ENV}"

# cu121 wheels work with driver CUDA 12.2
pip install torch==2.1.2 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

conda install -c conda-forge sentencepiece -y

pip install \
  "numpy>=1.26,<2" \
  "transformers>=4.51.0,<5.0" \
  "tokenizers>=0.21,<0.22" \
  accelerate \
  bitsandbytes \
  tqdm \
  "huggingface_hub>=0.26,<1.0"

python -c "import torch, transformers; print('qwen OK | torch', torch.__version__, '| transformers', transformers.__version__)"

echo ""
echo "==> Gemma env: ${GEMMA_ENV}"
conda deactivate
_setup_env "${GEMMA_ENV}"

pip install torch==2.1.2 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

pip install \
  "numpy>=1.26,<2" \
  "transformers>=4.44,<4.45" \
  accelerate bitsandbytes safetensors huggingface_hub tqdm einops

echo "==> Gemma Scope / SAE (judge imports sae_lens)"
conda install -y -c conda-forge "pandas>=2.0" "pyarrow>=14,<18" || true
pip install einops jaxtyping simple-parsing tenacity typing-extensions \
  docstring-parser wadler-lindig rich fsspec multiprocess xxhash dill aiohttp
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "datasets>=2.19,<3" sae-lens \
  || pip install "datasets>=2.19,<3" sae-lens --no-deps

python -c "import torch, transformers; from sae_lens import SAE; print('gemma OK | torch', torch.__version__, '| transformers', transformers.__version__, '| sae_lens OK')"

echo ""
echo "Done. Next on A800:"
echo "  conda activate ${QWEN_ENV}"
echo "  cd ~/MedRBench && bash scripts/stage1/run_stage2_a800_tmux.sh"
