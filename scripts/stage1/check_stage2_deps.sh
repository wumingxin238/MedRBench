#!/usr/bin/env bash
# Preflight before Stage-2 parallel run.
set -eu
cd "$(dirname "$0")/../.."

source "$(conda info --base)/etc/profile.d/conda.sh"

ok=0
warn=0
fail() { echo "FAIL: $*" >&2; ok=1; }
warn_msg() { echo "WARN: $*" >&2; warn=1; }

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || fail "nvidia-smi"

echo ""
echo "=== qwen3_infer (Qwen3-14B infer, GPU1) ==="
has_awq=0
has_bnb=0
if conda activate qwen3_infer 2>/dev/null; then
  python -V
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || fail "torch"
  python -c "import transformers; print('transformers', transformers.__version__)" || fail "transformers"
  if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
  fi
  if python -c "import awq" 2>/dev/null; then
    echo "autoawq: OK (recommended)"
    has_awq=1
  else
    echo "autoawq: MISSING — run: bash scripts/stage1/fix_qwen3_infer_deps.sh"
  fi
  if python -c "import bitsandbytes" 2>/dev/null; then
    echo "bitsandbytes: OK (4bit fallback)"
    has_bnb=1
  else
    echo "bitsandbytes: broken (libcusparse?) — install autoawq instead"
  fi
  if [[ "${has_awq}" -eq 0 && "${has_bnb}" -eq 0 ]]; then
    fail "Neither autoawq nor bitsandbytes works — run: bash scripts/stage1/fix_qwen3_infer_deps.sh"
  elif [[ "${has_awq}" -eq 0 ]]; then
    warn_msg "autoawq missing; infer relies on bitsandbytes 4bit only"
  fi
else
  fail "conda env qwen3_infer missing — run: bash scripts/stage1/setup_qwen_env.sh"
fi

echo ""
echo "=== gemma_scope (Gemma-9B judge, GPU0) ==="
if conda activate gemma_scope 2>/dev/null; then
  python -V
  CUDA_VISIBLE_DEVICES=0 python -c "import torch; x=torch.zeros(1, device='cuda:0'); print('GPU0 ok')" \
    || fail "GPU0 not usable (kill stale processes or reset)"
else
  fail "conda env gemma_scope missing"
fi

echo ""
echo "=== eval API (accuracy judge) ==="
cfg="scripts/server/config/eval_config.env"
if [[ -f "${cfg}" ]]; then
  # shellcheck source=/dev/null
  source <(sed 's/\r$//' "${cfg}")
  echo "EVAL_MODEL=${EVAL_MODEL:-unset}"
  echo "EVAL_BASE_URL=${EVAL_BASE_URL:-unset}"
  if [[ "${EVAL_BASE_URL:-}" == http://127.0.0.1:* ]] || [[ "${EVAL_BASE_URL:-}" == http://localhost:* ]]; then
    warn_msg "Local judge URL — ensure vLLM/Ollama is running, or set OpenAI gpt-5 in eval_config.env"
  fi
else
  warn_msg "${cfg} missing — acc-api needs EVAL_BASE_URL + EVAL_API_KEY"
fi

echo ""
if [[ "${ok}" -eq 0 ]]; then
  if [[ "${warn}" -eq 1 ]]; then
    echo "Preflight OK with warnings. Review WARN lines above."
  else
    echo "Preflight OK. Start: bash scripts/stage1/run_stage2_tmux.sh"
  fi
else
  echo "Fix FAIL items before starting Stage-2."
  exit 1
fi
