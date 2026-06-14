#!/usr/bin/env bash
# Gemma-9B reasoning_eval: o3-mini on GPU0 + deepseek-r1 on GPU1 (parallel).
# qwen3-8b already done — skip unless you pass --include-qwen.
#
# XShell usage:
#   tmux new -s gemma9b_re
#   Window 0: bash scripts/stage1/run_gemma9b_reasoning_eval_dual.sh gpu0
#   Ctrl+b c
#   Window 1: bash scripts/stage1/run_gemma9b_reasoning_eval_dual.sh gpu1
#
# Or one-shot status:
#   bash scripts/stage1/run_gemma9b_reasoning_eval_dual.sh status
#
set -eu
cd "$(dirname "$0")/../.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gemma_scope

export GEMMA_JUDGE_9B_MODE=4bit
export EVAL_DISABLE_WEB_SEARCH=1

_run_one() {
  local gpu="$1"
  local model="$2"
  echo ">>> GPU${gpu}  subject=${model}  group=both"
  python scripts/stage1/run_stage1_reasoning_eval.py \
    --task diagnosis \
    --subject-model "${model}" \
    --judge gemma-local \
    --gemma-size 9b \
    --group both \
    --gpu-id "${gpu}"
}

_status() {
  python -c "
import json
from pathlib import Path
for p in sorted(Path('data/Stage1/reasoning_eval').glob('diagnosis_gemma-9b-it_*.json')):
    d = json.loads(p.read_text(encoding='utf-8'))
    ok = sum(1 for c in d['cases'].values() if c.get('status') == 'ok')
    err = sum(1 for c in d['cases'].values() if c.get('status') == 'error')
    m = d.get('meta', {})
    print(f'{p.name}: {ok}/100 ok, {err} err  meta_ok={m.get(\"completed_ok\")}  eff={m.get(\"mean_efficiency\")}')
"
}

case "${1:-help}" in
  gpu0)
    _run_one 0 o3-mini
    ;;
  gpu1)
    _run_one 1 deepseek-r1
    ;;
  status)
    _status
    ;;
  prepare)
    python scripts/stage1/prepare_subject_outputs.py --task diagnosis
    _status
    ;;
  help|*)
    cat <<'EOF'
Usage:
  bash scripts/stage1/run_gemma9b_reasoning_eval_dual.sh prepare   # merge oracle subjects
  bash scripts/stage1/run_gemma9b_reasoning_eval_dual.sh status    # progress
  bash scripts/stage1/run_gemma9b_reasoning_eval_dual.sh gpu0     # tmux window 0: o3-mini
  bash scripts/stage1/run_gemma9b_reasoning_eval_dual.sh gpu1     # tmux window 1: deepseek-r1

Outputs (new):
  data/Stage1/reasoning_eval/diagnosis_gemma-9b-it_o3-mini_{direct,inference_augmented}.json
  data/Stage1/reasoning_eval/diagnosis_gemma-9b-it_deepseek-r1_{direct,inference_augmented}.json

Already done (skip):
  diagnosis_gemma-9b-it_qwen3-8b_*.json
EOF
    ;;
esac
