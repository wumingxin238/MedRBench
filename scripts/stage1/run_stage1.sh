#!/usr/bin/env bash
# Stage-1 MedRBench experiment runner (GPU-P100 server).
#
# Prerequisites:
#   conda activate gemma_scope
#   cd ~/MedRBench
#   Gemma 2B + SAE downloaded; oracle_diagnosis.json at repo root
#
# Phases (run in tmux):
#   bash scripts/stage1/run_stage1.sh prepare-diagnosis
#   bash scripts/stage1/run_stage1.sh qwen-diagnosis-8b
#   bash scripts/stage1/run_stage1.sh qwen-diagnosis-14b
#   bash scripts/stage1/run_stage1.sh prepare-diagnosis   # merge qwen outputs
#   bash scripts/stage1/run_stage1.sh gemma-diagnosis-2b
#   bash scripts/stage1/run_stage1.sh gemma-diagnosis-9b   # after 9B model download

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"

PHASE="${1:-help}"

prepare_diagnosis() {
  python scripts/stage1/prepare_subject_outputs.py --task diagnosis
}

prepare_treatment() {
  python scripts/stage1/prepare_subject_outputs.py --task treatment \
    ${ORACLE_TREATMENT:+--strong-src "$ORACLE_TREATMENT"}
}

gemma_eval_diagnosis() {
  local SIZE="${1:-2b}"
  prepare_diagnosis
  for MODEL in deepseek-r1 o3-mini qwen3-8b qwen3-14b; do
    echo ">>> Gemma ${SIZE} / ${MODEL}"
    python scripts/stage1/run_gemma_scope_eval.py \
      --task diagnosis --gemma-size "$SIZE" --subject-model "$MODEL" || true
  done
}

case "$PHASE" in
  prepare-diagnosis)
    prepare_diagnosis
    ;;
  prepare-treatment)
    prepare_treatment
    ;;
  qwen-diagnosis-8b)
    python scripts/stage1/run_qwen_inference.py --task diagnosis --model qwen3-8b
    ;;
  qwen-diagnosis-14b)
    python scripts/stage1/run_qwen_inference.py --task diagnosis --model qwen3-14b
    ;;
  qwen-diagnosis-14b-gpu1)
    CUDA_VISIBLE_DEVICES=1 bash scripts/stage1/run_qwen14b_diagnosis_gpu1.sh
    ;;
  qwen-treatment-8b)
    python scripts/stage1/run_qwen_inference.py --task treatment --model qwen3-8b
    ;;
  qwen-treatment-14b)
    python scripts/stage1/run_qwen_inference.py --task treatment --model qwen3-14b
    ;;
  gemma-diagnosis-2b)
    gemma_eval_diagnosis 2b
    ;;
  gemma-diagnosis-9b)
    gemma_eval_diagnosis 9b
    ;;
  gemma-9b-three)
    bash scripts/stage1/run_gemma_9b_diagnosis.sh
    ;;
  *)
    cat <<EOF
Usage: bash scripts/stage1/run_stage1.sh <phase>

Phases:
  prepare-diagnosis     Slice strong oracle + merge weak inference JSON
  prepare-treatment     Same for treatment (needs ORACLE_TREATMENT=path)
  qwen-diagnosis-8b     Local Qwen3-8B on demo diagnosis 100
  qwen-diagnosis-14b    Local Qwen3-14B on demo diagnosis 100
  qwen-diagnosis-14b-gpu1  Same, pinned to GPU 1 (qwen3_infer env)
  qwen-treatment-8b     Local Qwen3-8B on demo treatment 100
  qwen-treatment-14b    Local Qwen3-14B on demo treatment 100
  gemma-diagnosis-2b    Gemma 2B eval: 4 models x 2 groups x 100 cases
  gemma-diagnosis-9b    Gemma 9B eval (4bit; requires full 9B download)

Outputs:
  data/Stage1/oracle_diagnosis_subjects.json
  data/Stage1/inference/qwen3-*_{diagnosis,treatment}.json
  data/Stage1/gemma_scope/diagnosis_{2b,9b}_<model>.json
EOF
    ;;
esac
