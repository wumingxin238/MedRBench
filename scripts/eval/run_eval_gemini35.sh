#!/bin/bash
# Evaluate gemini2-ft on 35 cases with local Qwen judge
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT/src/Evaluation"

if [ -f "$PROJECT_ROOT/scripts/server/config/eval_config.env" ]; then
  # shellcheck source=/dev/null
  . "$PROJECT_ROOT/scripts/server/config/eval_config.env"
fi

MODEL_UNDER_TEST="${MODEL_UNDER_TEST:-gemini2-ft}"
PATIENT_CASES="${PATIENT_CASES:-$PROJECT_ROOT/data/MedRBench/test_cases.json}"
MODEL_OUTPUTS="${MODEL_OUTPUTS:-$PROJECT_ROOT/src/Inference/oracle_diagnosis_gemini.json}"
ACC_OUT="${ACC_OUT:-$PROJECT_ROOT/src/Evaluation/acc_results_qwen_judge}"
REASON_OUT="${REASON_OUT:-$PROJECT_ROOT/src/Evaluation/reasoning_results_qwen_judge}"

echo "==> Evaluator: backend=${EVAL_BACKEND:-openai} model=${EVAL_MODEL:-gpt-4o}"
echo "==> Subject: $MODEL_UNDER_TEST"

python oracle_diagnose_accuracy.py \
  --model "$MODEL_UNDER_TEST" \
  --sequential \
  --embedded-outputs \
  --patient-cases "$PATIENT_CASES" \
  --model-outputs "$MODEL_OUTPUTS" \
  --output-dir "$ACC_OUT"

python oracle_diagnose_reasoning.py \
  --model "$MODEL_UNDER_TEST" \
  --sequential \
  --no-web-search \
  --embedded-outputs \
  --patient-cases "$PATIENT_CASES" \
  --model-outputs "$MODEL_OUTPUTS" \
  --output-dir "$REASON_OUT"

echo "==> Done: $ACC_OUT/$MODEL_UNDER_TEST/ and $REASON_OUT/$MODEL_UNDER_TEST/"
