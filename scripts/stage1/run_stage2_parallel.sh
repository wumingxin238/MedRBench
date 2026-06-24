#!/usr/bin/env bash
# Stage-2 parallel workflow (2× GPU + API).
#
# One-shot tmux (recommended):
#   bash scripts/stage1/run_stage2_tmux.sh
#
# Manual panes:
#   bash scripts/stage1/run_stage2_parallel.sh infer-gpu1   # GPU1, qwen3_infer
#   bash scripts/stage1/run_stage2_parallel.sh gemma-gpu0   # GPU0, gemma_scope
#   bash scripts/stage1/run_stage2_parallel.sh acc-api      # API, no GPU
#
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"

STAGE2_MANIFEST="${STAGE2_MANIFEST:-data/MedRBench/stage2_manifest.json}"
CASES_400="${CASES_400:-data/MedRBench/diagnosis_400.json}"
SUBJECTS="${SUBJECTS:-data/Stage2/oracle_diagnosis_subjects.json}"
INFER_OUT="${INFER_OUT:-data/Stage2/inference/qwen3-14b-thinking_diagnosis.json}"
INFER_DIR="${INFER_DIR:-data/Stage2/inference}"
RE_OUT="${RE_OUT:-data/Stage2/reasoning_eval}"
ACC_OUT="${ACC_OUT:-data/Stage2/acc_results_gpt}"
ACC_WORKERS="${ACC_WORKERS:-8}"
PY="${STAGE2_PY:-}"
QWEN_QUANT_MODE="${QWEN_QUANT_MODE:-}"

_init_conda() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    return 0
  fi
  for d in "${CONDA_ROOT:-}" "$HOME/miniconda3" "$HOME/anaconda3"; do
    [[ -z "${d}" ]] && continue
    if [[ -f "${d}/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "${d}/etc/profile.d/conda.sh"
      return 0
    fi
  done
  echo "ERROR: conda not found. Set CONDA_ROOT or install miniconda." >&2
  exit 1
}

_ensure_python() {
  _init_conda
  if [[ -n "${PY}" ]] && "${PY}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    return 0
  fi
  conda activate qwen3_infer 2>/dev/null || conda activate gemma_scope
  PY=python
}

_run_py() {
  _ensure_python
  "${PY}" "$@"
}

_activate_qwen() {
  _init_conda
  conda activate qwen3_infer
  if [[ -f "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
  fi
  PY=python
  if "${PY}" -c "import awq" 2>/dev/null; then
    QWEN_QUANT_MODE="${QWEN_QUANT_MODE:-awq}"
  elif "${PY}" -c "import bitsandbytes" 2>/dev/null; then
    QWEN_QUANT_MODE="${QWEN_QUANT_MODE:-4bit}"
    echo "WARN: autoawq unavailable; using bitsandbytes 4bit (base Qwen3-14B)" >&2
  else
    echo "ERROR: need autoawq OR bitsandbytes." >&2
    echo "  Wheels: bash scripts/stage1/install_awq_from_wheels.sh" >&2
    echo "  4bit:   bash scripts/stage1/run_stage2_bnb_fallback.sh" >&2
    exit 1
  fi
}

_activate_gemma() {
  _init_conda
  conda activate gemma_scope
  PY=python
}

_source_eval_config() {
  local cfg="scripts/server/config/eval_config.env"
  [[ -f "${cfg}" ]] || return 0
  set -a
  # shellcheck source=/dev/null
  source <(sed 's/\r$//' "${cfg}")
  set +a
}

_prep() {
  _ensure_python
  echo "Using ${PY} ($("${PY}" -V 2>&1))"
  _run_py scripts/stage1/build_stage2_hard_subset.py
  mkdir -p data/Stage2/inference data/Stage2/reasoning_eval data/Stage2/acc_results_gpt
}

_merge_subjects() {
  _run_py scripts/stage1/prepare_subject_outputs.py \
    --task diagnosis \
    --manifest "${STAGE2_MANIFEST}" \
    --cases "${CASES_400}" \
    --out "${SUBJECTS}" \
    --strong-src oracle_diagnosis.json \
    --inference-dir "${INFER_DIR}" \
    --weak-models qwen3-14b-thinking
}

_infer_count() {
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${INFER_OUT}")
print(len(json.loads(p.read_text())) if p.is_file() else 0)
PY
}

_wait_infer_started() {
  local min="${1:-1}"
  while true; do
    n=$(_infer_count 2>/dev/null || echo 0)
    if [[ "${n}" -ge "${min}" ]]; then
      echo "Inference started (${n} cases in ${INFER_OUT})"
      return 0
    fi
    echo "Waiting for inference (${n}/${min}) ..."
    sleep 30
  done
}

_infer_gpu1() {
  _activate_qwen
  export CUDA_VISIBLE_DEVICES=1
  echo "=== [infer] GPU1  env=qwen3_infer  quant=${QWEN_QUANT_MODE} ==="
  nvidia-smi || true
  mkdir -p "${INFER_DIR}"
  "${PY}" scripts/stage1/run_qwen_inference.py \
    --task diagnosis \
    --model qwen3-14b-thinking \
    --cases "${CASES_400}" \
    --manifest "${STAGE2_MANIFEST}" \
    --out "${INFER_OUT}" \
    --gpu-id 1 \
    --quant-mode "${QWEN_QUANT_MODE}"
  _merge_subjects
}

_gemma_gpu0() {
  _activate_gemma
  export CUDA_VISIBLE_DEVICES=0
  export GEMMA_JUDGE_9B_MODE=4bit
  export EVAL_DISABLE_WEB_SEARCH=1
  echo "=== [gemma] GPU0  env=gemma_scope  (poll infer, incremental eval) ==="
  sleep "${GEMMA_START_DELAY:-20}"
  _wait_infer_started 1
  nvidia-smi || true
  while true; do
    _merge_subjects
    n=$(_infer_count 2>/dev/null || echo 0)
    echo "--- Gemma pass (infer ${n}/400 merged) ---"
    "${PY}" scripts/stage1/run_stage1_reasoning_eval.py \
      --task diagnosis \
      --subject-model qwen3-14b-thinking \
      --judge gemma-local \
      --gemma-size 9b \
      --group both \
      --cases "${CASES_400}" \
      --manifest "${STAGE2_MANIFEST}" \
      --outputs "${SUBJECTS}" \
      --out-dir "${RE_OUT}" \
      --gpu-id 0
    ok=$("${PY}" - <<PY
import json
from pathlib import Path
need = len(json.loads(Path("${STAGE2_MANIFEST}").read_text())["diagnosis"]["case_ids"])
for g in ("direct", "inference_augmented"):
    p = Path("${RE_OUT}") / f"diagnosis_gemma-9b-it_qwen3-14b-thinking_{g}.json"
    if not p.is_file():
        print(0); raise SystemExit
    d = json.loads(p.read_text())
    ok = sum(1 for c in d.get("cases", {}).values() if c.get("status") == "ok")
    if ok < need:
        print(0); raise SystemExit
print(1)
PY
)
    if [[ "${ok}" == "1" ]]; then
      echo "Gemma-9B reasoning_eval complete (400/400)."
      break
    fi
    echo "Gemma pass done; waiting for more infer (120s)..."
    sleep 120
  done
}

_acc_api() {
  _ensure_python
  _source_eval_config
  export EVAL_DISABLE_WEB_SEARCH=1
  : "${EVAL_MODEL:=gpt-5}"
  echo "=== [acc] judge=${EVAL_MODEL}  workers=${ACC_WORKERS} ==="
  sleep "${ACC_START_DELAY:-40}"
  while true; do
    _merge_subjects
    n_subj=$("${PY}" -c "
import json
from pathlib import Path
o=json.loads(Path('${SUBJECTS}').read_text())
print(sum(1 for c in o.values() if 'qwen3-14b-thinking' in c))
" 2>/dev/null || echo 0)
    if [[ "${n_subj}" -lt 1 ]]; then
      echo "No qwen outputs yet (${n_subj}); wait 60s..."
      sleep 60
      continue
    fi
    echo "--- Accuracy pass (${n_subj} subjects ready) ---"
    "${PY}" scripts/stage1/run_stage2_diagnosis_accuracy.py \
      --cases "${CASES_400}" \
      --outputs "${SUBJECTS}" \
      --out-dir "${ACC_OUT}" \
      --eval-model "${EVAL_MODEL}" \
      --workers "${ACC_WORKERS}" || true
    n=$(find "${ACC_OUT}/qwen3-14b-thinking" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    need=$("${PY}" -c "import json; print(len(json.load(open('${CASES_400}'))))")
    echo "Accuracy files: ${n}/${need}"
    if [[ "${n}" -ge "${need}" ]]; then
      echo "Accuracy complete."
      break
    fi
    sleep 180
  done
}

_status() {
  _run_py - <<'PY'
import json
from pathlib import Path

def count_ok(p, key="status"):
    if not p.is_file():
        return "missing"
    d = json.loads(p.read_text(encoding="utf-8"))
    if "cases" in d:
        ok = sum(1 for c in d["cases"].values() if c.get(key) == "ok")
        return f"{ok}/{len(d['cases'])}"
    return str(len(d))

m = Path("data/MedRBench/stage2_manifest.json")
if m.is_file():
    man = json.loads(m.read_text(encoding="utf-8"))
    print(f"Stage2 manifest: {man['counts']['diagnosis']} diagnosis cases")

inf = Path("data/Stage2/inference/qwen3-14b-thinking_diagnosis.json")
if inf.is_file():
    print(f"  infer: {len(json.loads(inf.read_text()))}/400 cases")

subj = Path("data/Stage2/oracle_diagnosis_subjects.json")
if subj.is_file():
    o = json.loads(subj.read_text())
    n = sum(1 for c in o.values() if "qwen3-14b-thinking" in c)
    print(f"  qwen subjects: {n}/400")

for label, p in [
    ("re direct", Path("data/Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-14b-thinking_direct.json")),
    ("re aug", Path("data/Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-14b-thinking_inference_augmented.json")),
]:
    print(f"  {label}: {count_ok(p)}")

acc = Path("data/Stage2/acc_results_gpt/qwen3-14b-thinking")
if acc.is_dir():
    fs = list(acc.glob("*.json"))
    ok = sum(1 for f in fs if json.loads(f.read_text(encoding="utf-8")).get("accuracy"))
    print(f"  acc files: {len(fs)}/400  correct={ok}")
PY
}

case "${1:-help}" in
  prep) _prep ;;
  merge) _merge_subjects ;;
  infer-gpu1) _infer_gpu1 ;;
  gemma-gpu0) _gemma_gpu0 ;;
  acc-api) _acc_api ;;
  status) _status ;;
  help|*)
    cat <<EOF
Stage-2 parallel (tmux: bash scripts/stage1/run_stage2_tmux.sh)

  prep          Build 400-case JSON + manifest
  infer-gpu1    Qwen3-14B-thinking AWQ on physical GPU1
  gemma-gpu0    Gemma-9B judge on physical GPU0 (incremental)
  acc-api       Accuracy judge via EVAL_* (parallel API workers)
  status        Progress

GPU map: infer=GPU1  gemma=GPU0  acc=no GPU
Fix deps: bash scripts/stage1/fix_qwen3_infer_deps.sh
EOF
    ;;
esac
