#!/usr/bin/env bash
# Stage-2 on A800: N-way Qwen infer shards + Gemma judge + API accuracy.
#
# Default GPU map (skip busy GPU 6 if set):
#   infer shards → INFER_GPUS (default 0,1,2,7)  fp16 Qwen3-14B-thinking
#   gemma judge  → GEMMA_GPU (default 3)
#   accuracy     → API workers (no GPU)
#
#   bash scripts/stage1/run_stage2_a800_parallel.sh status
#   bash scripts/stage1/run_stage2_a800_parallel.sh infer-shard 0 0
#   bash scripts/stage1/run_stage2_a800_parallel.sh merge
#   gemma                   Single-GPU Gemma (legacy)
#   gemma-shard IDX GPU     One Gemma shard (e.g. gemma-shard 0 0)
#   gemma-parallel          All Gemma shards on GEMMA_GPUS + merge
#   merge-gemma             Merge Gemma shard JSON → canonical
#
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"

STAGE2_MANIFEST="${STAGE2_MANIFEST:-data/MedRBench/stage2_manifest.json}"
CASES_400="${CASES_400:-data/MedRBench/diagnosis_400.json}"
SUBJECTS="${SUBJECTS:-data/Stage2/oracle_diagnosis_subjects.json}"
INFER_DIR="${INFER_DIR:-data/Stage2/inference}"
INFER_OUT="${INFER_OUT:-${INFER_DIR}/qwen3-14b-thinking_diagnosis.json}"
RE_OUT="${RE_OUT:-data/Stage2/reasoning_eval}"
ACC_OUT="${ACC_OUT:-data/Stage2/acc_results_gpt}"
ACC_GEMMA_OUT="${ACC_GEMMA_OUT:-data/Stage2/acc_results_gemma}"
ACC_WORKERS="${ACC_WORKERS:-8}"
MODEL="${STAGE2_MODEL:-qwen3-14b-thinking}"
TASK="${STAGE2_TASK:-diagnosis}"
QWEN_QUANT_MODE="${QWEN_QUANT_MODE:-fp16}"
INFER_GPUS="${INFER_GPUS:-0,1,2,7}"
GEMMA_GPU="${GEMMA_GPU:-3}"
GEMMA_GPUS="${GEMMA_GPUS:-0,1,2}"
GEMMA_JUDGE_FULL_GPU="${GEMMA_JUDGE_FULL_GPU:-1}"
SKIP_GPUS="${SKIP_GPUS:-6}"
# Comma-separated subjects for gemma-all / acc-all (14B usually done)
STAGE2_GEMMA_SUBJECTS="${STAGE2_GEMMA_SUBJECTS:-o3-mini,deepseek-r1}"
STAGE2_ACC_SUBJECTS="${STAGE2_ACC_SUBJECTS:-o3-mini,deepseek-r1}"
STAGE2_ACC_GEMMA_SUBJECTS="${STAGE2_ACC_GEMMA_SUBJECTS:-o3-mini,deepseek-r1,qwen3-14b-thinking}"
PY=""

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
  echo "ERROR: conda not found" >&2
  exit 1
}

_activate_qwen() {
  _init_conda
  conda activate qwen3_infer
  PY=python
}

_activate_gemma() {
  _init_conda
  conda activate gemma_scope
  PY="$(command -v python)"
  echo "gemma_scope python: ${PY}" >&2
  "${PY}" -c "import torch; print('torch', torch.__version__, file=__import__('sys').stderr)" || exit 1
}

_run_py() {
  _activate_qwen
  "${PY}" "$@"
}

_parse_infer_gpus() {
  INFER_GPU_LIST=()
  IFS=',' read -ra _raw <<< "${INFER_GPUS}"
  for g in "${_raw[@]}"; do
    g="${g// /}"
    [[ -z "${g}" ]] && continue
    if [[ -n "${SKIP_GPUS}" ]] && [[ ",${SKIP_GPUS}," == *",${g},"* ]]; then
      echo "Skipping GPU ${g} (SKIP_GPUS)" >&2
      continue
    fi
    INFER_GPU_LIST+=("${g}")
  done
  NUM_SHARDS="${#INFER_GPU_LIST[@]}"
  if [[ "${NUM_SHARDS}" -lt 1 ]]; then
    echo "ERROR: no infer GPUs after SKIP_GPUS filter (INFER_GPUS=${INFER_GPUS})" >&2
    exit 1
  fi
  echo "Infer shards: ${NUM_SHARDS} on GPUs ${INFER_GPU_LIST[*]} (quant=${QWEN_QUANT_MODE})"
}

_shard_out() {
  local idx="$1"
  echo "${INFER_DIR}/${MODEL}_${TASK}.shard${idx}.json"
}

_prep() {
  _run_py scripts/stage1/build_stage2_hard_subset.py
  mkdir -p "${INFER_DIR}" "${RE_OUT}" "${ACC_OUT}"
}

_merge_subjects() {
  _run_py scripts/stage1/prepare_subject_outputs.py \
    --task "${TASK}" \
    --manifest "${STAGE2_MANIFEST}" \
    --cases "${CASES_400}" \
    --out "${SUBJECTS}" \
    --strong-src oracle_diagnosis.json \
    --inference-dir "${INFER_DIR}" \
    --weak-models "${MODEL}"
}

_merge_shards() {
  _parse_infer_gpus
  _run_py scripts/stage1/merge_qwen_inference_shards.py \
    --model "${MODEL}" \
    --task "${TASK}" \
    --num-shards "${NUM_SHARDS}" \
    --inference-dir "${INFER_DIR}" \
    --out "${INFER_OUT}"
  _merge_subjects
}

_infer_count() {
  _activate_qwen
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${INFER_OUT}")
print(len(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else 0)
PY
}

_shard_count() {
  local idx="$1"
  _activate_qwen
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("$(_shard_out "${idx}")")
print(len(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else 0)
PY
}

_total_shard_cases() {
  _parse_infer_gpus
  local sum=0 n
  for ((i=0; i<NUM_SHARDS; i++)); do
    n=$(_shard_count "${i}" 2>/dev/null || echo 0)
    sum=$((sum + n))
  done
  echo "${sum}"
}

_infer_shard() {
  local shard_idx="${1:?shard index}"
  local physical_gpu="${2:?physical GPU}"
  _activate_qwen
  export CUDA_VISIBLE_DEVICES="${physical_gpu}"
  local out
  out=$(_shard_out "${shard_idx}")
  echo "=== [infer shard ${shard_idx}] GPU${physical_gpu} → ${out} ==="
  nvidia-smi || true
  "${PY}" scripts/stage1/run_qwen_inference.py \
    --task "${TASK}" \
    --model "${MODEL}" \
    --cases "${CASES_400}" \
    --manifest "${STAGE2_MANIFEST}" \
    --out "${out}" \
    --shard-index "${shard_idx}" \
    --num-shards "${NUM_SHARDS}" \
    --quant-mode "${QWEN_QUANT_MODE}"
  echo "=== shard ${shard_idx} done ==="
}

_infer_all() {
  _parse_infer_gpus
  local idx gpu
  for ((idx=0; idx<NUM_SHARDS; idx++)); do
    gpu="${INFER_GPU_LIST[$idx]}"
    _infer_shard "${idx}" "${gpu}"
  done
  _merge_shards
}

_parse_gemma_gpus() {
  GEMMA_GPU_LIST=()
  IFS=',' read -ra _raw <<< "${GEMMA_GPUS}"
  for g in "${_raw[@]}"; do
    g="${g// /}"
    [[ -z "${g}" ]] && continue
    if [[ -n "${SKIP_GPUS}" ]] && [[ ",${SKIP_GPUS}," == *",${g},"* ]]; then
      echo "Skipping Gemma GPU ${g} (SKIP_GPUS)" >&2
      continue
    fi
    GEMMA_GPU_LIST+=("${g}")
  done
  GEMMA_NUM_SHARDS="${#GEMMA_GPU_LIST[@]}"
  if [[ "${GEMMA_NUM_SHARDS}" -lt 1 ]]; then
    echo "ERROR: no Gemma GPUs after SKIP_GPUS filter (GEMMA_GPUS=${GEMMA_GPUS})" >&2
    exit 1
  fi
  echo "Gemma shards: ${GEMMA_NUM_SHARDS} on GPUs ${GEMMA_GPU_LIST[*]} (full_gpu=${GEMMA_JUDGE_FULL_GPU})"
}

_gemma_env() {
  export GEMMA_JUDGE_9B_MODE="${GEMMA_JUDGE_9B_MODE:-fp16}"
  export GEMMA_JUDGE_FULL_GPU="${GEMMA_JUDGE_FULL_GPU}"
  export EVAL_DISABLE_WEB_SEARCH=1
}

_gemma_shard() {
  local shard_idx="${1:?shard index}"
  local physical_gpu="${2:?physical GPU}"
  local subject="${3:-${GEMMA_SUBJECT:-${MODEL}}}"
  _activate_gemma
  _gemma_env
  export CUDA_VISIBLE_DEVICES="${physical_gpu}"
  _parse_gemma_gpus
  echo "=== [gemma ${subject} shard ${shard_idx}/${GEMMA_NUM_SHARDS}] GPU${physical_gpu} ==="
  nvidia-smi || true
  "${PY}" scripts/stage1/run_stage1_reasoning_eval.py \
    --task "${TASK}" \
    --subject-model "${subject}" \
    --judge gemma-local \
    --gemma-size 9b \
    --group both \
    --cases "${CASES_400}" \
    --manifest "${STAGE2_MANIFEST}" \
    --outputs "${SUBJECTS}" \
    --out-dir "${RE_OUT}" \
    --shard-index "${shard_idx}" \
    --num-shards "${GEMMA_NUM_SHARDS}"
  echo "=== gemma ${subject} shard ${shard_idx} done ==="
}

_merge_gemma() {
  local subject="${1:-${GEMMA_SUBJECT:-${MODEL}}}"
  _parse_gemma_gpus
  _activate_gemma
  echo "=== merge-gemma ${subject} ==="
  "${PY}" scripts/stage1/merge_gemma_reasoning_shards.py \
    --task "${TASK}" \
    --model "${subject}" \
    --num-shards "${GEMMA_NUM_SHARDS}" \
    --out-dir "${RE_OUT}"
}

_gemma_parallel() {
  local subject="${1:-${GEMMA_SUBJECT:-${MODEL}}}"
  _parse_gemma_gpus
  local idx gpu
  for ((idx=0; idx<GEMMA_NUM_SHARDS; idx++)); do
    gpu="${GEMMA_GPU_LIST[$idx]}"
    _gemma_shard "${idx}" "${gpu}" "${subject}" &
  done
  wait
  _merge_gemma "${subject}"
}

_gemma_re_ok() {
  local subject="${1:?subject}"
  local need="${2:-400}"
  _activate_gemma
  "${PY}" - <<PY
import json
from pathlib import Path
need = ${need}
for g in ("direct", "inference_augmented"):
    p = Path("${RE_OUT}") / f"${TASK}_gemma-9b-it_${subject}_{g}.json"
    if not p.is_file():
        raise SystemExit(1)
    d = json.loads(p.read_text(encoding="utf-8"))
    ok = sum(1 for c in d.get("cases", {}).values() if c.get("status") == "ok")
    if ok < need:
        raise SystemExit(1)
PY
}

_gemma_all() {
  _activate_gemma
  local need subjects=() s
  need=$("${PY}" -c "import json; print(len(json.load(open('${STAGE2_MANIFEST}'))['${TASK}']['case_ids']))")
  IFS=',' read -ra subjects <<< "${STAGE2_GEMMA_SUBJECTS}"
  for s in "${subjects[@]}"; do
    s="${s// /}"
    [[ -z "${s}" ]] && continue
    if _gemma_re_ok "${s}" "${need}" 2>/dev/null; then
      echo "Gemma ${s}: already ${need}/${need}, skip"
      continue
    fi
    echo ">>> Gemma parallel: ${s}"
    _gemma_parallel "${s}"
  done
}

_acc_one() {
  local subject="${1:?subject model}"
  _init_conda
  conda activate qwen3_infer 2>/dev/null || conda activate gemma_scope
  PY=python
  if ! "${PY}" -c "import openai" 2>/dev/null; then
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "openai>=1.0.0" \
      || pip install "openai>=1.0.0"
  fi
  if [[ -f scripts/server/config/eval_config.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source <(sed 's/\r$//' scripts/server/config/eval_config.env)
    set +a
  fi
  export EVAL_DISABLE_WEB_SEARCH=1
  : "${EVAL_MODEL:=gpt-4o}"
  need=$("${PY}" -c "import json; print(len(json.load(open('${CASES_400}'))))")
  n=$(find "${ACC_OUT}/${subject}" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${n}" -ge "${need}" ]]; then
    echo "Acc ${subject}: already ${n}/${need}, skip"
    return 0
  fi
  echo "=== [acc ${subject}] judge=${EVAL_MODEL} workers=${ACC_WORKERS} (${n}/${need}) ==="
  "${PY}" scripts/stage1/run_stage2_diagnosis_accuracy.py \
    --subject-model "${subject}" \
    --cases "${CASES_400}" \
    --outputs "${SUBJECTS}" \
    --out-dir "${ACC_OUT}" \
    --eval-model "${EVAL_MODEL}" \
    --workers "${ACC_WORKERS}"
}

_acc_all() {
  local subjects=() s
  IFS=',' read -ra subjects <<< "${STAGE2_ACC_SUBJECTS}"
  for s in "${subjects[@]}"; do
    s="${s// /}"
    [[ -z "${s}" ]] && continue
    _acc_one "${s}" &
  done
  wait
  echo "=== acc-all done ==="
}

_acc_gemma_one() {
  local subject="${1:?subject model}"
  _activate_gemma
  _gemma_env
  export CUDA_VISIBLE_DEVICES="${GEMMA_GPU}"
  need=$("${PY}" -c "import json; print(len(json.load(open('${CASES_400}'))))")
  n=$("${PY}" - <<PY
import json
from pathlib import Path
d = Path("${ACC_GEMMA_OUT}") / "${subject}"
need = ${need}
if not d.is_dir():
    print(0)
    raise SystemExit(0)
ok = 0
for f in d.glob("*.json"):
    try:
        j = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if j.get("acc_error"):
        continue
    if str(j.get("acc_judge", "")).startswith("gemma-local-"):
        ok += 1
print(ok)
PY
)
  if [[ "${n}" -ge "${need}" ]]; then
    echo "Acc-gemma ${subject}: already ${n}/${need} valid, skip"
    return 0
  fi
  echo "=== [acc-gemma ${subject}] GPU${GEMMA_GPU} (${n}/${need} valid) ==="
  "${PY}" scripts/stage1/run_stage2_diagnosis_accuracy_gemma.py \
    --subject-model "${subject}" \
    --cases "${CASES_400}" \
    --outputs "${SUBJECTS}" \
    --out-dir "${ACC_GEMMA_OUT}"
}

_acc_gemma_all() {
  local subjects=() s
  IFS=',' read -ra subjects <<< "${STAGE2_ACC_GEMMA_SUBJECTS}"
  for s in "${subjects[@]}"; do
    s="${s// /}"
    [[ -z "${s}" ]] && continue
    _acc_gemma_one "${s}"
  done
  echo "=== acc-gemma-all done ==="
}

_eval_remaining() {
  echo "=== Stage-2 remaining: Gemma(${STAGE2_GEMMA_SUBJECTS}) + Acc(${STAGE2_ACC_SUBJECTS}) ==="
  _acc_all
  _gemma_all
}

_gemma() {
  _activate_gemma
  _gemma_env
  export CUDA_VISIBLE_DEVICES="${GEMMA_GPU}"
  echo "=== [gemma] GPU${GEMMA_GPU} fp16/incremental (single process) ==="
  _parse_infer_gpus
  need=$("${PY}" -c "import json; print(len(json.load(open('${STAGE2_MANIFEST}'))['${TASK}']['case_ids']))")
  while true; do
    _merge_shards 2>/dev/null || true
    n=$(_infer_count 2>/dev/null || echo 0)
    echo "--- Gemma pass (merged infer ${n}/${need}) ---"
    "${PY}" scripts/stage1/run_stage1_reasoning_eval.py \
      --task "${TASK}" \
      --subject-model "${MODEL}" \
      --judge gemma-local \
      --gemma-size 9b \
      --group both \
      --cases "${CASES_400}" \
      --outputs "${SUBJECTS}" \
      --out-dir "${RE_OUT}"
    ok=$("${PY}" - <<PY
import json
from pathlib import Path
need = ${need}
for g in ("direct", "inference_augmented"):
    p = Path("${RE_OUT}") / f"${TASK}_gemma-9b-it_${MODEL}_{g}.json"
    if not p.is_file():
        print(0); raise SystemExit
    d = json.loads(p.read_text(encoding="utf-8"))
    ok = sum(1 for c in d.get("cases", {}).values() if c.get("status") == "ok")
    if ok < need:
        print(0); raise SystemExit
print(1)
PY
)
    if [[ "${ok}" == "1" ]]; then
      echo "Gemma reasoning_eval complete (${need}/${need})."
      break
    fi
    echo "Gemma waiting for more infer (90s)..."
    sleep 90
  done
}

_acc_api() {
  _init_conda
  conda activate qwen3_infer 2>/dev/null || conda activate gemma_scope
  PY=python
  if ! "${PY}" -c "import openai" 2>/dev/null; then
    echo "Installing openai for API accuracy judge..." >&2
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "openai>=1.0.0" \
      || pip install "openai>=1.0.0"
  fi
  if [[ -f scripts/server/config/eval_config.env ]]; then
    set -a
    # shellcheck source=/dev/null
    source <(sed 's/\r$//' scripts/server/config/eval_config.env)
    set +a
  fi
  export EVAL_DISABLE_WEB_SEARCH=1
  : "${EVAL_MODEL:=gpt-4o}"
  echo "=== [acc] judge=${EVAL_MODEL} workers=${ACC_WORKERS} model=${MODEL} ==="
  need=$("${PY}" -c "import json; print(len(json.load(open('${CASES_400}'))))")
  while true; do
    _merge_shards 2>/dev/null || true
    n_subj=$("${PY}" -c "
import json
from pathlib import Path
o=json.loads(Path('${SUBJECTS}').read_text())
print(sum(1 for c in o.values() if '${MODEL}' in c))
" 2>/dev/null || echo 0)
    if [[ "${n_subj}" -lt 1 ]]; then
      echo "No subjects yet (${n_subj}); wait 60s..."
      sleep 60
      continue
    fi
    echo "--- Accuracy pass (${n_subj}/${need} subjects) ---"
    "${PY}" scripts/stage1/run_stage2_diagnosis_accuracy.py \
      --subject-model "${MODEL}" \
      --cases "${CASES_400}" \
      --outputs "${SUBJECTS}" \
      --out-dir "${ACC_OUT}" \
      --eval-model "${EVAL_MODEL}" \
      --workers "${ACC_WORKERS}" || true
    n=$(find "${ACC_OUT}/${MODEL}" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    echo "Accuracy files: ${n}/${need}"
    if [[ "${n}" -ge "${need}" ]]; then
      echo "Accuracy complete."
      break
    fi
    sleep 120
  done
}

_status() {
  _parse_infer_gpus 2>/dev/null || true
  _run_py - <<'PY'
import json
from pathlib import Path

def count_ok(p):
    if not p.is_file():
        return "missing"
    d = json.loads(p.read_text(encoding="utf-8"))
    if "cases" in d:
        ok = sum(1 for c in d["cases"].values() if c.get("status") == "ok")
        return f"{ok}/{len(d['cases'])}"
    return str(len(d))

models = ["qwen3-14b-thinking", "o3-mini", "deepseek-r1"]
inf_dir = Path("data/Stage2/inference")
model = "qwen3-14b-thinking"
merged = inf_dir / f"{model}_diagnosis.json"
if merged.is_file():
    print(f"  infer merged: {len(json.loads(merged.read_text()))}/400")

subj = Path("data/Stage2/oracle_diagnosis_subjects.json")
if subj.is_file():
    o = json.loads(subj.read_text())
    for m in models:
        n = sum(1 for c in o.values() if m in c)
        print(f"  subjects {m}: {n}/400")

re_dir = Path("data/Stage2/reasoning_eval")
for m in models:
    for g in ("direct", "inference_augmented"):
        p = re_dir / f"diagnosis_gemma-9b-it_{m}_{g}.json"
        print(f"  gemma {m} {g}: {count_ok(p)}")

acc_root = Path("data/Stage2/acc_results_gpt")
for m in models:
    d = acc_root / m
    if d.is_dir():
        n = len(list(d.glob("*.json")))
        print(f"  acc-gpt {m}: {n}/400")
    else:
        print(f"  acc-gpt {m}: missing")

acc_g = Path("data/Stage2/acc_results_gemma")
for m in models:
    d = acc_g / m
    if d.is_dir():
        n = len(list(d.glob("*.json")))
        print(f"  acc-gemma {m}: {n}/400")
    else:
        print(f"  acc-gemma {m}: missing")
PY
}

case "${1:-help}" in
  prep) _prep ;;
  merge) _merge_shards ;;
  infer-shard)
    _parse_infer_gpus
    _infer_shard "${2:?shard index}" "${3:?physical GPU}"
    ;;
  infer-all) _infer_all ;;
  gemma) _gemma ;;
  gemma-shard)
    _parse_gemma_gpus
    _gemma_shard "${2:?shard index}" "${3:?physical GPU}" "${4:-}"
    ;;
  gemma-parallel) _gemma_parallel "${2:-}" ;;
  gemma-all) _gemma_all ;;
  merge-gemma) _merge_gemma "${2:-}" ;;
  acc) _acc_api ;;
  acc-one) _acc_one "${2:?model}" ;;
  acc-all) _acc_all ;;
  acc-gemma-one) _acc_gemma_one "${2:?model}" ;;
  acc-gemma-all) _acc_gemma_all ;;
  eval-remaining) _eval_remaining ;;
  status) _status ;;
  help|*)
    cat <<EOF
Stage-2 A800 multi-GPU (default: 4 infer shards + gemma + acc)

  prep                    Build 400-case data
  infer-shard IDX GPU     One shard (e.g. infer-shard 0 0)
  infer-all               All shards sequential + merge
  merge                   Merge infer shard JSON → ${INFER_OUT}
  gemma                   Single-GPU Gemma (GPU ${GEMMA_GPU}, incremental)
  gemma-shard IDX GPU [MODEL]  One Gemma shard (default MODEL=qwen3-14b-thinking)
  gemma-parallel [MODEL]    N-GPU parallel + merge-gemma
  gemma-all                 o3-mini + deepseek-r1 (STAGE2_GEMMA_SUBJECTS)
  merge-gemma [MODEL]       Merge Gemma .shardN.json
  acc                       Acc for STAGE2_MODEL (default 14B)
  acc-one MODEL             Acc one model (GPT-4o API → ${ACC_OUT})
  acc-all                   Acc o3 + deepseek parallel (GPT-4o)
  acc-gemma-one MODEL       Acc one model (local Gemma-9B → ${ACC_GEMMA_OUT})
  acc-gemma-all             Acc all STAGE2_ACC_GEMMA_SUBJECTS (Gemma-9B, sequential)
  eval-remaining            acc-all then gemma-all
  status                    Progress (all 3 models)

Env:
  INFER_GPUS=0,1,2,7   GEMMA_GPUS=0,1,2   GEMMA_JUDGE_FULL_GPU=1
  SKIP_GPUS=6   GEMMA_GPU=3   QWEN_QUANT_MODE=fp16

Parallel Gemma (stop serial job first, preserves legacy progress):
  GEMMA_GPUS=0,1,2 bash scripts/stage1/run_stage2_a800_gemma_parallel.sh
EOF
    ;;
esac
