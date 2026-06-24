#!/usr/bin/env bash
# A800: 3-way parallel Gemma reasoning_eval (full GPU, no CPU offload).
#
# Stop any serial Gemma job first — shards import completed rows from legacy JSON.
#
#   bash scripts/stage1/run_stage2_a800_gemma_parallel.sh
#   tmux attach -t stage2-gemma
#
# Customize:
#   GEMMA_GPUS=0,1,2 SKIP_GPUS=6 bash scripts/stage1/run_stage2_a800_gemma_parallel.sh
#
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"
SESSION="${GEMMA_SESSION:-stage2-gemma}"

sed -i 's/\r$//' \
  scripts/stage1/run_stage2_a800_parallel.sh \
  scripts/stage1/run_stage2_a800_gemma_parallel.sh \
  2>/dev/null || true

GEMMA_GPUS="${GEMMA_GPUS:-0,1,2}"
SKIP_GPUS="${SKIP_GPUS:-6}"
GEMMA_JUDGE_FULL_GPU="${GEMMA_JUDGE_FULL_GPU:-1}"
export GEMMA_GPUS SKIP_GPUS GEMMA_JUDGE_FULL_GPU

IFS=',' read -ra _gpus <<< "${GEMMA_GPUS}"
GPU_LIST=()
for g in "${_gpus[@]}"; do
  g="${g// /}"
  [[ -z "${g}" ]] && continue
  if [[ ",${SKIP_GPUS}," == *",${g},"* ]]; then
    continue
  fi
  GPU_LIST+=("${g}")
done
NUM_SHARDS="${#GPU_LIST[@]}"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "ERROR: no Gemma GPUs (GEMMA_GPUS=${GEMMA_GPUS} SKIP_GPUS=${SKIP_GPUS})" >&2
  exit 1
fi

echo "Gemma parallel: ${NUM_SHARDS} shards on GPUs ${GPU_LIST[*]} (GEMMA_JUDGE_FULL_GPU=${GEMMA_JUDGE_FULL_GPU})"
bash scripts/stage1/run_stage2_a800_parallel.sh status || true

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  if [[ "${GEMMA_FORCE:-0}" == "1" ]]; then
    tmux kill-session -t "${SESSION}"
  else
    echo "Session ${SESSION} exists. attach: tmux attach -t ${SESSION}"
    echo "Restart: GEMMA_FORCE=1 bash scripts/stage1/run_stage2_a800_gemma_parallel.sh"
    exit 0
  fi
fi

_run="cd '${ROOT}' && bash scripts/stage1/run_stage2_a800_parallel.sh"

tmux new-session -d -s "${SESSION}" -n gemma \
  "${_run} gemma-shard 0 ${GPU_LIST[0]}; echo '=== GEMMA SHARD 0 DONE ==='; exec bash"

for ((i=1; i<NUM_SHARDS; i++)); do
  tmux split-window -t "${SESSION}:gemma" \
    "${_run} gemma-shard ${i} ${GPU_LIST[$i]}; echo '=== GEMMA SHARD ${i} DONE ==='; exec bash"
done

tmux split-window -t "${SESSION}:gemma" \
  "while true; do sleep 120; cd '${ROOT}' && bash scripts/stage1/run_stage2_a800_parallel.sh status; done"

tmux select-layout -t "${SESSION}:gemma" tiled

for ((i=0; i<NUM_SHARDS; i++)); do
  tmux select-pane -t "${SESSION}:gemma.${i}" -T "gemma-${i}-GPU${GPU_LIST[$i]}"
done
tmux select-pane -t "${SESSION}:gemma.${NUM_SHARDS}" -T "status"

echo ""
echo "Started tmux: ${SESSION}  (${NUM_SHARDS} Gemma workers)"
echo "  attach:  tmux attach -t ${SESSION}"
echo "  status:  bash scripts/stage1/run_stage2_a800_parallel.sh status"
echo "  merge:   bash scripts/stage1/run_stage2_a800_parallel.sh merge-gemma"
echo ""
echo "When all shards finish:"
echo "  bash scripts/stage1/run_stage2_a800_parallel.sh merge-gemma"
