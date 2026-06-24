#!/usr/bin/env bash
# A800 Stage-2: 4 infer shards + gemma + acc in tmux.
#
#   bash scripts/stage1/setup_a800_env.sh          # first time
#   bash scripts/stage1/run_stage2_a800_tmux.sh
#   tmux attach -t stage2-a800
#
# Customize GPUs (avoid busy GPU 6):
#   INFER_GPUS=0,1,2,7 GEMMA_GPU=3 SKIP_GPUS=6 bash scripts/stage1/run_stage2_a800_tmux.sh
#
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"
SESSION="${STAGE2_SESSION:-stage2-a800}"

sed -i 's/\r$//' \
  scripts/stage1/setup_a800_env.sh \
  scripts/stage1/run_stage2_a800_parallel.sh \
  scripts/stage1/run_stage2_a800_tmux.sh \
  scripts/stage1/run_stage2_parallel.sh \
  scripts/server/config/eval_config.env \
  2>/dev/null || true

INFER_GPUS="${INFER_GPUS:-0,1,2,7}"
GEMMA_GPU="${GEMMA_GPU:-3}"
SKIP_GPUS="${SKIP_GPUS:-6}"
QWEN_QUANT_MODE="${QWEN_QUANT_MODE:-fp16}"

export INFER_GPUS GEMMA_GPU SKIP_GPUS QWEN_QUANT_MODE

echo "==> A800 Stage-2 preflight"
bash scripts/stage1/run_stage2_a800_parallel.sh prep

# Parse GPU list for shard count
IFS=',' read -ra _gpus <<< "${INFER_GPUS}"
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
  echo "ERROR: no infer GPUs (INFER_GPUS=${INFER_GPUS} SKIP_GPUS=${SKIP_GPUS})" >&2
  exit 1
fi

echo "Layout: ${NUM_SHARDS} infer shards on GPUs ${GPU_LIST[*]}, gemma on GPU ${GEMMA_GPU}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  if [[ "${STAGE2_FORCE:-0}" == "1" ]]; then
    tmux kill-session -t "${SESSION}"
  else
    echo "Session ${SESSION} exists. attach: tmux attach -t ${SESSION}"
    echo "Restart: STAGE2_FORCE=1 bash scripts/stage1/run_stage2_a800_tmux.sh"
    exit 0
  fi
fi

_run="cd '${ROOT}' && bash scripts/stage1/run_stage2_a800_parallel.sh"

tmux new-session -d -s "${SESSION}" -n run \
  "${_run} infer-shard 0 ${GPU_LIST[0]}; echo '=== SHARD 0 DONE ==='; exec bash"

for ((i=1; i<NUM_SHARDS; i++)); do
  tmux split-window -t "${SESSION}:run" \
    "${_run} infer-shard ${i} ${GPU_LIST[$i]}; echo '=== SHARD ${i} DONE ==='; exec bash"
done

tmux split-window -t "${SESSION}:run" \
  "sleep 30; ${_run} gemma; echo '=== GEMMA DONE ==='; exec bash"

tmux split-window -t "${SESSION}:run" \
  "sleep 45; ${_run} acc; echo '=== ACC DONE ==='; exec bash"

tmux select-layout -t "${SESSION}:run" tiled

for ((i=0; i<NUM_SHARDS; i++)); do
  tmux select-pane -t "${SESSION}:run.${i}" -T "infer-${i}-GPU${GPU_LIST[$i]}"
done
tmux select-pane -t "${SESSION}:run.${NUM_SHARDS}" -T "gemma-GPU${GEMMA_GPU}"
tmux select-pane -t "${SESSION}:run.$((NUM_SHARDS + 1))" -T "acc-API"

echo ""
echo "Started tmux: ${SESSION}  (${NUM_SHARDS} infer + gemma + acc)"
echo "  attach:  tmux attach -t ${SESSION}"
echo "  status:  bash scripts/stage1/run_stage2_a800_parallel.sh status"
echo "  merge:   bash scripts/stage1/run_stage2_a800_parallel.sh merge"
echo ""
echo "GPU map:"
echo "  infer  → ${GPU_LIST[*]}  (${QWEN_QUANT_MODE} Qwen3-14B-thinking, ~100 cases each)"
echo "  gemma  → GPU${GEMMA_GPU}  Gemma-9B judge (incremental)"
echo "  acc    → API  (${ACC_WORKERS:-8} workers)"
