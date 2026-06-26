#!/usr/bin/env bash
# Stage-2 剩余评估：o3-mini + deepseek-r1 × 400
#   - GPT-4o Acc（API 并行，不占 GPU）
#   - Gemma-9B reasoning（需空闲 GPU）
#
#   bash scripts/stage1/run_stage2_a800_eval_remaining.sh
#   tmux attach -t stage2-rest
#
# 当前 GPU 若全满：先只跑 acc（ACC_ONLY=1），Gemma 稍后再开 GEMMA_GPUS=...
#
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"
SESSION="${STAGE2_REST_SESSION:-stage2-rest}"

sed -i 's/\r$//' \
  scripts/stage1/run_stage2_a800_parallel.sh \
  scripts/stage1/run_stage2_a800_eval_remaining.sh \
  2>/dev/null || true

GEMMA_GPUS="${GEMMA_GPUS:-0,1,2}"
SKIP_GPUS="${SKIP_GPUS:-6}"
GEMMA_JUDGE_FULL_GPU="${GEMMA_JUDGE_FULL_GPU:-1}"
ACC_WORKERS="${ACC_WORKERS:-16}"
export GEMMA_GPUS SKIP_GPUS GEMMA_JUDGE_FULL_GPU ACC_WORKERS

_run="cd '${ROOT}' && bash scripts/stage1/run_stage2_a800_parallel.sh"

echo "==> Stage-2 remaining eval"
bash scripts/stage1/run_stage2_a800_parallel.sh status || true
echo ""
echo "Pending: Gemma-9B + Acc for: ${STAGE2_GEMMA_SUBJECTS:-o3-mini,deepseek-r1}"
echo "14B qwen3-14b-thinking should already be 400/400 — skipped if complete."
echo ""

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  if [[ "${REST_FORCE:-0}" == "1" ]]; then
    tmux kill-session -t "${SESSION}"
  else
    echo "Session ${SESSION} exists. attach: tmux attach -t ${SESSION}"
    echo "Restart: REST_FORCE=1 bash scripts/stage1/run_stage2_a800_eval_remaining.sh"
    exit 0
  fi
fi

if [[ "${ACC_ONLY:-0}" == "1" ]]; then
  tmux new-session -d -s "${SESSION}" -n rest \
    "${_run} acc-all; echo '=== ACC ALL DONE ==='; exec bash"
  echo "Started ${SESSION} (acc-all only). attach: tmux attach -t ${SESSION}"
  exit 0
fi

# Pane 0: o3 acc | Pane 1: deepseek acc | Pane 2: gemma-all (sequential o3 then deepseek, 3 GPU each)
tmux new-session -d -s "${SESSION}" -n rest \
  "${_run} acc-one o3-mini; echo '=== ACC o3 DONE ==='; exec bash"

tmux split-window -t "${SESSION}:rest" -h \
  "${_run} acc-one deepseek-r1; echo '=== ACC deepseek DONE ==='; exec bash"

tmux split-window -t "${SESSION}:rest" -v \
  "${_run} gemma-all; echo '=== GEMMA ALL DONE ==='; exec bash"

tmux split-window -t "${SESSION}:rest" \
  "while true; do sleep 180; cd '${ROOT}' && bash scripts/stage1/run_stage2_a800_parallel.sh status; done"

tmux select-layout -t "${SESSION}:rest" tiled
tmux select-pane -t "${SESSION}:rest.0" -T "acc-o3"
tmux select-pane -t "${SESSION}:rest.1" -T "acc-deepseek"
tmux select-pane -t "${SESSION}:rest.2" -T "gemma-all"
tmux select-pane -t "${SESSION}:rest.3" -T "status"

echo ""
echo "Started tmux: ${SESSION}"
echo "  attach:  tmux attach -t ${SESSION}"
echo "  status:  bash scripts/stage1/run_stage2_a800_parallel.sh status"
echo ""
echo "If Gemma fails (GPU OOM): ACC_ONLY=1 bash scripts/stage1/run_stage2_a800_eval_remaining.sh"
echo "Then when GPUs free: GEMMA_GPUS=0,1,2 bash scripts/stage1/run_stage2_a800_parallel.sh gemma-all"
