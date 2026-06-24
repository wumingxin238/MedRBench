#!/usr/bin/env bash
# Stage-2: tmux 3-pane parallel (infer | gemma | acc).
#
#   bash scripts/stage1/run_stage2_tmux.sh
#   tmux attach -t stage2
#   bash scripts/stage1/run_stage2_parallel.sh status
#
# Restart (kill old session):
#   STAGE2_FORCE=1 bash scripts/stage1/run_stage2_tmux.sh
#
set -eu
cd "$(dirname "$0")/../.."
ROOT="$PWD"
SESSION="${STAGE2_SESSION:-stage2}"

sed -i 's/\r$//' scripts/stage1/run_stage2_parallel.sh scripts/stage1/run_stage2_tmux.sh 2>/dev/null || true

echo "==> Stage-2 preflight"
if ! bash scripts/stage1/check_stage2_deps.sh; then
  if [[ "${STAGE2_AUTO_FIX:-1}" == "1" ]]; then
    echo "==> Auto-fix qwen3_infer (autoawq)..."
    bash scripts/stage1/fix_qwen3_infer_deps.sh
    bash scripts/stage1/check_stage2_deps.sh || {
      echo "Preflight still failing — fix errors above before tmux start." >&2
      exit 1
    }
  else
    exit 1
  fi
fi

bash scripts/stage1/run_stage2_parallel.sh prep

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  if [[ "${STAGE2_FORCE:-0}" == "1" ]]; then
    echo "Killing existing session ${SESSION}"
    tmux kill-session -t "${SESSION}"
  else
    echo "Session ${SESSION} already exists."
    echo "  attach:  tmux attach -t ${SESSION}"
    echo "  restart: STAGE2_FORCE=1 bash scripts/stage1/run_stage2_tmux.sh"
    exit 0
  fi
fi

# Non-interactive tmux panes don't load .bashrc — cd to repo root explicitly.
_run="cd '${ROOT}' && bash scripts/stage1/run_stage2_parallel.sh"

tmux new-session -d -s "${SESSION}" -n run \
  "${_run} infer-gpu1; echo '=== INFER DONE ==='; exec bash"

tmux split-window -h -t "${SESSION}:run" \
  "sleep 15; ${_run} gemma-gpu0; echo '=== GEMMA DONE ==='; exec bash"

tmux split-window -v -t "${SESSION}:run.0" \
  "sleep 25; ${_run} acc-api; echo '=== ACC DONE ==='; exec bash"

tmux select-pane -t "${SESSION}:run.0" -T "infer-GPU1"
tmux select-pane -t "${SESSION}:run.1" -T "gemma-GPU0"
tmux select-pane -t "${SESSION}:run.2" -T "acc-API"

tmux select-layout -t "${SESSION}:run" tiled

echo ""
echo "Started tmux session: ${SESSION}"
echo "  attach:  tmux attach -t ${SESSION}"
echo "  status:  bash scripts/stage1/run_stage2_parallel.sh status"
echo ""
echo "Parallel layout:"
echo "  infer  → GPU1  qwen3_infer   Qwen3-14B-thinking AWQ  (400 cases)"
echo "  gemma  → GPU0  gemma_scope   Gemma-9B judge           (incremental)"
echo "  acc    → API   eval_config   accuracy judge           (8 workers)"
