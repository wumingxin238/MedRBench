#!/usr/bin/env bash
# Install AWQ deps from local wheel files (no conda, no source compile).
#
# On a machine with good network (your PC), download wheels:
#   bash scripts/stage1/download_awq_wheels.sh
# Then scp folder to server:
#   scp -r scripts/stage1/wheels zhangzhuoyu@GPU-P100-2:~/MedRBench/scripts/stage1/
# On server:
#   bash scripts/stage1/install_awq_from_wheels.sh
#
set -eu
cd "$(dirname "$0")/../.."
WHEEL_DIR="scripts/stage1/wheels"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen3_infer

if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
fi

if [[ ! -d "${WHEEL_DIR}" ]] || [[ -z "$(ls -A "${WHEEL_DIR}"/*.whl 2>/dev/null)" ]]; then
  echo "ERROR: No wheels in ${WHEEL_DIR}/" >&2
  echo "  On PC:  bash scripts/stage1/download_awq_wheels.sh" >&2
  echo "  Then:   scp -r scripts/stage1/wheels user@server:~/MedRBench/scripts/stage1/" >&2
  exit 1
fi

echo "==> Installing from ${WHEEL_DIR}/*.whl"
pip install "${WHEEL_DIR}"/*.whl

if ! python -c "import awq" 2>/dev/null; then
  echo "==> autoawq missing; install from wheel without deps"
  awq_whl=$(ls "${WHEEL_DIR}"/autoawq-*.whl 2>/dev/null | head -1 || true)
  if [[ -n "${awq_whl}" ]]; then
    pip install --no-deps "${awq_whl}"
  else
    pip install --no-deps autoawq==0.2.9
  fi
fi

echo "==> Verify"
python -W ignore::DeprecationWarning -c "import awq; print('awq OK')"
python -c "import bitsandbytes" 2>/dev/null && echo "bitsandbytes OK" || true
echo "Done."
