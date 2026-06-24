#!/usr/bin/env bash
# Download binary wheels on a PC (Windows WSL / Linux / Mac), then scp to P100 server.
#   bash scripts/stage1/download_awq_wheels.sh
#   scp -r scripts/stage1/wheels user@GPU-P100-2:~/MedRBench/scripts/stage1/
#
set -eu
cd "$(dirname "$0")/../.."
OUT="scripts/stage1/wheels"
mkdir -p "${OUT}"

PYVER="${PYVER:-cp310}"
PLATFORM="${PLATFORM:-manylinux2014_x86_64}"

echo "==> Downloading wheels to ${OUT}/"
pip download -d "${OUT}" \
  --only-binary :all: \
  --python-version 3.10 \
  --platform "${PLATFORM}" \
  --implementation cp \
  --abi "${PYVER}" \
  "pandas==2.0.3" \
  "pyarrow==17.0.0" \
  "numpy==1.26.4" \
  "multiprocess" "xxhash" "fsspec" "aiohttp" "dill" \
  "pyyaml" "requests" "tqdm" "packaging" "filelock" \
  "attrs" "frozenlist" "multidict" "yarl" "aiosignal" "async-timeout" \
  "charset-normalizer" "idna" "urllib3" "certifi" \
  2>/dev/null || pip download -d "${OUT}" --only-binary :all: \
  "pandas==2.0.3" "pyarrow==17.0.0" "numpy==1.26.4" \
  multiprocess xxhash fsspec aiohttp dill pyyaml requests tqdm packaging

pip download -d "${OUT}" --no-deps "datasets==2.19.0"
pip download -d "${OUT}" --no-deps "autoawq==0.2.9" 2>/dev/null || true

echo ""
echo "Wheels saved:"
ls -lh "${OUT}"/*.whl
echo ""
echo "Next: scp -r ${OUT} user@GPU-P100-2:~/MedRBench/scripts/stage1/"
echo "      ssh server 'bash scripts/stage1/install_awq_from_wheels.sh'"
