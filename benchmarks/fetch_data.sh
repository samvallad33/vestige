#!/usr/bin/env bash
# Fetch official LongMemEval_S and LoCoMo-10 datasets with SHA256 integrity verification.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${DATA_DIR}"

LME_EXPECTED="d6f21ea9c60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
LME_EXPECTED_ALT="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
LOCOMO_EXPECTED="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

compute_sha256() {
  local file="$1"
  local res=""
  if command -v sha256sum >/dev/null 2>&1; then
    res="$(sha256sum "$file" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    res="$(shasum -a 256 "$file" | awk '{print $1}')"
  elif command -v python3 >/dev/null 2>&1; then
    res="$(python3 -c "import hashlib; print(hashlib.sha256(open('$file', 'rb').read()).hexdigest())")"
  elif command -v python >/dev/null 2>&1; then
    res="$(python -c "import hashlib; print(hashlib.sha256(open('$file', 'rb').read()).hexdigest())")"
  else
    echo "error: no sha256 tool found" >&2
    exit 1
  fi
  echo "$res" | tr -d '\r\n '
}

CURL_CMD="curl"
if command -v curl.exe >/dev/null 2>&1; then
  CURL_CMD="curl.exe"
fi

echo "=== Fetching Benchmark Datasets ==="

LME_FILE="${DATA_DIR}/longmemeval_s_cleaned.json"
LME_ACTUAL=""
if [ -f "${LME_FILE}" ]; then
  LME_ACTUAL="$(compute_sha256 "${LME_FILE}")"
fi

if [ "${LME_ACTUAL}" != "${LME_EXPECTED}" ]; then
  echo "Downloading LongMemEval_S..."
  ${CURL_CMD} -fSL "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
    -o "${LME_FILE}"
  LME_ACTUAL="$(compute_sha256 "${LME_FILE}")"
fi

if [ "${LME_ACTUAL}" != "${LME_EXPECTED}" ]; then
  echo "FAIL: LongMemEval_S sha256 mismatch!" >&2
  echo "Expected: ${LME_EXPECTED}" >&2
  echo "Actual:   ${LME_ACTUAL}" >&2
  exit 1
fi
echo "LongMemEval_S sha256 verified: ${LME_ACTUAL} (OK)"

LOCOMO_FILE="${DATA_DIR}/locomo10.json"
LOCOMO_ACTUAL=""
if [ -f "${LOCOMO_FILE}" ]; then
  LOCOMO_ACTUAL="$(compute_sha256 "${LOCOMO_FILE}")"
fi

if [ "${LOCOMO_ACTUAL}" != "${LOCOMO_EXPECTED}" ]; then
  echo "Downloading LoCoMo-10..."
  ${CURL_CMD} -fSL "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json" \
    -o "${LOCOMO_FILE}"
  LOCOMO_ACTUAL="$(compute_sha256 "${LOCOMO_FILE}")"
fi

if [ "${LOCOMO_ACTUAL}" != "${LOCOMO_EXPECTED}" ]; then
  echo "FAIL: LoCoMo-10 sha256 mismatch!" >&2
  echo "Expected: ${LOCOMO_EXPECTED}" >&2
  echo "Actual:   ${LOCOMO_ACTUAL}" >&2
  exit 1
fi
echo "LoCoMo-10 sha256 verified: ${LOCOMO_ACTUAL} (OK)"
echo "All datasets verified successfully."
