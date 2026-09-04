#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR='/root/siton-tmp/20235947'

PROJECT_DIR="${ROOT_DIR}/shixun-sft"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/prepare_gsm8k_data.py" \
  --gsm8k_dir "${GSM8K_DIR:-/root/siton-data-1f55405a64d24fe2819a81c90df30517/20260617/gsm8k}" \
  --output_dir "${PROJECT_DIR}/data" \
  --train_limit "${TRAIN_LIMIT:-0}" \
  --test_limit "${TEST_LIMIT:-0}"
