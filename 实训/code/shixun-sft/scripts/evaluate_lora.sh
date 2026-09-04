#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/siton-tmp/20235947/shixun-sft}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/evaluate_gsm8k_predictions.py" \
  --predictions "${PREDICTIONS:-${PROJECT_DIR}/outputs/qwen15_gsm8k_lora_predict/generated_predictions.jsonl}" \
  --output_dir "${EVAL_DIR:-${PROJECT_DIR}/outputs/eval_lora}"
