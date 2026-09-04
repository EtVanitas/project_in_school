#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/siton-tmp/20235947/shixun-sft}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/plot_training_curves.py" \
  --state "${TRAINER_STATE:-${PROJECT_DIR}/outputs/qwen15_gsm8k_lora_sft/trainer_state.json}" \
  --output_dir "${FIGURE_DIR:-${PROJECT_DIR}/outputs/figures}"
