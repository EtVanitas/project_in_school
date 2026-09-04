#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/siton-data-1f55405a64d24fe2819a81c90df30517/20260617/shixun-dpo}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-/root/siton-data-1f55405a64d24fe2819a81c90df30517/20260617/LlamaFactory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-/root/siton-tmp/20235947/shixun-dpo/configs/qwen15_math_step_full_dpo_0.01.yaml}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONPATH="${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${LLAMA_FACTORY_DIR}"
"${PYTHON_BIN}" -c "import torch; print('CUDA_VISIBLE_DEVICES=', '${CUDA_VISIBLE_DEVICES}'); print('torch.cuda.is_available()=', torch.cuda.is_available()); print('visible_device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
"${PYTHON_BIN}" -m llamafactory.cli train "${CONFIG}"
