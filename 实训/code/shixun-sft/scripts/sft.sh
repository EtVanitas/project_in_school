#!/usr/bin/env bash
# ==============================================================================
# LlamaFactory SFT 训练脚本（带日志、自动重试、目录预创建）
# ==============================================================================
set -uo pipefail

# ── 基础路径配置 ────────────────────────────────────────────────────────────────
ROOT_DIR='/root/siton-data-1f55405a64d24fe2819a81c90df30517/20260617'
PROJECT_DIR="${ROOT_DIR}/shixun-sft"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$ROOT_DIR/LlamaFactory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-/root/siton-tmp/20235947/shixun-sft/configs/qwen15_gsm8k_full_sft.yaml}"
GPU_ID="${GPU_ID:-0}"

# ── 日志配置 ────────────────────────────────────────────────────────────────────
LOG_DIR="/root/siton-tmp"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ── 重试配置 ────────────────────────────────────────────────────────────────────
MAX_RETRIES="${MAX_RETRIES:-5}"       # 最大重试次数
RETRY_DELAY="${RETRY_DELAY:-30}"     # 每次重试前等待秒数

# ── 环境变量 ────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONPATH="${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ── 日志函数 ────────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

# 同时输出到终端和日志文件的 exec 重定向（保留颜色/进度条在终端）
exec > >(tee -a "${LOG_FILE}") 2>&1

# ── 预创建输出目录（防止 checkpoint 保存失败） ─────────────────────────────────
pre_create_output_dirs() {
    # 从 yaml 里提取 output_dir 字段（兼容有无空格的写法）
    local output_dir
    output_dir="$(grep -E '^\s*output_dir\s*:' "${CONFIG}" \
        | head -1 \
        | sed 's/.*output_dir\s*:\s*//' \
        | tr -d '"'"'"' \
        | xargs)"

    if [[ -z "${output_dir}" ]]; then
        log "⚠️  未能从配置文件中解析 output_dir，跳过预创建"
        return
    fi

    log "📁 预创建输出目录: ${output_dir}"
    mkdir -p "${output_dir}"
}

# ── 主函数 ──────────────────────────────────────────────────────────────────────
main() {
    log "════════════════════════════════════════════════════"
    log "🚀 训练启动"
    log "   CONFIG  : ${CONFIG}"
    log "   LOG     : ${LOG_FILE}"
    log "   GPU     : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    log "   重试上限: ${MAX_RETRIES} 次，间隔 ${RETRY_DELAY}s"
    log "════════════════════════════════════════════════════"

    # 预创建目录
    pre_create_output_dirs

    cd "${LLAMA_FACTORY_DIR}"

    # GPU / CUDA 检查
    log "🔍 CUDA 环境检查"
    "${PYTHON_BIN}" -c "
import torch
print('CUDA_VISIBLE_DEVICES =', '${CUDA_VISIBLE_DEVICES}')
print('torch.cuda.is_available() =', torch.cuda.is_available())
print('visible_device_count =', torch.cuda.device_count())
print('device0 =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"

    # ── 重试循环 ──────────────────────────────────────────────────────────────
    local attempt=1
    while true; do
        log "▶️  第 ${attempt}/${MAX_RETRIES} 次训练尝试"

        if "${PYTHON_BIN}" -m llamafactory.cli train "${CONFIG}"; then
            log "✅ 训练成功完成（第 ${attempt} 次尝试）"
            return 0
        fi

        local exit_code=$?
        log "❌ 训练退出，exit_code=${exit_code}（第 ${attempt}/${MAX_RETRIES} 次）"

        if (( attempt >= MAX_RETRIES )); then
            log "🛑 已达最大重试次数 ${MAX_RETRIES}，终止"
            return 1
        fi

        attempt=$(( attempt + 1 ))
        log "⏳ ${RETRY_DELAY}s 后重试（第 ${attempt}/${MAX_RETRIES} 次）..."
        sleep "${RETRY_DELAY}"

        # 重试前再次确保目录存在（防止首次保存失败丢失目录）
        pre_create_output_dirs
    done
}

main "$@"