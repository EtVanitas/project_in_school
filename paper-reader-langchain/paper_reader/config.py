"""全局配置与错误分类：路径、模型与数据源常量；错误即数据的统一观测词汇。"""

import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

# 数据目录
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data"))).resolve()
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "app.db"
DOCS_DIR = DATA_DIR / "docs"
CACHE_DIR = DATA_DIR / "cache"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-v4-flash-vision-exp")

# 本地模型
LOCAL_MODEL_DIR = ROOT_DIR / "models"
LOCAL_LLM_NAME = os.getenv("LOCAL_LLM_NAME", "Qwen3-4B")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "multilingual-e5-small")
LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "1") == "1"
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "30"))

# 数据源
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

# 服务与限额
PORT = int(os.getenv("PORT", "8000"))              # 服务端口
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))   # 网络请求重试次数
CONTEXT_RENDER_LIMIT = 50000  # 检索块渲染字符上限（仅截断注入视图，与模型预算脱钩）

# 工具与笔记配置
MAX_STEPS = 6              # Agent 最大调用步数
MAX_TOOL_CALLS = 10        # 单轮问答工具总调用量上限
TOOL_TIMEOUT = 20          # 文本工具超时（秒）
VISION_TOOL_TIMEOUT = 90   # 多模态工具超时（秒）
TOTAL_TIMEOUT = 300        # 总响应超时（秒）
MAX_TOOL_OUTPUT = 4000     # 工具输出最大字符数
MEMORY_MAX_CHARS = 4000    # 笔记记忆注入上限
NOTE_BATCH_CHARS = 6000    # 整理保底单批喂入本地 4B 的对话字符预算
PROFILE_MAX_CHARS = 600    # 用户画像文件上限

ORG_TRIGGER_TOKENS = int(os.getenv("ORG_TRIGGER_TOKENS", "100000"))  # 自动整理触发线（last_input_tokens ≥ 此值）
ORG_AUTO_ENABLED = os.getenv("ORG_AUTO_ENABLED", "1") == "1"         # 自动触发总开关（手动整理不受限）
ORG_MAX_TOKENS = 2000      # 单次整理输出 token 上限（云端 max_tokens / 本地生成上限）
ORG_QA_MAX_CHARS = 3000    # 单次整理 Q/A 正文上限（超限截断标注）

# 本地 4B 在线/离线任务开关
PREJUDGE_ENABLED = os.getenv("PREJUDGE_ENABLED", "0") == "1"   # 意图路由+query改写开关
PREJUDGE_TIMEOUT = 2.0     # 预判超时
PROFILE_ENABLED = os.getenv("PROFILE_ENABLED", "1") == "1"     # 用户画像提炼与注入开关

# 模型加载锁
MODEL_LOAD_LOCK = threading.Lock()

# 错误分类与统一观测
KIND_PARAM = "param"          # 参数校验失败（缺参/类型/范围）→ 喂回自愈，不计失败计数；
KIND_TIMEOUT = "timeout"      # 工具执行超时（20s / 90s）→ 喂回 + 失败计数 +1；
KIND_TOOL = "tool"            # 工具执行抛错 → 喂回 + 失败计数 +1（同一工具连续 3 次弃用）；
KIND_LLM = "llm"              # 云端/本地模型调用失败 → 抛给调用方（图节点降级 / api 兜底），不进工具观测；
KIND_INTERNAL = "internal"    # 其他未预期异常 → 按 tool 处理。


def observation(kind: str, tool: str, reason: str) -> str:
    """统一错误观测文本（喂回模型的结构化数据，非异常）。"""
    return f"工具错误 {kind}（{tool}）：{reason}"


def counts_failure(kind: str) -> bool:
    """是否计入「同一工具连续失败」计数（param 属参数问题，不计入弃用逻辑）。"""
    return kind != KIND_PARAM


def ensure_dirs() -> None:
    """确保数据目录存在。"""
    for d in (DB_DIR, DOCS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """配置统一日志输出。"""
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")


def silence_hf() -> None:
    """静默 transformers 进度条。"""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:
        pass
