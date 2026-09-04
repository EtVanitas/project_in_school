"""全局配置：常量、路径、目录初始化与日志（模块级读取环境变量）。"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent  # 项目根目录
# .env 显式配置优先于同名系统环境变量；无 .env 时自动使用系统环境变量
load_dotenv(ROOT_DIR / ".env", override=True)

CATEGORIES = ["机器学习", "强化学习", "图像生成", "大语言模型", "多模态", "其它"]
PAPER_ROOT = Path(os.getenv("PAPER_ROOT", str(ROOT_DIR / "papers"))).resolve()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
AUTO_OPEN_PDF = os.getenv("AUTO_OPEN_PDF", "true").strip().lower() == "true"  # 下载后自动打开 PDF


def _paper_dirs(kind: str) -> list[Path]:
    """某状态（unread/read）下的全部目录：根目录 + 各分类子目录。"""
    base = PAPER_ROOT / kind
    return [base] + [base / c for c in CATEGORIES]


def all_paper_dirs() -> list[Path]:
    """全库检索目录（translated 为旧系统遗留目录，存在时才参与检索）。"""
    dirs = _paper_dirs("unread") + _paper_dirs("read")
    translated = PAPER_ROOT / "translated"
    return dirs + ([translated] if translated.exists() else [])


def ensure_paper_library() -> None:
    """确保论文库目录结构存在：papers/{unread,read}/{6 分类}。"""
    for kind in ("unread", "read"):
        for d in _paper_dirs(kind):
            d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """配置统一日志输出。"""
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
