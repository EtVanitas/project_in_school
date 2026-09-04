"""AI 层：DeepSeek LLM 懒加载工厂 + 分析任务（分类 / 总结）。

统一走 LLM 纯文本调用 + 本地归一；LLM 缺失或调用失败时业务级回退，不向上层抛异常。
切换其它供应商只需修改 get_llm 一处。
"""

import logging
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

_llm_cache: dict = {}


def get_llm(temperature: float = 0.1, model: Optional[str] = None) -> Optional[object]:
    """懒加载 ChatOpenAI（DeepSeek 兼容接口）；未配置 Key/依赖缺失/初始化失败时返回 None。"""
    if not config.DEEPSEEK_API_KEY:
        return None
    cache_key = (temperature, model)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model or config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=temperature,
            timeout=120,
            max_retries=2,
        )
        _llm_cache[cache_key] = llm
        return llm
    except Exception as e:  # 依赖缺失等环境问题
        logger.error(f"初始化 LLM 失败: {e}")
        return None


def has_llm() -> bool:
    """是否配置了可用的 LLM。"""
    return get_llm() is not None


def _normalize_category(raw: str) -> str:
    """模型输出归一为合法分类名，无法识别回退「其它」。"""
    text = raw or ""
    return next((c for c in config.CATEGORIES if c in text), "其它")


def classify_text(text: str) -> str:
    """AI 判定论文领域（六大分类之一）；无 LLM 或失败回退「其它」。"""
    llm = get_llm()
    if llm is None:
        return "其它"
    try:
        resp = llm.invoke([
            ("system", f"你是学术论文分类专家。判断论文属于哪个领域，只输出以下名称之一：{'、'.join(config.CATEGORIES)}"),
            ("human", f"论文内容:\n{text[:2500]}"),
        ])
        return _normalize_category(str(resp.content))
    except Exception as e:
        logger.warning(f"分类调用失败，回退「其它」: {e}")
        return "其它"


def summarize_text(title: str, text: str) -> str:
    """生成中文结构化总结（约 400 字）；无 LLM 返回空串，失败返回说明，不抛出。"""
    llm = get_llm(temperature=0.3)
    if llm is None:
        return ""
    try:
        resp = llm.invoke([
            ("system", "你是资深学术研究员。基于论文内容用中文生成约 400 字结构化总结"
             "（Markdown 列表）：研究问题、方法思路、关键结果与贡献。"),
            ("human", f"论文标题: {title}\n\n论文内容（节选）:\n{text[:12000]}"),
        ])
        return str(resp.content).strip()
    except Exception as e:
        logger.warning(f"生成总结失败: {e}")
        return f"（生成总结失败: {e}）"
