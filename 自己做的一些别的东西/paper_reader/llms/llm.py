"""模型层（云端 + 本地业务封装）：DeepSeek 问答/整理/多模态/引用校验。

云端：流式问答、统一整理（复用问答组装序列吃前缀缓存）、多模态图表解读、引用校验护栏；
未配置 Key 时返回友好提示。本地业务：保底整理、画像提炼、前置预判（意图路由 + query 改写），
模型基础设施（懒加载/串行锁/超时）在 llms/local_llm.py。
"""

import base64
import json
import logging
import re
import time
from typing import AsyncIterator

from langchain_core.messages import HumanMessage

from .. import config, db
from ..docs import documents
from . import prompts
from .context import build_model_context
from .local_llm import LocalLLMError, local_run

logger = logging.getLogger(__name__)

_llm_cache: dict = {}


def get_llm(temperature: float = 0.3):
    """懒加载 ChatOpenAI（DeepSeek 兼容接口）；未配置 Key/初始化失败返回 None。"""
    if not config.DEEPSEEK_API_KEY:
        return None
    if temperature in _llm_cache:
        return _llm_cache[temperature]
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=temperature,
            timeout=120,
            max_retries=config.MAX_RETRIES,
            stream_usage=True,  # 流式响应携带 usage（token 用量埋点）
        )
        _llm_cache[temperature] = llm
        return llm
    except Exception as e:
        logger.error("初始化 DeepSeek 客户端失败: %s", e)
        return None


def has_llm() -> bool:
    """是否配置了可用的 LLM。"""
    return get_llm() is not None


async def stream_answer(doc_title: str, context: str, history: list[dict],
                        question: str, selected_text: str = "",
                        conversation_id: int | None = None) -> AsyncIterator[str]:
    """流式生成问答回复（逐段返回文本）；带 API 用量埋点。"""
    llm = get_llm()
    if llm is None:
        yield "未配置 DEEPSEEK_API_KEY，无法调用模型。请在项目根目录 .env 中配置后重试。"
        return
    started = time.monotonic()
    usage: dict = {}
    msgs = build_model_context(kind="answer", doc_title=doc_title, history=history,
                               context_text=context, question=question,
                               selected_text=selected_text)
    try:
        async for chunk in llm.astream(msgs):
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
            text = chunk.content
            if isinstance(text, str) and text:
                yield text
        tokens_in = int(usage.get("input_tokens") or 0)
        db.note_input_tokens(conversation_id, tokens_in)  # 记账：当前上下文 token 规模（自动整理触发判定）
        db.record_event("api_llm", event_name="stream_answer", conversation_id=conversation_id,
                        latency_ms=db.ms_since(started),
                        tokens_in=tokens_in,
                        tokens_out=int(usage.get("output_tokens") or 0))
    except Exception as e:
        logger.exception("流式回答失败")
        db.record_event("api_llm", event_name="stream_answer", conversation_id=conversation_id,
                        ok=False, latency_ms=db.ms_since(started), detail=str(e)[:200])
        yield f"\n\n（调用模型失败：{e}，请稍后重试）"


# ===== 引用校验护栏 =====

_PAGE_CITE_RE = re.compile(r"第\s*(\d+)(?:\s*[-–~—至]\s*(\d+))?\s*页")


def verify_citations(answer: str, used_pages: set[int]) -> list[str]:
    """校验回答中的页码引用是否都来自提供的上下文，返回越界引用描述列表（护栏）。

    单点引用（第 N 页）严格校验；区间引用（第 M-N 页）宽容处理——
    区间内 ≥50% 页码在提供范围内视为概括性表述，否则整体记为越界。
    """
    if not used_pages:
        return []
    bad: list[str] = []
    for m in _PAGE_CITE_RE.finditer(answer):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if end < start:
            start, end = end, start
        end = min(end, start + 50)  # 防御异常大范围引用
        if start == end:
            if start not in used_pages:
                bad.append(str(start))
        else:
            hit = sum(1 for p in range(start, end + 1) if p in used_pages)
            if hit / (end - start + 1) < 0.5:
                bad.append(f"{start}-{end}")
    if bad:
        logger.warning("引用校验发现越界引用: %s（可用页码共 %d 页）", bad, len(used_pages))
    return bad


# ===== 整理统一（云端主路径复用组装序列吃缓存；本地 4B 保底） =====


async def organize_cloud(doc_title: str, history: list[dict],
                         profile_text: str = "", conversation_id: int | None = None) -> str:
    """云端整理（主路径）：复用问答组装序列（吃前缀缓存），仅末尾追加一条整理指令。

    history 为模型层增量条目（与上一轮问答的历史组装逐字节同源）；返回整理结果正文
    （Q/A 文本）；未配置 Key / 调用失败抛 RuntimeError（由调用方转本地保底）。

    注意：不得注入 memory_text（现有笔记）——实测证明模型会把已是问答格式的记忆区
    直接当作答案复读（输出恒为旧 Q/A），忽略 history 增量；增量天然不含已整理内容，
    无需旧笔记参照。
    """
    client = get_llm()
    if client is None:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法云端整理")
    instruction = prompts.ORGANIZE_SYS.format(limit=config.ORG_QA_MAX_CHARS)
    msgs = build_model_context(kind="agent", doc_title=doc_title, history=history,
                               context_text="",
                               profile_text=profile_text, question=instruction)
    started = time.monotonic()
    usage: dict = {}
    parts: list[str] = []
    try:
        async for chunk in client.bind(max_tokens=config.ORG_MAX_TOKENS).astream(msgs):
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
            text = chunk.content
            if isinstance(text, str) and text:
                parts.append(text)
    except Exception as e:
        db.record_event("api_llm", event_name="organize", conversation_id=conversation_id,
                        ok=False, latency_ms=db.ms_since(started), detail=str(e)[:200])
        raise RuntimeError(f"云端整理失败：{e}") from e
    db.record_event("api_llm", event_name="organize", conversation_id=conversation_id,
                    latency_ms=db.ms_since(started),
                    tokens_in=int(usage.get("input_tokens") or 0),
                    tokens_out=int(usage.get("output_tokens") or 0))
    return "".join(parts)


async def organize_local_batch(segment: str, conversation_id: int | None = None) -> str:
    """本地 4B 保底：单批对话片段（渲染文本）→ Q/A 整理文本；失败抛 RuntimeError。"""
    if not config.LOCAL_LLM_ENABLED:
        raise RuntimeError("本地模型已关闭，无法整理对话")
    try:
        return await local_run(
            [{"role": "system",
              "content": prompts.ORGANIZE_LOCAL_SYS.format(limit=config.ORG_QA_MAX_CHARS)},
             {"role": "user", "content": segment}],
            max_new_tokens=config.ORG_MAX_TOKENS, timeout=120, tag="organize",
            conversation_id=conversation_id)
    except LocalLLMError as e:
        raise RuntimeError(f"整理对话失败：{e}") from e


async def update_profile_cloud(profile_text: str, history: list[dict], doc_title: str = "",
                               conversation_id: int | None = None) -> str | None:
    """云端画像更新（整理成功后的第二次调用，同前缀吃缓存）；失败返回 None 保留旧画像。"""
    if not config.PROFILE_ENABLED:
        return None
    client = get_llm()
    if client is None:
        return None
    instruction = prompts.PROFILE_UPDATE_INSTRUCTION.format(limit=config.PROFILE_MAX_CHARS)
    msgs = build_model_context(kind="agent", doc_title=doc_title, history=history,
                               context_text="", profile_text=profile_text, question=instruction)
    started = time.monotonic()
    usage: dict = {}
    parts: list[str] = []
    try:
        async for chunk in client.bind(max_tokens=1000).astream(msgs):
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
            text = chunk.content
            if isinstance(text, str) and text:
                parts.append(text)
    except Exception as e:
        db.record_event("api_llm", event_name="update_profile", conversation_id=conversation_id,
                        ok=False, latency_ms=db.ms_since(started), detail=str(e)[:200])
        logger.info("云端画像更新失败（保留旧画像）：%s", e)
        return None
    db.record_event("api_llm", event_name="update_profile", conversation_id=conversation_id,
                    latency_ms=db.ms_since(started),
                    tokens_in=int(usage.get("input_tokens") or 0),
                    tokens_out=int(usage.get("output_tokens") or 0))
    return "".join(parts) or None


# ===== 本地任务：对话文本渲染与画像（离线，本地 4B） =====


def _format_dialog(messages: list[dict], max_chars: int) -> str:
    """把模型层条目格式化为对话文本（单条截断；超出总上限保留最新部分）。"""
    lines: list[str] = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "助手"
        content = str(m.get("content") or "").strip()
        if content:
            lines.append(f"{role}：{content[:1500]}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        text = "（较早内容已省略）\n…\n" + text[-max_chars:]
    return text


async def update_profile(profile_text: str, messages: list[dict],
                         conversation_id: int | None = None) -> str | None:
    """旧画像 + 本次对话 → 新画像（三段小体量，预算 500 token）；失败返回 None 保留旧画像。

    仅在「整理」离线流程内顺带调用（云端不可用时的保底路径），不进入在线问答链。
    """
    if not config.PROFILE_ENABLED or not config.LOCAL_LLM_ENABLED:
        return None
    user_text = (f"【现有画像】\n{profile_text or '（暂无，这是第一次提炼）'}\n\n"
                 f"【本次对话】\n{_format_dialog(messages, 4000)}")
    try:
        return await local_run(
            [{"role": "system", "content": prompts.PROFILE_SYS.format(limit=config.PROFILE_MAX_CHARS)},
             {"role": "user", "content": user_text}],
            max_new_tokens=500, timeout=60, tag="update_profile",
            conversation_id=conversation_id)
    except Exception as e:
        logger.info("画像更新失败（保留旧版）：%s", e)
        return None


# ===== 前置预判（本地 4B：意图路由 + query 改写合并为一次调用） =====

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)  # 预判输出中的 JSON 段提取


def _prejudge_input(question: str, history: list[dict] | None) -> str:
    """拼装预判用户输入：最近 2 条用户消息作指代消解语境（各截 80 字）。"""
    recent = [str(m.get("content") or "")[:80] for m in (history or []) if m.get("role") == "user"][-2:]
    context_line = f"最近提问：{'；'.join(recent)}\n" if recent else ""
    return f"{context_line}当前问题：{question[:200]}"


async def prejudge(question: str, history: list[dict] | None = None) -> dict | None:
    """本地 4B 一次调用返回 {"need_retrieval": bool, "query": str}；关闭/失败/超时返回 None（调用方走默认链路）。"""
    if not config.LOCAL_LLM_ENABLED or not config.PREJUDGE_ENABLED:
        return None
    try:
        text = await local_run(
            [{"role": "system", "content": prompts.PREJUDGE_SYS},
             {"role": "user", "content": _prejudge_input(question, history)}],
            max_new_tokens=64, timeout=config.PREJUDGE_TIMEOUT, tag="prejudge")
    except Exception as e:  # LocalLLMError 等一切异常降级为不预判
        logger.info("预判不可用（降级行为）：%s", e)
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    need = data.get("need_retrieval")
    if not isinstance(need, bool):
        return None
    rewrite = str(data.get("query") or "").strip()
    return {"need_retrieval": need, "query": rewrite[:120] if need else ""}


# ===== 多模态图表解读（PDF 单页渲染 → DeepSeek vision 流式） =====


class VisionError(RuntimeError):
    """多模态解读失败（未配置 Key / 渲染失败 / 模型调用失败）。"""


def vision_available() -> bool:
    """是否可调用（需配置 DEEPSEEK_API_KEY）。"""
    return bool(config.DEEPSEEK_API_KEY)


_vision_llm = None


def _get_vision_llm():
    """懒加载 vision 客户端（独立于主模型缓存；未配置/初始化失败返回 None）。"""
    global _vision_llm
    if _vision_llm is not None:
        return _vision_llm
    if not vision_available():
        return None
    try:
        from langchain_openai import ChatOpenAI

        _vision_llm = ChatOpenAI(
            model=config.DEEPSEEK_VISION_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=0.2,
            timeout=120,
            max_retries=config.MAX_RETRIES,
            stream_usage=True,  # 流式响应携带 usage（token 用量埋点）
        )
        return _vision_llm
    except Exception as e:
        logger.error("初始化 vision 客户端失败: %s", e)
        return None


def _build_vision_messages(pdf_path: str, page: int, question: str) -> list:
    """渲染页面 → base64 data URL 的图文消息；渲染失败抛 VisionError。"""
    try:
        image = documents.render_page_image(pdf_path, page)
    except ValueError as e:
        raise VisionError(str(e)) from e
    except Exception as e:
        raise VisionError(f"页面渲染失败: {e}") from e
    b64 = base64.b64encode(image).decode()
    prompt = prompts.VISION_PROMPT.format(page=page)
    if question.strip():
        prompt += f"\n用户的具体问题：{question.strip()[:300]}"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]
    return [HumanMessage(content=content)]


async def describe_page_stream(pdf_path: str, page: int, question: str = "",
                               conversation_id: int | None = None) -> AsyncIterator[str]:
    """流式解读一页（逐段 yield 文本）；失败抛 VisionError（含面向用户提示）。"""
    llm = _get_vision_llm()
    if llm is None:
        raise VisionError("未配置 DEEPSEEK_API_KEY，无法进行图表解读")
    msgs = _build_vision_messages(pdf_path, page, question)
    started = time.monotonic()
    usage: dict = {}
    try:
        async for chunk in llm.astream(msgs):
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
            text = chunk.content
            if isinstance(text, str) and text:
                yield text
    except Exception as e:
        db.record_event("api_llm", event_name="vision_page", conversation_id=conversation_id,
                        ok=False, latency_ms=db.ms_since(started), detail=str(e)[:200])
        raise VisionError(f"图表解读失败：{e}") from e
    tokens_in = int(usage.get("input_tokens") or 0)
    db.note_input_tokens(conversation_id, tokens_in)  # 记账：当前上下文 token 规模（自动整理触发判定）
    db.record_event("api_llm", event_name="vision_page", conversation_id=conversation_id,
                    latency_ms=db.ms_since(started),
                    tokens_in=tokens_in,
                    tokens_out=int(usage.get("output_tokens") or 0),
                    detail=f"第 {page} 页")


async def describe_page_text(pdf_path: str, page: int, question: str = "") -> str:
    """非流式收集版（Agent 工具用）：返回解读全文；不可用/失败抛 VisionError。"""
    parts: list[str] = []
    async for piece in describe_page_stream(pdf_path, page, question):
        parts.append(piece)
    text = "".join(parts).strip()
    if not text:
        raise VisionError("vision 模型未返回内容")
    return text
