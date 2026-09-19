"""模型层：DeepSeek 云端（问答/多模态/引用校验）+ 本地 Qwen3-1.7B（对话总结）。

云端：流式问答、多模态图表解读、引用校验护栏；未配置 Key 时返回友好提示。
本地：懒加载单例 + 串行锁，仅承担「信息压缩」任务（对话总结），所有文字生成由 DeepSeek 承担。
"""

import asyncio
import base64
import logging
import re
import time
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import config, db, documents

logger = logging.getLogger(__name__)

_llm_cache: dict = {}

_SYSTEM_ANSWER = (
    "你是「智能论文阅读助手」，基于用户提供的论文内容回答问题。\n"
    "规则：1) 始终用中文回答；2) 标注信息来源的页码（如「第 3 页」）；"
    "3) 只引用论文内容中出现过的【第 N 页】标记对应的页码，不得引用未提供的页码；"
    "4) 论文内容不足以回答时明确说明「资料不足」，不要编造；"
    "5) 用 Markdown 组织答案，结构清晰。"
)


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


def _build_answer_messages(doc_title: str, context: str, history: list[dict],
                           question: str, selected_text: str = "") -> list:
    """组装问答消息：系统提示 + 多轮历史 + 论文上下文 + 当前问题。"""
    msgs: list = [SystemMessage(_SYSTEM_ANSWER)]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(m["content"]))
        else:
            msgs.append(AIMessage(m["content"]))
    user_text = f"论文标题：{doc_title}\n\n论文内容：\n{context}\n\n"
    if selected_text:
        user_text += f"我在论文中选中的片段（重点参考）：\n{selected_text}\n\n"
    user_text += f"我的问题：{question}"
    msgs.append(HumanMessage(user_text))
    return msgs


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
    try:
        async for chunk in llm.astream(_build_answer_messages(doc_title, context, history, question, selected_text)):
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
            text = chunk.content
            if isinstance(text, str) and text:
                yield text
        db.record_event("api_llm", name="stream_answer", conversation_id=conversation_id,
                        latency_ms=db.ms_since(started),
                        tokens_in=int(usage.get("input_tokens") or 0),
                        tokens_out=int(usage.get("output_tokens") or 0))
    except Exception as e:
        logger.exception("流式回答失败")
        db.record_event("api_llm", name="stream_answer", conversation_id=conversation_id,
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


# ===== 对话总结整理（「总结整理」按钮：压缩对话 → 笔记） =====

_SUMMARY_SYS = (
    "你是论文阅读助手的对话整理器。把用户与助手的一段对话压缩成结构化中文笔记（Markdown）：\n"
    "## 讨论主题\n（本次对话围绕的核心问题，1-3 条）\n"
    "## 核心结论\n（对话得出的关键结论与细节，保留页码引用）\n"
    "## 待跟进问题\n（尚未解决或值得继续追问的点）\n"
    "要求：忠实于对话内容，不编造、不添加对话中没有的信息；总长度控制在 800 字以内。"
)


def _format_dialog(messages: list[dict], max_chars: int) -> str:
    """把消息列表格式化为对话文本（单条截断；超出总上限保留最新部分）。"""
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


async def summarize_dialog(doc_title: str, messages: list[dict],
                           conversation_id: int | None = None,
                           max_chars: int = 12000) -> str:
    """把一段对话压缩为结构化总结（Markdown），使用本地模型（零 Token 成本）；未配置/失败抛 RuntimeError。"""
    if not config.LOCAL_LLM_ENABLED:
        raise RuntimeError("本地模型已关闭，无法整理对话")
    user_text = f"论文标题：{doc_title}\n\n对话记录：\n{_format_dialog(messages, max_chars)}"
    try:
        # 长文生成（上限 2000 token）：超时独立放宽；埋点统一由 _local_run 记录
        return await _local_run([{"role": "system", "content": _SUMMARY_SYS},
                                 {"role": "user", "content": user_text}],
                                max_new_tokens=2000, timeout=120, tag="summarize",
                                conversation_id=conversation_id)
    except LocalLLMError as e:
        raise RuntimeError(f"整理对话失败：{e}") from e


# ===== 多模态图表解读（PDF 单页渲染 → DeepSeek vision 流式） =====

_VISION_PROMPT = (
    "这是一篇学术论文的第 {page} 页截图。请用中文解读本页的图、表或公式："
    "说明它展示的内容、关键数据或结论；若本页没有图表，简要概述本页正文要点。"
)


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
    prompt = _VISION_PROMPT.format(page=page)
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
        db.record_event("api_llm", name="vision_page", conversation_id=conversation_id,
                        ok=False, latency_ms=db.ms_since(started), detail=str(e)[:200])
        raise VisionError(f"图表解读失败：{e}") from e
    db.record_event("api_llm", name="vision_page", conversation_id=conversation_id,
                    latency_ms=db.ms_since(started),
                    tokens_in=int(usage.get("input_tokens") or 0),
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


# ===== 本地轻量模型（Qwen3-1.7B：对话总结，零 API token） =====

_THINK_RE = re.compile(r" thinking.*?(?:<｜end▁of▁thinking｜>|$)", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


class LocalLLMError(RuntimeError):
    """本地模型不可用 / 超时 / 输出异常（调用方应捕获并降级）。"""


_model_pair = None  # (model, tokenizer)
_async_lock: asyncio.Lock | None = None


def _async_guard() -> asyncio.Lock:
    global _async_lock
    if _async_lock is None:
        _async_lock = asyncio.Lock()
    return _async_lock


def local_status() -> dict:
    """供健康检查/统计展示。"""
    return {"enabled": config.LOCAL_LLM_ENABLED, "loaded": _model_pair is not None,
            "name": config.LOCAL_LLM_NAME}


def _load_local_model():
    """加载模型单例；失败抛 LocalLLMError。"""
    global _model_pair
    if _model_pair is not None:
        return _model_pair
    with config.MODEL_LOAD_LOCK:  # 与向量模型共用加载锁，避免 transformers 并发导入竞态
        if _model_pair is not None:
            return _model_pair
        if not config.LOCAL_LLM_ENABLED:
            raise LocalLLMError("本地模型已关闭（LOCAL_LLM_ENABLED=0）")
        path = config.LOCAL_MODEL_DIR / config.LOCAL_LLM_NAME
        if not path.exists():
            raise LocalLLMError(f"未找到本地模型目录：{path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            config.silence_hf()
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16).to("cuda")
            else:
                model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
            model.eval()
            _model_pair = (model, tokenizer)
            logger.info("本地模型加载完成：%s（device=%s）", path, model.device)
            return _model_pair
        except Exception as e:
            raise LocalLLMError(f"加载本地模型失败：{e}") from e


def _build_prompt(tokenizer, messages: list[dict]) -> str:
    """应用 chat 模板（优先关闭 thinking）。"""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        msgs = [dict(m) for m in messages]
        msgs[-1]["content"] = msgs[-1]["content"] + " /no_think"
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _local_generate(messages: list[dict], max_new_tokens: int, temperature: float = 0.0) -> tuple[str, int, int]:
    """同步生成一次，返回 (去 thinking 后的纯文本，输入 token 数，输出 token 数)。"""
    import torch

    model, tokenizer = _load_local_model()
    prompt = _build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs: dict = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    n_in = int(inputs["input_ids"].shape[1])
    text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
    return _THINK_RE.sub("", text).strip(), n_in, int(out.shape[1] - n_in)


async def _local_run(messages: list[dict], max_new_tokens: int, temperature: float = 0.0,
                     timeout: float | None = None, tag: str = "task",
                     conversation_id: int | None = None) -> str:
    """串行 + 超时包装：全局锁内执行，保证单卡不并发；带埋点（延迟/token/成败）。

    超时后线程无法从外部取消：等它真正跑完再释放锁，避免残留推理与下一次调用同卡并发。
    """
    t = timeout or config.LOCAL_LLM_TIMEOUT
    started = time.monotonic()
    async with _async_guard():
        fut = asyncio.get_running_loop().run_in_executor(
            None, _local_generate, messages, max_new_tokens, temperature)
        try:
            text, n_in, n_out = await asyncio.wait_for(asyncio.shield(fut), t)
        except LocalLLMError:
            db.record_event("local_llm", name=tag, conversation_id=conversation_id, ok=False,
                             latency_ms=db.ms_since(started), detail="模型加载/执行失败")
            raise
        except asyncio.TimeoutError as e:
            try:  # 等底层线程结束（锁保持），而非让残留推理与下轮调用并发抢卡
                await fut
            except Exception:
                pass
            db.record_event("local_llm", name=tag, conversation_id=conversation_id, ok=False,
                             latency_ms=db.ms_since(started), detail=f"超时（>{t}s）")
            raise LocalLLMError(f"本地模型超时（>{t}s）") from e
        except Exception as e:
            db.record_event("local_llm", name=tag, conversation_id=conversation_id, ok=False,
                             latency_ms=db.ms_since(started), detail=str(e)[:200])
            raise LocalLLMError(f"本地模型执行失败：{e}") from e
    db.record_event("local_llm", name=tag, conversation_id=conversation_id,
                     latency_ms=db.ms_since(started), tokens_in=n_in, tokens_out=n_out)
    return text

