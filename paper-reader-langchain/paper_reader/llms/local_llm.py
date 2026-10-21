"""本地轻量模型的一切（Qwen3-4B）：基础设施 + 本地任务业务封装。

基础设施：懒加载单例 + 串行锁 + 超时包装 + 埋点（local_run）；
本地任务：前置预判（意图路由 + query 改写，在线可选）、整理保底、画像提炼（离线）。
在线承担前置预判、离线承担整理保底与画像；单卡串行（全局 asyncio.Lock），
超时后等底层线程真正结束再释放锁，避免残留推理与下一次调用同卡并发。
失败统一抛 LocalLLMError（调用方降级）。云端（DeepSeek）业务封装在 llm.py。
"""

import asyncio
import json
import logging
import re
import time

from .. import config, db
from . import prompts

logger = logging.getLogger(__name__)

# ===== 常量 / 加载 / 生成 / 串行执行 =====

_THINK_RE = re.compile(r" thinking.*?(?:<｜end▁of▁thinking｜>|$)", re.DOTALL)


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


async def local_run(messages: list[dict], max_new_tokens: int, temperature: float = 0.0,
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
            db.record_event("local_llm", event_name=tag, conversation_id=conversation_id, ok=False,
                             latency_ms=db.ms_since(started), detail="模型加载/执行失败")
            raise
        except asyncio.TimeoutError as e:
            try:  # 等底层线程结束（锁保持），而非让残留推理与下轮调用并发抢卡
                await fut
            except Exception:
                pass
            db.record_event("local_llm", event_name=tag, conversation_id=conversation_id, ok=False,
                             latency_ms=db.ms_since(started), detail=f"超时（>{t}s）")
            raise LocalLLMError(f"本地模型超时（>{t}s）") from e
        except Exception as e:
            db.record_event("local_llm", event_name=tag, conversation_id=conversation_id, ok=False,
                             latency_ms=db.ms_since(started), detail=str(e)[:200])
            raise LocalLLMError(f"本地模型执行失败：{e}") from e
    db.record_event("local_llm", event_name=tag, conversation_id=conversation_id,
                     latency_ms=db.ms_since(started), tokens_in=n_in, tokens_out=n_out)
    return text


# ===== 整理保底（本地 4B 分批，云端不可用时的降级链） =====


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


# ===== 画像提炼（离线，本地 4B） =====


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


# ===== 前置预判（在线可选：意图路由 + query 改写合并为一次调用） =====

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
