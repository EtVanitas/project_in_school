"""本地轻量模型基础设施（Qwen3-4B）：懒加载单例 + 串行锁 + 超时包装 + 埋点。

在线承担前置预判、离线承担整理保底与画像提炼的模型支撑（业务封装见 llm.py）；
单卡串行（全局 asyncio.Lock），超时后等底层线程真正结束再释放锁，
避免残留推理与下一次调用同卡并发。失败统一抛 LocalLLMError（调用方降级）。
"""

import asyncio
import logging
import re
import time

from .. import config, db

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
