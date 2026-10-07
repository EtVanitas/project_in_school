"""护栏式 ReAct Agent：模型自主调用工具的多步循环。

核心设计：统一入口、4 个只读论文工具 + 1 个勘误提议工具（声明与执行收口在 tools.py 四段管线）、
MAX_STEPS 上限、超时控制、失败自愈；事件流（tool_start / tool_end / delta / retract / note_issue /
termination）供 SSE 转发与 api 层落库；模型层消息按轮原子写入 model_messages（断连整轮不写）。
运行内复用（三期）：单次运行内同工具同参直接返上次结果（零失效风险），事件带 cached 标记。
"""

import json
import logging
import time
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from . import config, db
from .llms import llm as llm_module, tools
from .llms.context import build_model_context

logger = logging.getLogger(__name__)

# Agent 配置（从 config 导入）
MAX_STEPS = config.MAX_STEPS
TOTAL_TIMEOUT = config.TOTAL_TIMEOUT
_MAX_TOOL_OUTPUT = config.MAX_TOOL_OUTPUT


def _assemble_tool_calls(gathered) -> list[dict]:
    """从聚合的流式 chunk 提取工具调用（优先解析结果，兜底手工拼接分片）。"""
    if gathered.tool_calls:
        return [{"name": c.get("name") or "", "args": c.get("args") or {}, "id": c.get("id") or ""}
                for c in gathered.tool_calls]
    calls: dict = {}
    for tc in gathered.tool_call_chunks or []:
        idx = tc.get("index", 0)
        entry = calls.setdefault(idx, {"name": "", "args": "", "id": ""})
        if tc.get("name"):
            entry["name"] += tc["name"]
        if tc.get("args"):
            entry["args"] += tc["args"]
        if tc.get("id"):
            entry["id"] = tc["id"]
    out: list[dict] = []
    for entry in calls.values():
        args: dict = {}
        if entry["args"]:
            try:
                args = json.loads(entry["args"])
            except json.JSONDecodeError:
                args = {}
        out.append({"name": entry["name"], "args": args, "id": entry["id"]})
    return out


async def run_agent(doc: dict, history: list[dict], question: str, selected_text: str,
                    context: str, used_pages: set[int],
                    conversation_id: int | None = None,
                    memory_text: str = "", profile_text: str = "") -> AsyncIterator[dict]:
    """ReAct 主循环；yield 事件 dict。

    对外事件：tool_start / tool_end / delta / retract / note_issue；
    内部事件：termination（终止原因，由 graph.agent_node 捕获进 state 经 done 下发，前端不可见）。

    doc 为最小文档视图（仅 id/title/pdf_path/arxiv_id），由 graph.agent_node 收敛后传入。
    used_pages 由调用方传入，原地并入所有工具返回内容涉及的页码（供引用校验）。
    LLM 调用失败时对外抛出异常，由调用方决定是否降级为直达问答。
    每轮 LLM 调用与工具执行均写入埋点（token/延迟/成败）；模型层消息按轮原子写入 model_messages。
    memory_text 为该论文的笔记记忆（历史讨论整理，附于上下文后供参考）。
    profile_text 为用户画像（跨论文提问行为统计，随系统提示注入）。

    主循环结构：初始化 → 每轮「[1] 流式调用 → [2] 记账 → [3] 自然结束 / [4][5] 工具执行
    → [6] 轮末落库」；步数 / 超时 / 工具总量触发上限时进入强制收敛段。
    """
    # ---- 初始化：模型 / 工具绑定 / 消息上下文 / 运行态 ----
    llm = llm_module.get_llm()
    if llm is None:
        yield {"type": "delta", "text": "未配置 DEEPSEEK_API_KEY，无法调用模型。请在项目根目录 .env 中配置后重试。"}
        yield {"type": "termination", "reason": "error"}
        return

    bound = llm.bind_tools(tools.openai_tools())
    msgs = build_model_context(kind="agent", doc_title=doc.get("title") or "", history=history,
                               context_text=context, memory_text=memory_text,
                               profile_text=profile_text, selected_text=selected_text,
                               question=question)

    deadline = time.monotonic() + TOTAL_TIMEOUT
    fail_counts: dict[str, int] = {}
    total_calls = 0  # 工具累计调用量（MAX_STEPS 只限轮数，单轮可多发；需控制总量上限）
    tool_cache: dict[str, tools.ToolResult] = {}  # 运行内复用：同工具同参直接返（只缓存成功结果）
    termination_reason = "step_limit"  # 循环自然耗尽即步数上限；其余路径逐点改写
    step = 0
    # ---- ReAct 主循环（每轮各段见 [1]～[6]）----
    while step < MAX_STEPS:
        if time.monotonic() > deadline:
            logger.warning("Agent 整响应超时，强制收敛：%s", question[:40])
            termination_reason = "timeout"
            break
        step += 1
        # [1] 流式调用本轮：gathered 聚合全部 chunk；无工具迹象时边流边转发
        gathered = None
        started = time.monotonic()
        stream = bound.astream(msgs)
        forwarded: list[str] = []  # 本轮已流式转发的文本（若该轮实为工具调用轮，需撤回过渡语）
        try:
            async for chunk in stream:
                gathered = chunk if gathered is None else gathered + chunk
                # 无工具调用迹象时边流边转发（最终回答的流式体验）
                if not gathered.tool_call_chunks and isinstance(chunk.content, str) and chunk.content:
                    forwarded.append(chunk.content)
                    yield {"type": "delta", "text": chunk.content}
        except Exception:
            db.record_event("api_llm", event_name="agent_round", conversation_id=conversation_id,
                             doc_id=doc.get("id"), ok=False,
                             latency_ms=db.ms_since(started), detail=f"step={step} 调用失败")
            raise
        finally:
            await stream.aclose()  # 显式关闭底层 HTTP 流，避免连接池析构告警
        if gathered is None:
            termination_reason = "error"
            break
        # [2] 组装工具调用 + 轮次记账（token 规模供自动整理触发判定）
        tool_calls = _assemble_tool_calls(gathered)
        um = getattr(gathered, "usage_metadata", None) or {}
        tokens_in = int(um.get("input_tokens") or 0)
        db.note_input_tokens(conversation_id, tokens_in)  # 记账：当前上下文 token 规模（自动整理触发判定）
        db.record_event("api_llm", event_name="agent_round", conversation_id=conversation_id,
                         doc_id=doc.get("id"), latency_ms=db.ms_since(started),
                         tokens_in=tokens_in,
                         tokens_out=int(um.get("output_tokens") or 0),
                         detail=f"step={step} tools={len(tool_calls)}")
        # [3] 无工具调用：最终回答（文本已流式转发完毕）→ 自然结束
        if not tool_calls:
            logger.info("Agent 第 %d 轮直接回答（%.1fs）：%s", step, time.monotonic() - started, question[:40])
            yield {"type": "termination", "reason": "natural_stop"}
            return  # 最终轮已流式输出完毕
        # [4] 工具轮准备：撤回过渡语 / 补全 call_id / 助手消息入上下文 / 建立本轮落库缓冲
        if forwarded:  # 工具调用轮：撤回此前误转发的过渡语（前端从末条消息尾部移除）
            leaked = "".join(forwarded)
            logger.info("Agent 第 %d 轮为工具轮，撤回已转发文本 %d 字符", step, len(leaked))
            yield {"type": "retract", "text": leaked}
        for ci, call in enumerate(tool_calls):
            if not call["id"]:
                call["id"] = f"call_{step}_{ci}"
        ai_content = gathered.content if isinstance(gathered.content, str) else ""
        msgs.append(AIMessage(content=ai_content, tool_calls=tool_calls))
        hit_total_limit = False
        turn_results: list[dict] = []  # 本轮模型层缓冲（助手 + 全部工具结果，轮末原子写入）
        # [5] 逐工具执行（三分支：运行内复用 → 连续失败弃用 → 实际执行）
        for call in tool_calls:
            name = call["name"] or "unknown"
            args_summary = tools.args_summary(name, call["args"])
            if total_calls >= config.MAX_TOOL_CALLS:  # 总量护栏：空观测补齐配对后强制收敛
                termination_reason = "tool_limit"
                hit_total_limit = True
                obs = "工具调用已达本轮上限，请基于已收集信息直接回答。"
                msgs.append(ToolMessage(content=obs, tool_call_id=call["id"]))
                turn_results.append({"call_id": call["id"], "content": obs})
                continue
            total_calls += 1
            yield {"type": "tool_start", "call_id": call["id"], "name": name,
                   "args": call["args"], "args_summary": args_summary, "step": step}
            cache_key = f"{name}|{json.dumps(call['args'], sort_keys=True, ensure_ascii=False)}"
            cached = cache_key in tool_cache
            if cached:  # 运行内复用：精确调用（工具+参数）命中优先于工具级失败弃用
                result = tool_cache[cache_key]
                elapsed = 0.0
                logger.info("Agent 工具复用 | step=%d tool=%s", step, name)
            elif fail_counts.get(name, 0) >= 2:
                result = tools.ToolResult(ok=False, summary="连续失败",
                                          observation="该工具连续失败 3 次，请改用其他工具或基于已有信息直接回答。")
                elapsed = 0.0
            else:
                t0 = time.monotonic()
                result = await tools.execute_tool(name, call["args"], doc)
                elapsed = time.monotonic() - t0
                if result.ok:  # 只缓存成功结果（失败可能是暂时性问题，保留重试语义）
                    tool_cache[cache_key] = result
                elif config.counts_failure(result.error_kind):
                    fail_counts[name] = fail_counts.get(name, 0) + 1
                used_pages.update(result.pages)
                logger.info("Agent 工具统计 | step=%d tool=%s ok=%s %.2fs chars=%d",
                            step, name, result.ok, elapsed, len(result.observation))
            db.record_event("tool", event_name=name, conversation_id=conversation_id,
                             doc_id=doc.get("id"), ok=result.ok, latency_ms=int(elapsed * 1000),
                             detail=(result.summary + "（运行内复用）" if cached else result.summary))
            yield {"type": "tool_end", "call_id": call["id"], "name": name, "ok": result.ok,
                   "is_error": not result.ok, "result_summary": result.summary,
                   "latency_ms": int(elapsed * 1000), "step": step, "cached": cached,
                   "pages": sorted(result.pages), "snippets": result.snippets}
            if name == "flag_note_issue" and result.ok:  # 勘误建议结构化转发（前端渲染更正卡片）
                ot = str(call["args"].get("original_text") or "").strip()
                ct = str(call["args"].get("correction") or "").strip()
                if ot and ct:
                    page_no = call["args"].get("evidence_page")
                    yield {"type": "note_issue", "call_id": call["id"], "step": step,
                           "arxiv_id": str(doc.get("arxiv_id") or doc.get("id") or ""),
                           "original_text": ot, "correction": ct,
                           "reason": str(call["args"].get("reason") or "").strip(),
                           "evidence_page": int(page_no) if isinstance(page_no, int) and page_no > 0 else 0}
            obs_text = result.observation[:_MAX_TOOL_OUTPUT]
            turn_results.append({"call_id": call["id"], "content": obs_text})
            msgs.append(ToolMessage(content=obs_text, tool_call_id=call["id"]))
        # [6] 轮末：原子写入本轮消息 + 总量护栏命中则退出循环
        if conversation_id and turn_results:  # 模型层轮级原子写入（断连则整轮不写，序列保持干净）
            try:
                db.append_model_turn(conversation_id, ai_content, tool_calls, turn_results)
            except Exception:
                logger.warning("模型层轮级写入失败（忽略，不影响本轮回答）", exc_info=True)
        if hit_total_limit:
            break

    # ---- 强制收敛：步数耗尽 / 超时 / 总量护栏被触发 → 不带工具再问一次，基于已收集信息作答 ----
    yield {"type": "termination", "reason": termination_reason}
    logger.info("Agent 达到步数上限或超时（step=%d, total_calls=%d, reason=%s），强制收敛最终回答",
                step, total_calls, termination_reason)
    msgs.append(HumanMessage(
        "（系统提示：工具调用已达上限或超时。请立即基于已收集到的信息用中文给出最终回答，"
        "不要再调用工具，标注信息来源页码。）"))
    started = time.monotonic()
    gathered = None
    try:
        stream = llm.astream(msgs)
        try:
            async for chunk in stream:
                gathered = chunk if gathered is None else gathered + chunk
                if isinstance(chunk.content, str) and chunk.content:
                    yield {"type": "delta", "text": chunk.content}
        finally:
            await stream.aclose()
        um = getattr(gathered, "usage_metadata", None) or {}
        tokens_in = int(um.get("input_tokens") or 0)
        db.note_input_tokens(conversation_id, tokens_in)  # 记账：当前上下文 token 规模（自动整理触发判定）
        db.record_event("api_llm", event_name="converge", conversation_id=conversation_id,
                         doc_id=doc.get("id"), latency_ms=db.ms_since(started),
                         tokens_in=tokens_in,
                         tokens_out=int(um.get("output_tokens") or 0), detail=f"step={step} 强制收敛")
    except Exception:
        db.record_event("api_llm", event_name="converge", conversation_id=conversation_id,
                         doc_id=doc.get("id"), ok=False,
                         latency_ms=db.ms_since(started), detail="强制收敛失败")
        logger.exception("强制收敛失败")
        yield {"type": "delta", "text": "（已达到工具调用上限，未能生成最终回答，请重新提问。）"}
