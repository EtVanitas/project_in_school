"""护栏式 ReAct Agent：模型自主调用工具的多步循环。

核心设计：统一入口、4 个只读论文工具 + 1 个勘误提议工具、MAX_STEPS 上限、超时控制、失败自愈；
工具轨迹写入 db.agent_steps，事件流供 SSE 转发，统计由 db.record_event 记录。
"""

import asyncio
import json
import logging
import time
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from . import db, documents
from . import llm as llm_module
from . import config

logger = logging.getLogger(__name__)

# Agent 配置（从 config 导入）
MAX_STEPS = config.MAX_STEPS
TOOL_TIMEOUT = config.TOOL_TIMEOUT
VISION_TOOL_TIMEOUT = config.VISION_TOOL_TIMEOUT
TOTAL_TIMEOUT = config.TOTAL_TIMEOUT
_MAX_TOOL_OUTPUT = config._MAX_TOOL_OUTPUT

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_in_paper",
            "description": "在当前打开的论文中检索与查询最相关的段落，返回带页码的原文片段。"
                           "查找具体方法、实验结果、术语定义时使用。建议用英文关键词。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索关键词（英文效果更佳）"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "读取当前论文指定页码的完整文本。已知目标内容所在页码、需要该页全部细节时使用。",
            "parameters": {
                "type": "object",
                "properties": {"page": {"type": "integer", "description": "页码（1 起）"}},
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_outline",
            "description": "获取当前论文的章节目录（标题 + 页码）。用于了解论文整体结构或定位主题所在章节。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_page",
            "description": "用多模态模型查看当前论文指定页的图/表/公式截图并解读。"
                           "当问题涉及图表内容、图像细节、公式排版（文本提取难以还原）时使用。",
            "parameters": {
                "type": "object",
                "properties": {"page": {"type": "integer", "description": "页码（1 起）"}},
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_note_issue",
            "description": "提交「历史讨论记忆（笔记）」的勘误建议：核实后发现笔记中某条结论与论文原文矛盾、明显有误时使用。"
                           "仅在已用工具核实后调用；不会直接修改笔记，系统将把建议展示给用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_text": {"type": "string", "description": "笔记中可能有误的原句（尽量逐字摘录，便于定位替换）"},
                    "correction": {"type": "string", "description": "建议的修正文本（以论文事实为准）"},
                    "reason": {"type": "string", "description": "问题说明：与论文何处矛盾 / 无依据 / 自相矛盾"},
                    "evidence_page": {"type": "integer", "description": "核实的依据页码（论文页，1 起；可选）"},
                },
                "required": ["original_text", "correction", "reason"],
            },
        },
    },
]

_SYSTEM_AGENT = (
    "你是「智能论文阅读助手」的 Agent：通过工具访问当前论文，自主决策、多步完成任务。\n"
    "当前论文标题：{title}\n"
    "工作规则：\n"
    "1) 当前论文的预检索上下文已随消息提供（带【第 N 页】标记）；信息足够时直接回答，不要多余调用工具；\n"
    "2) 需要更多细节时用 search_in_paper（建议英文关键词）；已知具体页码时用 read_page；需要结构时用 get_outline；"
    "问题涉及图表/图像内容时用 look_at_page 查看该页截图；\n"
    "3) 若提供了「历史讨论记忆」，仅在确有帮助时参考——它是过往对话的自动整理，可能有误；"
    "发现其中结论与论文原文矛盾时先核实、以论文为准，并调用 flag_note_issue 提交勘误建议（系统展示给用户确认）；\n"
    "4) 回答始终用中文，标注信息来源页码（只引用实际出现过的页码）；资料不足时明确说明，不要编造；\n"
    "5) 调用工具时直接调用，不要在回复里写「让我查一下」之类的过渡语；用 Markdown 组织最终答案。"
)


def _args_summary(name: str, args: dict) -> str:
    """工具调用参数的一句话摘要（前端轨迹展示用）。"""
    q = str(args.get("query", "")).strip()[:50]
    if name == "search_in_paper":
        return f"检索「{q}」"
    if name == "read_page":
        return f"读取第 {args.get('page', '?')} 页"
    if name == "get_outline":
        return "获取论文目录"
    if name == "look_at_page":
        return f"查看第 {args.get('page', '?')} 页图表"
    if name == "flag_note_issue":
        return "提交笔记勘误建议"
    return f"调用 {name}"


def _exec_tool(name: str, args: dict, doc: dict) -> tuple[str, set[int], str]:
    """同步执行工具（在线程池中运行），返回 (观测文本, 涉及页码集合, 过程摘要)。"""
    pdf_path = doc.get("pdf_path") or ""

    if name == "search_in_paper":
        query = str(args.get("query", "")).strip()
        if not query:
            return "缺少 query 参数。", set(), "参数缺失"
        hits = documents.search_chunks(pdf_path, query, top_k=4)
        if not hits:
            return "未检索到相关段落（可换用更具体的英文关键词，或用 read_page 浏览指定页）。", set(), "未命中"
        pages = {c.page for c in hits}
        text = "\n\n".join(f"【第 {c.page} 页】{c.text[:800]}" for c in hits)
        return text, pages, f"命中 {len(hits)} 个段落（第 " + "、".join(str(p) for p in sorted(pages)) + " 页）"

    if name == "read_page":
        try:
            page = int(args.get("page", 0))
        except (TypeError, ValueError):
            return "page 参数无效。", set(), "参数缺失"
        text = documents.get_page_text(pdf_path, page)
        if text.startswith("页码超出范围"):
            return text, set(), "页码越界"
        return text[:_MAX_TOOL_OUTPUT], {page}, f"读取第 {page} 页（{len(text)} 字符）"

    if name == "get_outline":
        toc = documents.get_outline(pdf_path)
        if not toc:
            return "该 PDF 没有内置章节目录。", set(), "无目录"
        lines = "\n".join(
            f"{'  ' * (t['level'] - 1)}{t['title']}（第 {t['page']} 页）" for t in toc[:60])
        return lines, {t["page"] for t in toc}, f"{len(toc)} 个章节"

    if name == "flag_note_issue":
        original = str(args.get("original_text", "")).strip()
        correction = str(args.get("correction", "")).strip()
        if not original or not correction:
            return "缺少 original_text 或 correction 参数（请先核实再提交）。", set(), "参数缺失"
        page = args.get("evidence_page")
        pages = {int(page)} if isinstance(page, int) and page > 0 else set()
        return ("勘误建议已提交，系统将展示更正卡片供用户确认（不会直接修改笔记）。请继续完成回答。",
                pages, "提交勘误建议")

    return f"未知工具：{name}", set(), "未知工具"


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
                    memory_text: str = "") -> AsyncIterator[dict]:
    """ReAct 主循环；yield 事件 dict（agent_step / agent_observation / delta / note_issue / retract）。

    used_pages 由调用方传入，原地并入所有工具返回内容涉及的页码（供引用校验）。
    LLM 调用失败时对外抛出异常，由调用方决定是否降级为直达问答。
    每轮 LLM 调用与工具执行均写入埋点（token/延迟/成败）。
    memory_text 为该论文的笔记记忆（历史讨论整理，附于上下文后供参考）。
    """
    llm = llm_module.get_llm()
    if llm is None:
        yield {"type": "delta", "text": "未配置 DEEPSEEK_API_KEY，无法调用模型。请在项目根目录 .env 中配置后重试。"}
        return

    bound = llm.bind_tools(_TOOLS)
    msgs: list = [SystemMessage(_SYSTEM_AGENT.format(title=doc.get("title") or "未命名"))]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(m["content"]))
        else:
            msgs.append(AIMessage(m["content"]))
    user_text = f"当前论文的预检索上下文：\n{context}\n\n" if context else ""
    if memory_text:
        user_text += memory_text + "\n\n"
    if selected_text:
        user_text += f"我选中的片段（重点参考）：\n{selected_text}\n\n"
    user_text += f"我的问题：{question}"
    msgs.append(HumanMessage(user_text))

    deadline = time.monotonic() + TOTAL_TIMEOUT
    fail_counts: dict[str, int] = {}
    step = 0
    while step < MAX_STEPS:
        if time.monotonic() > deadline:
            logger.warning("Agent 整响应超时，强制收敛：%s", question[:40])
            break
        step += 1
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
            db.record_event("api_llm", name="agent_round", conversation_id=conversation_id,
                             doc_id=doc.get("id"), ok=False,
                             latency_ms=db.ms_since(started), detail=f"step={step} 调用失败")
            raise
        finally:
            await stream.aclose()  # 显式关闭底层 HTTP 流，避免连接池析构告警
        if gathered is None:
            break
        tool_calls = _assemble_tool_calls(gathered)
        um = getattr(gathered, "usage_metadata", None) or {}
        db.record_event("api_llm", name="agent_round", conversation_id=conversation_id,
                         doc_id=doc.get("id"), latency_ms=db.ms_since(started),
                         tokens_in=int(um.get("input_tokens") or 0),
                         tokens_out=int(um.get("output_tokens") or 0),
                         detail=f"step={step} tools={len(tool_calls)}")
        if not tool_calls:
            logger.info("Agent 第 %d 轮直接回答（%.1fs）：%s", step, time.monotonic() - started, question[:40])
            return  # 最终轮已流式输出完毕
        if forwarded:  # 工具调用轮：撤回此前误转发的过渡语（前端从末条消息尾部移除）
            leaked = "".join(forwarded)
            logger.info("Agent 第 %d 轮为工具轮，撤回已转发文本 %d 字符", step, len(leaked))
            yield {"type": "retract", "text": leaked}
        for ci, call in enumerate(tool_calls):
            if not call["id"]:
                call["id"] = f"call_{step}_{ci}"
        msgs.append(AIMessage(
            content=gathered.content if isinstance(gathered.content, str) else "",
            tool_calls=tool_calls))
        for ci, call in enumerate(tool_calls):
            name = call["name"] or "unknown"
            args_summary = _args_summary(name, call["args"])
            if fail_counts.get(name, 0) >= 2:
                obs, pages, brief, ok = "该工具连续失败 3 次，请改用其他工具或基于已有信息直接回答。", set(), "连续失败", False
                elapsed = 0.0
            else:
                yield {"type": "agent_step", "step": step, "tool": name,
                       "args_summary": args_summary}
                t0 = time.monotonic()
                wait_timeout = VISION_TOOL_TIMEOUT if name == "look_at_page" else TOOL_TIMEOUT
                try:
                    if name == "look_at_page":  # 多模态工具：vision 客户端基于 astream，原生异步调用
                        page_no = int(call["args"].get("page") or 0)
                        obs = await asyncio.wait_for(
                            llm_module.describe_page_text(str(doc.get("pdf_path") or ""), page_no),
                            timeout=wait_timeout)
                        pages = {page_no}
                        brief = f"多模态解读第 {page_no} 页图表（{len(obs)} 字符）"
                    else:
                        obs, pages, brief = await asyncio.wait_for(
                            asyncio.to_thread(_exec_tool, name, call["args"], doc), timeout=wait_timeout)
                    ok = True
                except asyncio.TimeoutError:
                    obs, pages, brief, ok = f"工具执行超时（>{wait_timeout}s）。", set(), "执行超时", False
                except Exception as e:
                    logger.exception("工具 %s 执行失败", name)
                    obs, pages, brief, ok = f"工具执行出错：{e}", set(), "执行出错", False
                if not ok:
                    fail_counts[name] = fail_counts.get(name, 0) + 1
                used_pages.update(pages)
                elapsed = time.monotonic() - t0
                logger.info("Agent 工具统计 | step=%d tool=%s ok=%s %.2fs chars=%d",
                            step, name, ok, elapsed, len(obs))
            db.record_event("tool", name=name, conversation_id=conversation_id,
                             doc_id=doc.get("id"), ok=ok, latency_ms=int(elapsed * 1000), detail=brief)
            if conversation_id:
                try:  # 会话轨迹（与统计职责分离；失败静默）
                    db.add_agent_step(conversation_id, step, name, args_summary, ok, brief)
                except Exception:
                    pass
            yield {"type": "agent_observation", "step": step, "tool": name, "ok": ok,
                   "summary": (brief + f"（{elapsed:.1f}s）") if ok else brief}
            if name == "flag_note_issue" and ok:  # 勘误建议结构化转发（前端渲染更正卡片）
                ot = str(call["args"].get("original_text") or "").strip()
                ct = str(call["args"].get("correction") or "").strip()
                if ot and ct:
                    page_no = call["args"].get("evidence_page")
                    yield {"type": "note_issue", "step": step,
                           "arxiv_id": str(doc.get("arxiv_id") or doc.get("id") or ""),
                           "original_text": ot, "correction": ct,
                           "reason": str(call["args"].get("reason") or "").strip(),
                           "evidence_page": int(page_no) if isinstance(page_no, int) and page_no > 0 else 0}
            msgs.append(ToolMessage(content=obs[:_MAX_TOOL_OUTPUT], tool_call_id=call["id"]))

    # 步数耗尽 / 超时：强制基于已收集信息收敛（不带工具再问一次）
    logger.info("Agent 达到步数上限或超时（step=%d），强制收敛最终回答", step)
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
        db.record_event("api_llm", name="converge", conversation_id=conversation_id,
                         doc_id=doc.get("id"), latency_ms=db.ms_since(started),
                         tokens_in=int(um.get("input_tokens") or 0),
                         tokens_out=int(um.get("output_tokens") or 0), detail=f"step={step} 强制收敛")
    except Exception:
        db.record_event("api_llm", name="converge", conversation_id=conversation_id,
                         doc_id=doc.get("id"), ok=False,
                         latency_ms=db.ms_since(started), detail="强制收敛失败")
        logger.exception("强制收敛失败")
        yield {"type": "delta", "text": "（已达到工具调用上限，未能生成最终回答，请重新提问。）"}
