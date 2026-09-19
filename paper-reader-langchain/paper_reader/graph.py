"""LangGraph 状态图：状态定义 + 节点实现 + 图组装。

架构：guard→retrieve→agent（ReAct）→log（4 节点线性图）；RRF 融合 Top-12 直注入 Agent；
本地 1.7B 仅用于对话总结（信息压缩），所有文字生成由 DeepSeek 承担。
降级策略：检索异常转 Agent 自救，零命中均匀采样兜底，Agent 异常预检索直达问答。
"""

import asyncio
import hashlib
import logging
from typing import TypedDict

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from . import agent as agent_module
from . import config, db, documents, llm, notes

logger = logging.getLogger(__name__)

TOP_CANDIDATES = 12   # 检索返回的候选块数（RRF 融合后）
KEEP_TOP = 6          # 保留的上下文块数


# ===== 状态定义 =====

class QAState(TypedDict, total=False):
    """对话请求全生命周期状态（节点间传递；含 API 层收集事件所需信息）。

    所有字段必须 JSON 可序列化（checkpointer 落盘约束）。
    """

    # 输入（api.py 每轮构建）
    doc: dict              # 文档行（id/title/pdf_path 等；检索与 Agent 使用）
    conversation_id: int
    question: str          # 用户问题（guard 规范化后）
    selected_text: str
    pending_msg_id: int    # 已入库的当前用户消息 id（短期记忆过滤用）

    # 检索
    chunks: list[dict]     # 候选块 [{"id","page","text"}]（RRF 融合顺序）

    # 上下文与产出
    context: str           # 注入 Agent 的检索上下文（带【第 N 页】标记）
    used_pages: list[int]  # 答案涉及页码（引用校验用）
    final_answer: str      # Agent 最终答案

    # 降级与观测
    failed: bool           # 严重降级标志（Agent 失败置位，可观测）
    degraded: str          # 降级原因描述（可观测，不阻断回答）


def initial_state(doc: dict, conversation_id: int, question: str,
                  selected_text: str = "", pending_msg_id: int = 0) -> dict:
    """构建每轮请求的完整初始状态（全字段显式重置，避免 checkpoint 残留干扰）。"""
    return {
        "doc": doc,
        "conversation_id": int(conversation_id),
        "question": question,
        "selected_text": selected_text or "",
        "pending_msg_id": int(pending_msg_id or 0),
        "chunks": [],
        "context": "", "used_pages": [], "final_answer": "",
        "failed": False, "degraded": "",
    }


# ===== 事件流（轨迹 / 流式协议） =====

def _writer():
    """custom 流写入器；图外调用（如单测直调节点）返回静默丢弃器。"""
    try:
        return get_stream_writer()
    except RuntimeError:
        def _discard(_payload: dict) -> None:
            pass
        return _discard


def _emit_step(tool: str, args_summary: str) -> int:
    """发送步骤开始事件，返回唯一 step 号（基于 MD5 哈希，相同输入产生相同编号便于调试）。"""
    step = int(hashlib.md5(f"{tool}:{args_summary}".encode()).hexdigest(), 16) % (10**6)
    _writer()({"type": "agent_step", "step": step, "tool": tool, "args_summary": args_summary})
    return step


def _emit_obs(step: int, tool: str, ok: bool, summary: str) -> None:
    """发送步骤完成事件（与开始事件以 step+tool 配对补全）。"""
    _writer()({"type": "agent_observation", "step": step, "tool": tool, "ok": ok, "summary": summary})


# ===== 节点实现 =====

async def guard_node(state: dict) -> dict:
    """护栏：问题规范化（空白折叠 + 超长截断），空消息按问候处理。"""
    q = " ".join(str(state.get("question") or "").split())
    return {"question": q[:1000] or "你好"}


async def retrieve_node(state: dict) -> dict:
    """混合检索（BM25+ 覆盖度 + 向量 RRF）Top-12；零命中时均匀采样兜底（防跨语言零召回）。"""
    query = state["question"]
    pdf_path = str((state.get("doc") or {}).get("pdf_path") or "")
    step = _emit_step("retrieve", f"混合检索「{query[:40]}」")
    try:
        chunks = await asyncio.to_thread(documents.retrieve_top_chunks, pdf_path, query, TOP_CANDIDATES)
        if not chunks:  # 词法 + 向量均零命中（如中文问题对英文正文且向量不可用）：均匀采样保证全文视野
            chunks = await asyncio.to_thread(documents.sample_chunks, pdf_path, TOP_CANDIDATES)
    except Exception as e:
        logger.exception("检索失败，转 Agent 工具自救：%s", e)
        _emit_obs(step, "retrieve", False, "检索失败，转工具自查")
        return {"chunks": [], "failed": True}
    cand = [{"id": c.id, "page": c.page, "text": c.text} for c in chunks]
    head_pages = sorted({c["page"] for c in cand[:KEEP_TOP]})
    _emit_obs(step, "retrieve", True,
              f"候选 {len(cand)} 块" + (f"（头部覆盖第 {'、'.join(map(str, head_pages))} 页）" if head_pages else ""))
    return {"chunks": cand}


async def agent_node(state: dict) -> dict:
    """ReAct 内核：短期记忆（最近消息窗口）+ 该论文笔记记忆注入，事件原样转发。"""
    doc = state.get("doc") or {}
    conv_id = int(state.get("conversation_id") or 0)
    question = state["question"]
    selected_text = str(state.get("selected_text") or "")
    # 上下文直接从 chunks 构建（RRF 排序后的头部）
    chunks = state.get("chunks") or []
    pdf_path = str((doc or {}).get("pdf_path") or "")
    context, used_pages_list = _build_context_from_chunks(chunks, pdf_path)
    step = _emit_step("agent", "多轮工具推理与回答")
    writer = _writer()

    # 短期记忆：最近消息窗口原文（失败自动回退最近历史）；过滤已入库的当前问题
    try:
        recent = await notes.build_short_term(conv_id)
    except Exception as e:
        logger.warning("短期记忆构建失败，退化为最近历史: %s", e)
        recent = db.list_messages(conv_id, limit=config.CHAT_HISTORY_TURNS)
    pending = state.get("pending_msg_id")
    history_msgs = [m for m in recent if m.get("id") != pending]

    # 该论文记忆：笔记文件（「总结整理」产物或用户手写）注入作背景；失败返回空串，不阻断问答
    memory_text = ""
    try:
        key = str(doc.get("arxiv_id") or doc.get("id") or "")
        if key:
            memory_text = await asyncio.to_thread(notes.paper_memory_text, key)
    except Exception:
        logger.warning("笔记记忆读取失败（跳过注入）", exc_info=True)

    used_pages: set[int] = set(used_pages_list)
    collected: list[str] = []
    degraded = False
    try:
        async for evt in agent_module.run_agent(
                doc, history_msgs, question, selected_text, context, used_pages,
                conversation_id=conv_id or None, memory_text=memory_text):
            writer(evt)
            if evt.get("type") == "delta":
                collected.append(str(evt.get("text") or ""))
    except Exception:
        logger.exception("Agent 执行异常，降级直达问答")
        degraded = True
        if not collected:  # 已有部分输出时不再追加（避免拼接矛盾内容）
            try:
                async for piece in llm.stream_answer(
                        str(doc.get("title") or ""), context, history_msgs, question,
                        selected_text, conversation_id=conv_id or None):
                    collected.append(piece)
                    writer({"type": "delta", "text": piece})
            except Exception as e2:
                logger.exception("降级问答失败")
                text = f"\n\n（调用模型失败：{e2}，请稍后重试）"
                collected.append(text)
                writer({"type": "delta", "text": text})
    answer = "".join(collected)
    _emit_obs(step, "agent", not degraded, "推理完成" if answer else "未产出内容")
    out: dict = {"final_answer": answer, "used_pages": sorted(used_pages)}
    if degraded:
        out["failed"] = True
        out["degraded"] = "Agent 异常，已降级直达问答"
    return out


def _build_context_from_chunks(chunks: list[dict], pdf_path: str) -> tuple[str, list[int]]:
    """从 RRF 排序后的候选块直接构建上下文（取头部 KEEP_TOP 个）。"""
    if not chunks or not pdf_path:
        return "", []
    picked = chunks[:KEEP_TOP]
    all_chunks = documents.get_chunks(pdf_path)
    ids = [int(c["id"]) for c in picked if 0 <= int(c["id"]) < len(all_chunks)]
    text = documents.render_context(all_chunks, ids)
    pages = sorted({all_chunks[i].page for i in ids})
    return text, pages


async def log_node(state: dict) -> dict:
    """图级观测：页码/降级写入埋点（供 /api/stats 汇总）。"""
    db.record_event(
        "interaction", name="graph_done",
        conversation_id=state.get("conversation_id"),
        doc_id=(state.get("doc") or {}).get("id"),
        detail=(f"pages={len(state.get('used_pages') or [])} "
                f"failed={bool(state.get('failed'))} degraded={state.get('degraded') or '-'}"),
    )
    return {}


# ===== 图组装与执行配置 =====

_graph = None
_saver: AsyncSqliteSaver | None = None


async def _get_saver_async() -> AsyncSqliteSaver:
    """AsyncSqliteSaver 单例（懒初始化）。"""
    global _saver
    if _saver is None:
        conn = await aiosqlite.connect(str(config.DB_PATH))
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        _saver = saver
        logger.info("checkpointer 就绪：%s", config.DB_PATH)
    return _saver


def build_graph(checkpointer=None):
    """构建并编译状态图（checkpointer 可注入，便于测试与断点恢复演练）。

    架构：guard→retrieve→agent→log（4 节点线性图，RRF 排序后直接建上下文）。
    """
    g = StateGraph(QAState)
    g.add_node("guard", guard_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("agent", agent_node)
    g.add_node("log", log_node)

    g.add_edge(START, "guard")
    g.add_edge("guard", "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_edge("agent", "log")
    g.add_edge("log", END)
    return g.compile(checkpointer=checkpointer)


async def get_graph_async():
    """生产单例（含 AsyncSqliteSaver 持久化；供 async 端点调用）。"""
    global _graph
    if _graph is None:
        saver = await _get_saver_async()
        if _graph is None:
            _graph = build_graph(saver)
    return _graph


async def aclose() -> None:
    """服务关闭时释放 checkpointer 连接（幂等）。"""
    global _saver, _graph
    if _saver is not None:
        await _saver.conn.close()
        _saver = None
        _graph = None
        logger.info("checkpointer 已关闭")


def run_config(conversation_id: int) -> dict:
    """执行配置：thread_id = 对话 id（断点续跑维度）。"""
    return {"configurable": {"thread_id": str(conversation_id)}}
