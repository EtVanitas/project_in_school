"""LangGraph 状态图：状态定义 + 节点实现 + 图组装。

架构：guard→prejudge→retrieve→agent（ReAct）→log（5 节点线性图）；
prejudge 用本地 4B 一次调用完成意图路由+query改写（失败/关闭则行为等价 v0.5.4）；
RRF 融合 Top-12 直注入 Agent；本地 4B 离线承担对话总结。
降级策略：预判失败走默认检索链路，检索异常转 Agent 自救，零命中均匀采样兜底，Agent 异常预检索直达问答。
"""

import asyncio
import logging
from typing import TypedDict

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from . import agent as agent_module
from . import config, db
from .docs import documents, notes, retrieval
from .llms import llm

logger = logging.getLogger(__name__)

TOP_CANDIDATES = 12   # 检索返回的候选块数（RRF 融合后）
KEEP_TOP = 6          # 保留的上下文块数


# ===== 状态定义 =====

class QAState(TypedDict, total=False):
    """对话请求全生命周期状态（节点间传递；含 API 层收集事件所需信息）。

    所有字段必须 JSON 可序列化（checkpointer 落盘约束）。
    """

    # 输入（api.py 每轮构建）
    doc: dict              # 文档行（id/title/pdf_path/arxiv_id；检索与 Agent 使用）
    conversation_id: int
    question: str          # 用户问题（guard 规范化后）
    selected_text: str
    pending_llm_seq: int   # 模型层当前 user 条目的 seq（model_messages 过滤用）

    # 检索（prejudge 节点产出，供 retrieve 分支）
    need_retrieval: bool   # 预判：本题是否需要检索论文（False 跳过预检索注入）
    rewritten_query: str   # 预判：改写出的英文检索关键词（空串表示不用）
    chunks: list[dict]     # 候选块 [{"id","page","text"}]（RRF 融合顺序）

    # 产出与观测
    used_pages: list[int]  # 答案涉及页码（引用校验用）

    # 降级与观测
    failed: bool           # 严重降级标志（Agent 失败置位，可观测）
    degraded: str          # 降级原因描述（可观测，不阻断回答）
    termination_reason: str  # 终止原因（natural_stop/step_limit/tool_limit/timeout/error）
    citations: list[dict]  # 引用回溯：[{page, snippet}]（snippet 空示无原文块）


def initial_state(doc: dict, conversation_id: int, question: str,
                  selected_text: str = "", pending_llm_seq: int = 0) -> dict:
    """构建每轮请求的完整初始状态（全字段显式重置，避免 checkpoint 残留干扰）。"""
    return {
        "doc": doc,
        "conversation_id": int(conversation_id),
        "question": question,
        "selected_text": selected_text or "",
        "pending_llm_seq": int(pending_llm_seq or 0),
        "need_retrieval": True, "rewritten_query": "",
        "chunks": [],
        "used_pages": [],
        "failed": False, "degraded": "",
        "termination_reason": "", "citations": [],
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


# ===== 节点实现 =====

async def guard_node(state: dict) -> dict:
    """护栏：问题规范化（空白折叠 + 超长截断），空消息按问候处理。"""
    q = " ".join(str(state.get("question") or "").split())
    return {"question": q[:1000] or "你好"}


async def prejudge_node(state: dict) -> dict:
    """前置预判（本地 4B 一次调用）：意图路由 + query 改写写进 state；关闭/失败发默认值，行为等价 v0.5.4。"""
    if not config.PREJUDGE_ENABLED:
        return {"need_retrieval": True, "rewritten_query": ""}
    conv_id = int(state.get("conversation_id") or 0)
    # 最近几条历史作指代消解语境（模型层条目尾部，与 agent 节点同源；排除本轮未回答条目）
    history: list[dict] = []
    try:
        if conv_id:
            rows = db.list_model_messages(conv_id)
            pending_seq = state.get("pending_llm_seq")
            text_rows = [m for m in rows if m.get("role") in ("user", "assistant")
                         and m.get("seq") != pending_seq]
            history = text_rows[-4:]
    except Exception:
        logger.warning("预判取历史失败（仅用当前问题判断）", exc_info=True)
    decision = await llm.prejudge(state["question"], history)
    if decision is None:
        return {"need_retrieval": True, "rewritten_query": ""}
    need, rewrite = decision["need_retrieval"], decision["query"]
    return {"need_retrieval": need, "rewritten_query": rewrite}


async def retrieve_node(state: dict) -> dict:
    """混合检索（BM25+ 覆盖度 + 向量 RRF）Top-12；零命中时均匀采样兜底（防跨语言零召回）。

    预判分支：need_retrieval=False 直接空块跳过注入；有改写词时双路检索（词法用原文+改写拼接，
    向量用改写词单独编码——中文原文会占据 e5 截断窗口导致英文改写词被切掉）。
    """
    query = state["question"]
    pdf_path = str((state.get("doc") or {}).get("pdf_path") or "")
    if not state.get("need_retrieval", True):
        return {"chunks": []}
    rewrite = str(state.get("rewritten_query") or "").strip()
    try:
        if rewrite:
            chunks = await asyncio.to_thread(
                retrieval.retrieve_top_chunks_dual, pdf_path, f"{query} {rewrite}", rewrite, TOP_CANDIDATES)
        else:
            chunks = await asyncio.to_thread(retrieval.retrieve_top_chunks, pdf_path, query, TOP_CANDIDATES)
        if not chunks:  # 词法 + 向量均零命中（如中文问题对英文正文且向量不可用）：均匀采样保证全文视野
            chunks = await asyncio.to_thread(retrieval.sample_chunks, pdf_path, TOP_CANDIDATES)
    except Exception as e:
        logger.exception("检索失败，转 Agent 工具自救：%s", e)
        return {"chunks": [], "failed": True}
    cand = [{"id": c.id, "page": c.page, "text": c.text} for c in chunks]
    return {"chunks": cand}


async def agent_node(state: dict) -> dict:
    """ReAct 内核：模型层历史（model_messages）+ 笔记记忆 + 画像注入；工具事件原样转发并收集引用映射。"""
    # ---- 准备：状态字段 / 上下文与引用映射（chunks 头部）/ 事件出口 ----
    doc = state.get("doc") or {}
    conv_id = int(state.get("conversation_id") or 0)
    question = state["question"]
    selected_text = str(state.get("selected_text") or "")
    # 上下文直接从 chunks 构建（RRF 排序后的头部）
    chunks = state.get("chunks") or []
    pdf_path = str((doc or {}).get("pdf_path") or "")
    context, base_pages, cite_map = _build_context_from_chunks(chunks, pdf_path)
    writer = _writer()

    # ---- 三类背景加载（历史 / 记忆 / 画像）：各自 try 包裹，失败仅告警不阻断问答 ----
    # 模型层历史：全量加载（v0.9 缓存纪律：不做在线裁剪，前缀稳定优先）；过滤当前轮 user 条目
    history_msgs: list[dict] = []
    if conv_id:
        try:
            rows = db.list_model_messages(conv_id)
            pending_seq = state.get("pending_llm_seq")
            history_msgs = [m for m in rows if m.get("seq") != pending_seq]
        except Exception:
            logger.warning("模型层历史加载失败（跳过历史续接）", exc_info=True)

    # 该论文记忆：笔记文件（「总结整理」产物或用户手写）注入作背景；失败返回空串，不阻断问答
    memory_text = ""
    try:
        key = str(doc.get("arxiv_id") or doc.get("id") or "")
        if key:
            memory_text = await asyncio.to_thread(notes.paper_memory_text, key)
    except Exception:
        logger.warning("笔记记忆读取失败（跳过注入）", exc_info=True)

    # 用户画像：跨论文的小体量行为画像（_profile.md）作辅助背景；无文件/关闭时为空串不注入
    profile_text = ""
    try:
        profile_text = await asyncio.to_thread(notes.user_profile_text)
    except Exception:
        logger.warning("用户画像读取失败（跳过注入）", exc_info=True)

    # ---- 执行 ReAct 内核：事件原样转发（delta / tool 事件），捕获终止原因与工具页码映射 ----
    # doc 收敛为最小文档视图（循环内仅消费 id/title/pdf_path/arxiv_id，避免全行搭车传递）
    doc_view = {k: doc.get(k) for k in ("id", "title", "pdf_path", "arxiv_id")}
    used_pages: set[int] = set(base_pages)
    collected: list[str] = []
    degraded = False
    termination_reason = ""
    try:
        async for evt in agent_module.run_agent(
                doc_view, history_msgs, question, selected_text, context, used_pages,
                conversation_id=conv_id or None, memory_text=memory_text,
                profile_text=profile_text):
            if evt.get("type") == "termination":  # 内部事件：捕获终止原因，不转发前端
                termination_reason = str(evt.get("reason") or "")
                continue
            if evt.get("type") == "tool_end":  # 工具命中页码片段：引用回溯映射（工具优先覆盖检索层）
                for p, snip in (evt.get("snippets") or {}).items():
                    try:
                        cite_map[int(p)] = str(snip)
                    except (TypeError, ValueError):
                        pass
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
    # ---- 汇总：引用列表 / 终止原因（降级时附 failed 标记） ----
    citations = [{"page": p, "snippet": cite_map.get(p, "")} for p in sorted(used_pages)]
    out: dict = {"used_pages": sorted(used_pages),
                 "termination_reason": termination_reason or ("error" if degraded else "natural_stop"),
                 "citations": citations}
    if degraded:
        out["failed"] = True
        out["degraded"] = "Agent 异常，已降级直达问答"
    return out


def _build_context_from_chunks(chunks: list[dict], pdf_path: str) -> tuple[str, list[int], dict[int, str]]:
    """从 RRF 排序后的候选块直接构建上下文（取头部 KEEP_TOP 个）；产出页码集合与片段映射（引用回溯）。"""
    if not chunks or not pdf_path:
        return "", [], {}
    picked = chunks[:KEEP_TOP]
    all_chunks = documents.get_chunks(pdf_path)
    ids = [int(c["id"]) for c in picked if 0 <= int(c["id"]) < len(all_chunks)]
    text = retrieval.render_context(all_chunks, ids)
    pages = sorted({p for i in ids for p in all_chunks[i].pages})  # 跨页块展开为页码区间
    snippets: dict[int, str] = {}
    for i in ids:
        for p in all_chunks[i].pages:
            snippets.setdefault(p, all_chunks[i].text[:300])
    return text, pages, snippets


async def log_node(state: dict) -> dict:
    """图级观测：页码/降级/终止原因写入埋点（供 /api/stats 汇总）。"""
    db.record_event(
        "interaction", event_name="graph_done",
        conversation_id=state.get("conversation_id"),
        doc_id=(state.get("doc") or {}).get("id"),
        detail=(f"pages={len(state.get('used_pages') or [])} "
                f"failed={bool(state.get('failed'))} degraded={state.get('degraded') or '-'} "
                f"reason={state.get('termination_reason') or '-'}"),
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

    架构：guard→prejudge→retrieve→agent→log（5 节点线性图，RRF 排序后直接建上下文）。
    """
    g = StateGraph(QAState)
    g.add_node("guard", guard_node)
    g.add_node("prejudge", prejudge_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("agent", agent_node)
    g.add_node("log", log_node)

    g.add_edge(START, "guard")
    g.add_edge("guard", "prejudge")
    g.add_edge("prejudge", "retrieve")
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
