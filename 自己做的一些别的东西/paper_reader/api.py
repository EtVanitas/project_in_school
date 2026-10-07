"""FastAPI 服务：REST 路由 + SSE 流式问答。"""

import asyncio
import json
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__, config, db, graph
from .docs import notes, retrieval, sources
from .llms import llm, local_llm, tools

# Windows 上补充常见前端资源的 MIME 类型
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

logger = logging.getLogger(__name__)

db.init_db()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """服务生命周期：启动时登记 docs 目录已有 PDF（库被清空后可恢复），关闭时释放图 checkpointer 连接。"""
    for p in tools.self_check():  # 工具声明自检（失败仅记录，不阻断启动）
        logger.error("工具声明自检未通过：%s", p)
    try:
        sources.rescan_documents_dir()
    except Exception as e:  # 启动登记失败不阻断服务（文档库可为空，靠下载/发现补充）
        logger.warning("启动扫描 docs 登记失败（已忽略）：%s", e)
    yield
    await graph.aclose()


app = FastAPI(title="智能论文阅读助手", version=__version__, lifespan=lifespan)

# 开发模式：Vite dev server 跨端口访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 请求模型 =====

class DownloadItem(BaseModel):
    source: str = "arxiv"
    arxiv_id: str = ""
    title: str = ""
    authors: str = ""
    abstract: str = ""
    published: str = ""
    url: str = ""


class DownloadRequest(BaseModel):
    items: list[DownloadItem] = []
    arxiv_id: str = ""


class ChatRequest(BaseModel):
    doc_id: int
    message: str = Field(min_length=1)
    selected_text: str = ""
    conversation_id: Optional[int] = None


class NoteUpdateRequest(BaseModel):
    content: str = Field(min_length=1)


class VisionPageRequest(BaseModel):
    doc_id: int
    page: int = Field(ge=1)
    conversation_id: Optional[int] = None


class DiscoverRequest(BaseModel):
    source: str = "hf_papers"
    period: str = "daily"
    query: str = ""
    limit: int = 20


def _sse(payload: dict) -> str:
    """SSE 数据帧（data: JSON\n\n格式）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _persist_step(payload: dict, starts: dict, conv_id: int, msg_id: int) -> None:
    """tool_end 事件到位即写 tool_steps（身份链：call_id / message_id 随事件即达，失败忽略）。

    显式字段映射（事件 → 落库列；新增事件字段须在此登记去向，防漏字段）：
      tool_end.name→tool_name / ok→ok / result_summary→result_summary；
      tool_start.args_summary→args_summary、args→args_full（JSON 截 500）；call_id、msg_id→身份链列。
      有意不落库：latency_ms / cached / pages / snippets（实时呈现专用，历史回放不依赖）。
    """
    call_id = str(payload.get("call_id") or "")
    start = starts.pop(call_id, {})
    try:
        db.add_tool_step(
            conv_id, str(payload.get("name") or ""),
            str(start.get("args_summary") or ""), bool(payload.get("ok")),
            str(payload.get("result_summary") or ""), message_id=msg_id, call_id=call_id,
            args_full=json.dumps(start.get("args") or {}, ensure_ascii=False)[:500],
        )
    except Exception:
        logger.warning("工具轨迹落库失败（忽略）", exc_info=True)


# ===== 自动整理触发（v0.9 三期：单线 100k，轮末后台 + 轮前同步兜底） =====

_bg_tasks: set = set()  # 后台整理任务持引用（防 GC）


def _auto_organize_due(conv_id: int) -> bool:
    """达线判定：总开关开启 且 last_input_tokens ≥ 触发线（读取失败→False，不影响主流程）。"""
    if not config.ORG_AUTO_ENABLED:
        return False
    try:
        conv = db.get_conversation(conv_id)
    except Exception:
        logger.warning("自动整理达线判定失败（忽略）", exc_info=True)
        return False
    return bool(conv) and int(conv.get("last_input_tokens") or 0) >= config.ORG_TRIGGER_TOKENS


async def _organize_sync(conv_id: int, exclude_seq: int = 0) -> None:
    """轮前同步兜底：等待整理完成（含排队等轮末后台任务收尾）；失败仅记录，不阻断问答。"""
    try:
        await notes.organize_conversation(conv_id, exclude_seq=exclude_seq)
    except Exception:
        logger.warning("轮前同步整理失败（忽略，下轮再试）", exc_info=True)


async def _run_organize_bg(conv_id: int) -> None:
    """轮末后台整理（失败仅记录，下轮再试）。"""
    try:
        await notes.organize_conversation(conv_id)
    except Exception:
        logger.warning("后台整理失败（忽略，下轮再试）", exc_info=True)


def _maybe_organize_async(conv_id: int) -> None:
    """轮末触发：达线则建立后台整理任务（不阻塞本轮响应；幂等/防抖由 notes 串行锁与水位保证）。"""
    if not _auto_organize_due(conv_id):
        return
    task = asyncio.create_task(_run_organize_bg(conv_id))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# ===== 发现与下载 =====

@app.post("/api/discover")
async def discover(req: DiscoverRequest | None = None):
    """获取候选：hf_papers（daily/weekly/monthly 榜单 + 个性化重排）或 arxiv（关键词检索）。"""
    r = req or DiscoverRequest()
    try:
        if r.source == "arxiv":
            if not r.query.strip():
                raise HTTPException(status_code=400, detail="请输入搜索关键词")
            items = await run_in_threadpool(sources.arxiv_search, r.query.strip(), max(1, r.limit))
            return {"items": items, "stale": False, "cached_at": "", "period": ""}
        data = await run_in_threadpool(sources.fetch_hf_papers, r.period)
    except HTTPException:
        raise
    except Exception as e:  # 抓取/检索失败：透出原因（含重试信息），供前端提示
        logger.warning("发现源获取失败：%s", e)
        raise HTTPException(status_code=502, detail=f"获取候选失败：{e}")
    data["items"] = sources.rank_personalized(data["items"])[:max(1, r.limit)]
    return data


@app.post("/api/download")
async def download(req: DownloadRequest):
    """下载候选论文或指定 arXiv ID 入库（未读区）。"""
    items = [it.model_dump() for it in req.items]
    if req.arxiv_id.strip():
        items.append({"source": "arxiv", "arxiv_id": req.arxiv_id.strip()})
    if not items:
        raise HTTPException(status_code=400, detail="没有要下载的论文")

    def _run() -> dict:
        result: dict = {"downloaded": [], "skipped": [], "errors": []}
        for it in items:
            try:
                r = sources.download_into_library(it)
                result["downloaded" if r["status"] == "downloaded" else "skipped"].append(r["message"])
            except Exception as e:  # 单篇失败不影响其余
                label = it.get("arxiv_id") or it.get("title") or "未知论文"
                result["errors"].append(f"{label}：{e}")
        return result

    return await run_in_threadpool(_run)


# ===== 文档库 =====

@app.get("/api/docs")
async def list_docs(read_status: Optional[str] = None):
    """文档列表（?read_status=unread|read）。"""
    return db.list_documents(read_status)


@app.get("/api/catalog")
async def catalog():
    """目录卡：全部文档 + 摘要 + 笔记数（笔记文件按 arxiv_id 统计，数据库 notes 表已废弃）。"""
    rows = db.list_catalog()
    counts: dict[str, int] = {}
    try:
        for n in notes.list_notes():
            key = str(n.get("arxiv_id") or "")
            counts[key] = counts.get(key, 0) + 1
    except Exception as e:  # 笔记目录异常不影响目录视图
        logger.warning("统计笔记数失败：%s", e)
    for r in rows:
        r["note_count"] = counts.get(str(r.get("arxiv_id") or ""), 0)
    return rows


@app.get("/api/docs/{doc_id}")
async def get_doc(doc_id: int):
    """文档详情。"""
    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


@app.get("/api/docs/{doc_id}/file")
async def get_doc_file(doc_id: int):
    """PDF 文件流（浏览器阅读用）。

    no-cache 强制浏览器每次条件验证：默认无 Cache-Control 会触发启发式缓存，
    文档 id 复用/重排后浏览器可能长期沿用旧 id 的 PDF 内容（实测踩坑）。
    """
    doc = db.get_document(doc_id)
    if doc is None or not doc["pdf_path"]:
        raise HTTPException(status_code=404, detail="PDF 不存在")
    path = Path(doc["pdf_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF 文件缺失")
    return FileResponse(path, media_type="application/pdf", filename=path.name,
                        headers={"Cache-Control": "no-cache"})


@app.post("/api/docs/{doc_id}/mark_read")
async def mark_read(doc_id: int):
    """标记已读。"""
    if db.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.set_read_status(doc_id, "read")
    return db.get_document(doc_id)


@app.post("/api/docs/{doc_id}/mark_unread")
async def mark_unread(doc_id: int):
    """退回未读。"""
    if db.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.set_read_status(doc_id, "unread")
    return db.get_document(doc_id)


@app.delete("/api/docs/{doc_id}")
async def remove_doc(doc_id: int):
    """删除文档及其对话记录（PDF 由 db 层删除）。"""
    removed = db.delete_document(doc_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"message": f"已删除《{removed['title']}》"}


# ===== 对话 =====

@app.get("/api/conversations")
async def list_conversations(doc_id: Optional[int] = None):
    """对话列表（可 ?doc_id= 过滤）。"""
    return db.list_conversations(doc_id)


@app.get("/api/conversations/{conv_id}/messages")
async def list_chat_messages(conv_id: int):
    """对话消息历史。"""
    if db.get_conversation(conv_id) is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return db.list_chat_messages(conv_id)


@app.get("/api/conversations/{conv_id}/steps")
async def list_steps(conv_id: int):
    """会话工具轨迹（按 assistant 消息归属，供对话回放）。"""
    if db.get_conversation(conv_id) is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return db.list_tool_steps(conv_id)


@app.get("/api/conversations/{conv_id}/memory")
async def conversation_memory(conv_id: int):
    """整理状态：token 记账 / 触发线 / 整理水位 / 是否在整理（前端整理提示与按钮用）。"""
    try:
        return notes.organize_status(conv_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式问答：事件 meta / tool_start / tool_end / delta / retract / note_issue / done。
    图流程：guard→prejudge→retrieve→agent→log（5 节点线性图）。失败时降级为预检索直达问答。
    双层对话：显示层（chat_messages+tool_steps）完整账本；模型层（model_messages）供下轮续接、由整理统一消化。
    """
    doc = db.get_document(req.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    conv_id = req.conversation_id
    if conv_id is None or db.get_conversation(conv_id) is None:
        conv_id = db.create_conversation(doc["id"], req.message.strip()[:30])

    # 身份链：显示层 user 消息 + 模型层 user 条目（请求进入即写）；assistant 尽早建 streaming 空行
    db.add_chat_message(conv_id, "user", req.message, req.selected_text)
    pending_llm_seq = 0
    try:
        pending_llm_seq = db.append_model_message(conv_id, "user", req.message)
    except Exception:
        logger.warning("模型层 user 条目写入失败（忽略）", exc_info=True)
    msg_id = db.add_chat_message(conv_id, "assistant", "", status="streaming")
    db.record_event("interaction", event_name="question", conversation_id=conv_id, doc_id=doc["id"],
                    detail=req.message[:200])

    async def gen():
        if _auto_organize_due(conv_id):  # 轮前同步兜底（G1 显式例外）：轮末后台未完成/已失败时补刀
            yield _sse({"type": "organizing", "conversation_id": conv_id})
            await _organize_sync(conv_id, exclude_seq=pending_llm_seq)
        yield _sse({"type": "meta", "conversation_id": conv_id, "doc_id": doc["id"],
                    "message_id": msg_id})
        # ---- 本轮收集状态：输出全文 / 页码引用 / 工具配对 / 终止原因 / 定稿标记 ----
        collected: list[str] = []
        used_pages: set[int] = set()
        citations: list[dict] = []
        termination_reason = ""
        starts: dict[str, dict] = {}  # call_id → tool_start 载荷（tool_end 配对落库）
        finalized = False
        g = None
        cfg = None
        try:
            try:
                # 图执行（guard→prejudge→retrieve→agent→log），custom 流事件原样转发
                g = await graph.get_graph_async()
                cfg = graph.run_config(conv_id)
                state_in = graph.initial_state(doc, conv_id, req.message, req.selected_text or "",
                                               pending_llm_seq)
                async for payload in g.astream(state_in, config=cfg, stream_mode="custom"):
                    etype = payload.get("type")
                    if etype == "delta":
                        collected.append(str(payload.get("text") or ""))
                    elif etype == "tool_start":
                        starts[str(payload.get("call_id") or "")] = payload
                    elif etype == "tool_end":
                        _persist_step(payload, starts, conv_id, msg_id)
                    yield _sse(payload)
                final = (await g.aget_state(cfg)).values or {}
                used_pages = set(final.get("used_pages") or [])
                citations = list(final.get("citations") or [])
                termination_reason = str(final.get("termination_reason") or "")
            except Exception:
                logger.exception("图执行异常，降级为预检索直达问答")
                if g is not None and cfg is not None:  # 清理未完成节点，避免残留任务干扰下一轮
                    try:
                        await g.aupdate_state(cfg, {"failed": True}, as_node="log")
                    except Exception:
                        pass
                try:
                    context, pages = await run_in_threadpool(
                        retrieval.search_context, doc["pdf_path"], req.message)
                except Exception:
                    logger.exception("预检索失败，模型将无上下文作答")
                    context, pages = "", set()
                used_pages = set(pages)
                if not collected:
                    try:  # 模型层历史（当前轮 user 条目除外）；answer 形态自动跳过工具交互细节
                        history = [m for m in db.list_model_messages(conv_id)
                                   if m.get("seq") != pending_llm_seq]
                    except Exception:
                        history = []
                    try:
                        async for delta in llm.stream_answer(doc["title"], context, history, req.message,
                                                             req.selected_text, conversation_id=conv_id):
                            collected.append(delta)
                            yield _sse({"type": "delta", "text": delta})
                    except Exception as e2:
                        logger.exception("降级问答失败")
                        text = f"\n\n（调用模型失败：{e2}，请稍后重试）"
                        collected.append(text)
                        yield _sse({"type": "delta", "text": text})
                if not termination_reason:
                    termination_reason = "error"
            # ---- 定稿：引用校验护栏 + done 一次 UPDATE 写全文/终态（身份链收口）→ 轮末后台整理 ----
            full = "".join(collected) or "（模型未返回内容，请稍后重试）"
            bad_pages = llm.verify_citations(full, used_pages)
            warning = ""
            if bad_pages:
                warning = "引用校验：回答提到了未提供的页码（第 " + "、".join(map(str, bad_pages)) + " 页），请谨慎参考。"
            db.update_chat_message(msg_id, full, status="completed")
            try:  # 模型层：assistant 最终回复条目（保持轮次配对，供下轮续接）
                db.append_model_message(conv_id, "assistant", full)
            except Exception:
                logger.warning("模型层回复条目写入失败（忽略）", exc_info=True)
            db.record_event("interaction", event_name="answer", conversation_id=conv_id, doc_id=doc["id"],
                            detail=f"{len(full)} 字符")
            finalized = True
            # warning / termination_reason 仅随 done 实时下发不落库（历史回看无消费方）
            done: dict = {"type": "done", "message_id": msg_id, "conversation_id": conv_id,
                          "termination_reason": termination_reason or "natural_stop",
                          "citations": citations}
            if warning:
                done["warning"] = warning
            yield _sse(done)
            _maybe_organize_async(conv_id)  # 轮末后台触发（不阻塞响应；失败下轮再试）
        finally:
            if not finalized:  # 断连/取消：partial 定稿（保留已流出内容，供历史回看）
                try:
                    db.update_chat_message(msg_id, "".join(collected) or "（回答中断）", status="partial")
                except Exception:
                    logger.warning("partial 落库失败（忽略）", exc_info=True)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ===== 多模态图表解读 =====

@app.post("/api/vision_page")
async def vision_page(req: VisionPageRequest):
    """SSE：解读指定页的图/表/公式（页面渲染 → DeepSeek vision 流式）；事件 meta/delta/done。"""
    doc = db.get_document(req.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    conv_id = req.conversation_id
    if conv_id is None or db.get_conversation(conv_id) is None:
        conv_id = db.create_conversation(doc["id"], f"解读第 {req.page} 页图表")
    user_text = f"请解读第 {req.page} 页的图/表"
    db.add_chat_message(conv_id, "user", user_text)
    try:
        db.append_model_message(conv_id, "user", user_text)
    except Exception:
        logger.warning("模型层 user 条目写入失败（忽略）", exc_info=True)
    msg_id = db.add_chat_message(conv_id, "assistant", "", status="streaming")
    db.record_event("interaction", event_name="vision_page", conversation_id=conv_id, doc_id=doc["id"],
                    detail=f"第 {req.page} 页")

    async def gen():
        yield _sse({"type": "meta", "conversation_id": conv_id, "doc_id": doc["id"],
                    "message_id": msg_id})
        collected: list[str] = []
        finalized = False
        try:
            try:
                async for piece in llm.describe_page_stream(
                        str(doc.get("pdf_path") or ""), req.page, conversation_id=conv_id):
                    collected.append(piece)
                    yield _sse({"type": "delta", "text": piece})
            except Exception as e:  # 渲染/模型失败：转提示文本，不中断 SSE
                logger.warning("图表解读失败: %s", e)
                text = f"（图表解读失败：{e}）"
                collected.append(text)
                yield _sse({"type": "delta", "text": text})
            full = "".join(collected) or "（模型未返回内容，请稍后重试）"
            db.update_chat_message(msg_id, full)
            try:  # 模型层：assistant 条目（视觉轮也入账，保持两层一致）
                db.append_model_message(conv_id, "assistant", full)
            except Exception:
                logger.warning("模型层回复条目写入失败（忽略）", exc_info=True)
            finalized = True
            yield _sse({"type": "done", "message_id": msg_id, "conversation_id": conv_id})
            _maybe_organize_async(conv_id)  # 轮末后台触发（视觉轮同构）
        finally:
            if not finalized:  # 断连：partial 定稿
                try:
                    db.update_chat_message(msg_id, "".join(collected) or "（回答中断）", status="partial")
                except Exception:
                    logger.warning("partial 落库失败（忽略）", exc_info=True)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ===== 笔记（单文件存储） =====

@app.get("/api/notes")
async def list_notes(doc_id: Optional[int] = None):
    """笔记列表（?doc_id= 按文档过滤；笔记以 arxiv_id 为文件名键）。"""
    if doc_id is None:
        return notes.list_notes()
    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    key = str(doc.get("arxiv_id") or doc_id)
    note = notes.get_note(key)
    return [note] if note else []


@app.get("/api/notes/{arxiv_id}")
async def get_note_by_arxiv(arxiv_id: str):
    """获取指定 arxiv_id 的笔记详情。"""
    note = notes.get_note(arxiv_id)
    if note is None:
        raise HTTPException(status_code=404, detail="笔记不存在")
    return note


@app.post("/api/notes/{arxiv_id}")
async def create_or_update_note(arxiv_id: str, req: NoteUpdateRequest):
    """创建或更新指定 arxiv_id 的笔记（文件不存在时自动创建，标题优先取库内文档）。"""
    if not arxiv_id.strip():
        raise HTTPException(status_code=400, detail="arxiv_id 不能为空")
    try:
        if notes.get_note(arxiv_id) is None:
            norm = sources.normalize_arxiv_id(arxiv_id) or arxiv_id
            doc = db.find_document_by_arxiv(norm)
            title = (doc or {}).get("title") or norm
            notes.add_note(arxiv_id, title, req.content)
        else:
            notes.update_note(arxiv_id, req.content)
        note = notes.get_note(arxiv_id)
        if note is None:
            raise HTTPException(status_code=500, detail="笔记写入失败")
        return note
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/notes/{arxiv_id}")
async def delete_note_by_arxiv(arxiv_id: str):
    """删除指定 arxiv_id 的笔记。"""
    success = notes.delete_note(arxiv_id)
    if not success:
        raise HTTPException(status_code=404, detail="笔记不存在")
    return {"message": "笔记已删除"}


@app.post("/api/conversations/{conv_id}/summarize")
async def summarize_conversation(conv_id: int):
    """统一整理（手动）：增量 → Q/A 追加 `## 对话 <id>` 小节 + 画像更新 + 清账；
    云端为主（吃前缀缓存），失败转本地 4B 保底，双失败 502（状态不动）。"""
    try:
        return await notes.organize_conversation(conv_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ===== 健康检查与前端托管 =====

@app.get("/api/health")
async def health():
    """健康检查：云端 LLM（问答/图表解读）、本地模型（整理保底/画像）与多模态可用性。"""
    return {"status": "ok", "llm": llm.has_llm(), "local": local_llm.local_status(),
            "vision": llm.vision_available()}


@app.get("/api/stats")
async def stats(days: int = 7):
    """埋点统计（可观测性）：模型调用（本地/API）、token、工具、交互聚合（?days= 窗口，默认 7）。"""
    return await run_in_threadpool(db.stats_summary, max(1, min(90, days)))


@app.get("/{full_path:path}", include_in_schema=False)
async def spa(full_path: str):
    """生产模式：托管前端构建产物（SPA 回退到 index.html；no-cache 防旧构建缓存滞留）。"""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="接口不存在")
    dist = config.FRONTEND_DIST
    target = dist / full_path
    if full_path and target.is_file():
        return FileResponse(target, headers={"Cache-Control": "no-cache"})
    index = dist / "index.html"
    if index.exists():
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    return JSONResponse({"message": "前端未构建：开发模式请访问 http://localhost:5173，或先执行 npm run build"})
