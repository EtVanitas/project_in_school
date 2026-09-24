"""FastAPI 服务：REST 路由 + SSE 流式问答。"""

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

from . import (__version__, config, db, documents, graph, llm, notes,
               sources)

# Windows 上补充常见前端资源的 MIME 类型
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

logger = logging.getLogger(__name__)

db.init_db()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """服务生命周期：关闭时释放图 checkpointer 连接。"""
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
async def list_docs(status: Optional[str] = None):
    """文档列表（?status=unread|read）。"""
    return db.list_documents(status)


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
    """PDF 文件流（浏览器阅读用）。"""
    doc = db.get_document(doc_id)
    if doc is None or not doc["pdf_path"]:
        raise HTTPException(status_code=404, detail="PDF 不存在")
    path = Path(doc["pdf_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF 文件缺失")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.post("/api/docs/{doc_id}/mark_read")
async def mark_read(doc_id: int):
    """标记已读。"""
    if db.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.set_document_status(doc_id, "read")
    return db.get_document(doc_id)


@app.post("/api/docs/{doc_id}/mark_unread")
async def mark_unread(doc_id: int):
    """退回未读。"""
    if db.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    db.set_document_status(doc_id, "unread")
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
async def list_messages(conv_id: int):
    """对话消息历史。"""
    if db.get_conversation(conv_id) is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return db.list_messages(conv_id)


@app.get("/api/conversations/{conv_id}/steps")
async def list_steps(conv_id: int):
    """会话工具轨迹（按 assistant 消息归属，供对话回放）。"""
    if db.get_conversation(conv_id) is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return db.list_agent_steps(conv_id)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式问答：事件 meta / agent_step / agent_observation / delta / done。
    图流程：guard→prejudge→retrieve→agent→log（5 节点线性图）。失败时降级为预检索直达问答。
    """
    doc = db.get_document(req.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    conv_id = req.conversation_id
    if conv_id is None or db.get_conversation(conv_id) is None:
        conv_id = db.create_conversation(doc["id"], req.message.strip()[:30])

    history = db.list_messages(conv_id, limit=config.CHAT_HISTORY_TURNS)  # 当前问题之前的历史（降级路径用）
    pending_msg_id = db.add_message(conv_id, "user", req.message, req.selected_text)
    db.record_event("interaction", name="question", conversation_id=conv_id, doc_id=doc["id"],
                    detail=req.message[:200])

    async def gen():
        yield _sse({"type": "meta", "conversation_id": conv_id, "doc_id": doc["id"]})
        collected: list[str] = []
        used_pages: set[int] = set()
        g = None
        cfg = None
        try:
            # 图执行（guard→prejudge→retrieve→agent→log），custom 流事件原样转发
            g = await graph.get_graph_async()
            cfg = graph.run_config(conv_id)
            state_in = graph.initial_state(doc, conv_id, req.message, req.selected_text or "",
                                           pending_msg_id)
            async for payload in g.astream(state_in, config=cfg, stream_mode="custom"):
                if payload.get("type") == "delta":
                    collected.append(str(payload.get("text") or ""))
                yield _sse(payload)
            final = (await g.aget_state(cfg)).values or {}
            used_pages = set(final.get("used_pages") or [])
        except Exception:
            logger.exception("图执行异常，降级为预检索直达问答")
            if g is not None and cfg is not None:  # 清理未完成节点，避免残留任务干扰下一轮
                try:
                    await g.aupdate_state(cfg, {"failed": True}, as_node="log")
                except Exception:
                    pass
            try:
                context, pages = await run_in_threadpool(
                    documents.search_context, doc["pdf_path"], req.message)
            except Exception:
                logger.exception("预检索失败，模型将无上下文作答")
                context, pages = "", set()
            used_pages = set(pages)
            if not collected:
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
        # 落库 + 引用校验护栏
        full = "".join(collected) or "（模型未返回内容，请稍后重试）"
        msg_id = db.add_message(conv_id, "assistant", full)
        db.link_agent_steps(conv_id, msg_id)  # 本轮工具轨迹归属到该消息（会话回放）
        db.record_event("interaction", name="answer", conversation_id=conv_id, doc_id=doc["id"],
                        detail=f"{len(full)} 字符")
        done: dict = {"type": "done", "message_id": msg_id, "conversation_id": conv_id}
        bad_pages = llm.verify_citations(full, used_pages)
        if bad_pages:
            done["warning"] = "引用校验：回答提到了未提供的页码（第 " + "、".join(map(str, bad_pages)) + " 页），请谨慎参考。"
        yield _sse(done)

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
    db.add_message(conv_id, "user", f"请解读第 {req.page} 页的图/表")
    db.record_event("interaction", name="vision_page", conversation_id=conv_id, doc_id=doc["id"],
                    detail=f"第 {req.page} 页")

    async def gen():
        yield _sse({"type": "meta", "conversation_id": conv_id, "doc_id": doc["id"]})
        collected: list[str] = []
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
        msg_id = db.add_message(conv_id, "assistant", full)
        yield _sse({"type": "done", "message_id": msg_id, "conversation_id": conv_id})

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
    """总结整理：把整段对话压缩为结构化笔记，追加保存到该论文（可编辑，作下次讨论的背景记忆）。"""
    try:
        return await notes.summarize_conversation(conv_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ===== 健康检查与前端托管 =====

@app.get("/api/health")
async def health():
    """健康检查：云端 LLM（问答/图表解读）、本地模型（对话总结）与多模态可用性。"""
    return {"status": "ok", "llm": llm.has_llm(), "local": llm.local_status(),
            "vision": llm.vision_available()}


@app.get("/api/stats")
async def stats(days: int = 7):
    """埋点统计（可观测性）：模型调用（本地/API）、token、工具、交互聚合（?days= 窗口，默认 7）。"""
    return await run_in_threadpool(db.stats_summary, max(1, min(90, days)))


@app.get("/{full_path:path}", include_in_schema=False)
async def spa(full_path: str):
    """生产模式：托管前端构建产物（SPA 回退到 index.html）。"""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="接口不存在")
    dist = config.FRONTEND_DIST
    target = dist / full_path
    if full_path and target.is_file():
        return FileResponse(target)
    index = dist / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"message": "前端未构建：开发模式请访问 http://localhost:5173，或先执行 npm run build"})
