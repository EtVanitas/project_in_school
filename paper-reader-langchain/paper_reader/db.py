"""SQLite 存储层：建表、CRUD 与事件埋点。"""

import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL DEFAULT 'arxiv',
    arxiv_id   TEXT NOT NULL DEFAULT '',
    title      TEXT NOT NULL DEFAULT '',
    authors    TEXT NOT NULL DEFAULT '',
    abstract   TEXT NOT NULL DEFAULT '',
    published  TEXT NOT NULL DEFAULT '',
    url        TEXT NOT NULL DEFAULT '',
    pdf_path   TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'unread',
    created_at TEXT NOT NULL,
    read_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_arxiv ON documents(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_doc_id ON conversations(doc_id);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT NOT NULL,              -- user / assistant
    content         TEXT NOT NULL DEFAULT '',
    selected_text   TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);

CREATE TABLE IF NOT EXISTS event_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    conversation_id INTEGER,
    doc_id          INTEGER,
    ok              INTEGER NOT NULL DEFAULT 1,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    detail          TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    message_id      INTEGER,
    step            INTEGER NOT NULL DEFAULT 0,
    tool            TEXT NOT NULL DEFAULT '',
    args_summary    TEXT NOT NULL DEFAULT '',
    ok              INTEGER NOT NULL DEFAULT 1,
    summary         TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
"""


def _now() -> str:
    """当前时间文本。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_conn() -> sqlite3.Connection:
    """懒加载单连接（WAL 模式，字典游标）。"""
    global _conn
    if _conn is None:
        config.ensure_dirs()
        _conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


def init_db() -> None:
    """建表（幂等）。"""
    with _lock:
        conn = _get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """查询并转为 dict 列表。"""
    with _lock:
        rows = _get_conn().execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _execute(sql: str, params: tuple = ()) -> int:
    """执行写操作并提交，返回 lastrowid。"""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


# ===== documents =====

def add_document(source: str, arxiv_id: str, title: str, authors: str = "", abstract: str = "",
                 published: str = "", url: str = "", pdf_path: str = "") -> int:
    """新增文档（未读状态）。"""
    return _execute(
        "INSERT INTO documents(source, arxiv_id, title, authors, abstract, published, url, pdf_path, status, created_at)"
        " VALUES(?,?,?,?,?,?,?,?, 'unread', ?)",
        (source, arxiv_id, title, authors, abstract, published, url, pdf_path, _now()),
    )


def get_document(doc_id: int) -> Optional[dict]:
    """按 id 取文档。"""
    rows = _query("SELECT * FROM documents WHERE id=?", (doc_id,))
    return rows[0] if rows else None


def find_document_by_arxiv(arxiv_id: str) -> Optional[dict]:
    """按 arXiv ID（不含版本）查重。"""
    rows = _query("SELECT * FROM documents WHERE arxiv_id=?", (arxiv_id,))
    return rows[0] if rows else None


def _norm_title(title: str) -> str:
    """标题归一（小写、统一空白），用于标题级查重。"""
    return re.sub(r"\s+", " ", (title or "").lower()).strip()


def find_document_by_title(title: str) -> Optional[dict]:
    """按归一化标题查重（忽略大小写/空白）。"""
    key = _norm_title(title)
    if not key:
        return None
    for row in _query("SELECT * FROM documents"):
        if _norm_title(row["title"]) == key:
            return row
    return None


def list_documents(status: Optional[str] = None) -> list[dict]:
    """按状态列出文档（新→旧）。"""
    if status:
        return _query("SELECT * FROM documents WHERE status=? ORDER BY id DESC", (status,))
    return _query("SELECT * FROM documents ORDER BY id DESC")


def list_catalog() -> list[dict]:
    """目录卡：全部文档 + 对话数（笔记数由 api 层按笔记文件统计）。"""
    return _query(
        "SELECT d.*, COUNT(c.id) AS conversation_count"
        " FROM documents d LEFT JOIN conversations c ON c.doc_id=d.id"
        " GROUP BY d.id ORDER BY d.id DESC"
    )


def set_document_status(doc_id: int, status: str) -> None:
    """更新阅读状态：read 记录时间，unread 清空。"""
    if status == "read":
        _execute("UPDATE documents SET status='read', read_at=? WHERE id=?", (_now(), doc_id))
    else:
        _execute("UPDATE documents SET status='unread', read_at=NULL WHERE id=?", (doc_id,))


def delete_document(doc_id: int) -> Optional[dict]:
    """删除文档及其对话/消息/工具轨迹/PDF 文件。"""
    doc = get_document(doc_id)
    if doc is None:
        return None
    with _lock:
        conn = _get_conn()
        conv_ids = [r["id"] for r in conn.execute("SELECT id FROM conversations WHERE doc_id=?", (doc_id,)).fetchall()]
        for cid in conv_ids:
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
            conn.execute("DELETE FROM agent_steps WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM conversations WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.commit()
    try:
        pdf_path = doc.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            Path(pdf_path).unlink()
    except Exception as e:
        logger.warning("删除 PDF 失败：%s", e)
    return doc


# ===== conversations / messages =====

def create_conversation(doc_id: int, title: str) -> int:
    """新建对话。"""
    return _execute("INSERT INTO conversations(doc_id, title, created_at) VALUES(?,?,?)", (doc_id, title, _now()))


def get_conversation(conv_id: int) -> Optional[dict]:
    """按 id 取对话（附文档标题）。"""
    rows = _query(
        "SELECT c.*, d.title AS doc_title FROM conversations c JOIN documents d ON d.id=c.doc_id WHERE c.id=?",
        (conv_id,),
    )
    return rows[0] if rows else None


def list_conversations(doc_id: Optional[int] = None) -> list[dict]:
    """对话列表（新→旧，附文档标题）。"""
    if doc_id:
        return _query(
            "SELECT c.*, d.title AS doc_title FROM conversations c JOIN documents d ON d.id=c.doc_id"
            " WHERE c.doc_id=? ORDER BY c.id DESC", (doc_id,))
    return _query(
        "SELECT c.*, d.title AS doc_title FROM conversations c JOIN documents d ON d.id=c.doc_id ORDER BY c.id DESC")


def add_message(conversation_id: int, role: str, content: str, selected_text: str = "") -> int:
    """写入一条消息。"""
    return _execute(
        "INSERT INTO messages(conversation_id, role, content, selected_text, created_at) VALUES(?,?,?,?,?)",
        (conversation_id, role, content, selected_text, _now()),
    )


def list_messages(conversation_id: int, limit: int = 0) -> list[dict]:
    """对话消息（旧→新）；limit>0 时取最近 limit 条。"""
    if limit > 0:
        rows = _query(
            "SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?) ORDER BY id",
            (conversation_id, limit))
        return rows
    return _query("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,))


# ===== 事件埋点与统计 =====

def record_event(kind: str, name: str = "", conversation_id: Optional[int] = None,
                 doc_id: Optional[int] = None, ok: bool = True, latency_ms: int = 0,
                 tokens_in: int = 0, tokens_out: int = 0, detail: str = "") -> None:
    """写一条事件（失败静默，热路径安全）。"""
    try:
        _execute(
            "INSERT INTO event_log(kind, name, conversation_id, doc_id, ok, latency_ms, tokens_in, tokens_out, detail, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (kind, name, conversation_id, doc_id, 1 if ok else 0, int(latency_ms),
             int(tokens_in), int(tokens_out), detail[:1000], _now()),
        )
    except Exception as e:
        logger.debug("事件写入失败（已忽略）: %s", e)


def ms_since(started: float) -> int:
    """开始时间（time.monotonic()）到现在的毫秒数（埋点延迟计算）。"""
    return int((time.monotonic() - started) * 1000)


def events_summary(since: str) -> list[dict]:
    """按 kind 聚合事件：次数 / 成功数 / 平均延迟 / token 合计。"""
    return _query(
        "SELECT kind, COUNT(*) AS n, SUM(ok) AS ok_n, AVG(latency_ms) AS avg_ms,"
        " SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out"
        " FROM event_log WHERE created_at >= ? GROUP BY kind ORDER BY n DESC",
        (since,),
    )


def events_by_name(kind: str, since: str, limit: int = 20) -> list[dict]:
    """按 name 聚合某一类事件。"""
    return _query(
        "SELECT name, COUNT(*) AS n, SUM(ok) AS ok_n, AVG(latency_ms) AS avg_ms"
        " FROM event_log WHERE kind=? AND created_at >= ? GROUP BY name ORDER BY n DESC LIMIT ?",
        (kind, since, limit),
    )


def _rate(ok_n: int, n: int) -> float:
    """成功率（0-1，保留 3 位）；无样本返回 0。"""
    return round(ok_n / n, 3) if n else 0.0


def _with_rate(rows: list[dict]) -> list[dict]:
    """给按 name 聚合的结果补充成功率字段。"""
    for r in rows:
        r["rate"] = _rate(int(r["ok_n"] or 0), int(r["n"] or 0))
    return rows


def stats_summary(days: int = 7) -> dict:
    """聚合统计：totals + by_kind + 工具/本地/API 细分（供 GET /api/stats）。"""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    by_kind: dict = {}
    totals = {"events": 0, "api_tokens_in": 0, "api_tokens_out": 0, "local_calls": 0,
              "tool_calls": 0, "tool_rate": 0.0, "interactions": 0}
    for r in events_summary(since):
        kind = r["kind"]
        n = int(r["n"] or 0)
        by_kind[kind] = {
            "count": n,
            "ok": int(r["ok_n"] or 0),
            "rate": _rate(int(r["ok_n"] or 0), n),
            "avg_latency_ms": round(float(r["avg_ms"] or 0)),
            "tokens_in": int(r["tokens_in"] or 0),
            "tokens_out": int(r["tokens_out"] or 0),
        }
        totals["events"] += n
        if kind == "api_llm":
            totals["api_tokens_in"] += by_kind[kind]["tokens_in"]
            totals["api_tokens_out"] += by_kind[kind]["tokens_out"]
        elif kind == "local_llm":
            totals["local_calls"] = n
        elif kind == "tool":
            totals["tool_calls"] = n
            totals["tool_rate"] = by_kind[kind]["rate"]
        elif kind == "interaction":
            totals["interactions"] = n
    return {
        "window_days": days,
        "since": since,
        "totals": totals,
        "by_kind": by_kind,
        "tools": _with_rate(events_by_name("tool", since)),
        "local_tasks": _with_rate(events_by_name("local_llm", since)),
        "api_tasks": _with_rate(events_by_name("api_llm", since)),
    }


def add_agent_step(conversation_id: int, step: int, tool: str, args_summary: str,
                   ok: bool, summary: str) -> int:
    """记录一次 Agent 工具步骤。"""
    return _execute(
        "INSERT INTO agent_steps(conversation_id, step, tool, args_summary, ok, summary, created_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (conversation_id, step, tool, args_summary, 1 if ok else 0, summary, _now()),
    )


def link_agent_steps(conversation_id: int, message_id: int) -> None:
    """关联工具轨迹到消息。"""
    _execute("UPDATE agent_steps SET message_id=? WHERE conversation_id=? AND message_id IS NULL",
             (message_id, conversation_id))


def list_agent_steps(conversation_id: int) -> list[dict]:
    """获取会话的工具轨迹。"""
    return _query("SELECT * FROM agent_steps WHERE conversation_id=? ORDER BY id", (conversation_id,))
