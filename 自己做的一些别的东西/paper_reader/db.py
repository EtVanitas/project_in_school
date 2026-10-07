"""SQLite 存储层：建表、CRUD 与事件埋点。"""

import json
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
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL DEFAULT 'arxiv',
    arxiv_id     TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    authors      TEXT NOT NULL DEFAULT '',
    abstract     TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    pdf_path     TEXT NOT NULL DEFAULT '',
    version      INTEGER NOT NULL DEFAULT 0,
    pdf_hash     TEXT NOT NULL DEFAULT '',
    read_status  TEXT NOT NULL DEFAULT 'unread',        -- read / unread
    added_at     TEXT NOT NULL,                         -- 入库时间
    read_at      TEXT NOT NULL DEFAULT ''               -- 最近一次标记已读时间（'' = 从未标记）
);
CREATE INDEX IF NOT EXISTS idx_documents_arxiv ON documents(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_documents_read_status ON documents(read_status);

CREATE TABLE IF NOT EXISTS conversations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id            INTEGER NOT NULL,
    title             TEXT NOT NULL DEFAULT '',
    organized_seq     INTEGER NOT NULL DEFAULT 0,       -- 整理水位：模型层已整理的 model_messages.seq（只前推）
    last_input_tokens INTEGER NOT NULL DEFAULT 0,       -- 最近一次 LLM 调用 input_tokens（触发判定用）
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_doc_id ON conversations(doc_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT NOT NULL,              -- user / assistant
    content         TEXT NOT NULL DEFAULT '',
    selected_text   TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'completed',  -- streaming / completed / partial
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conv ON chat_messages(conversation_id);

CREATE TABLE IF NOT EXISTS event_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL,              -- interaction / api_llm / local_llm / tool
    event_name      TEXT NOT NULL DEFAULT '',
    conversation_id INTEGER,
    doc_id          INTEGER,
    ok              INTEGER NOT NULL DEFAULT 1,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    detail          TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    message_id      INTEGER,
    call_id         TEXT NOT NULL DEFAULT '',   -- 模型 tool_call.id（身份链）
    tool_name       TEXT NOT NULL DEFAULT '',
    args_summary    TEXT NOT NULL DEFAULT '',
    args_full       TEXT NOT NULL DEFAULT '',   -- 完整参数 JSON（排查/回放）
    ok              INTEGER NOT NULL DEFAULT 1,
    result_summary  TEXT NOT NULL DEFAULT '',   -- 结果摘要（回放展示）
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    seq             INTEGER NOT NULL,                   -- 组装顺序（对话内单调）
    role            TEXT NOT NULL,                      -- user / assistant / tool
    content         TEXT NOT NULL DEFAULT '',
    tool_calls      TEXT NOT NULL DEFAULT '',           -- assistant: [{"name","args","id"}] JSON
    tool_call_id    TEXT NOT NULL DEFAULT ''            -- tool：与调用配对
);
CREATE INDEX IF NOT EXISTS idx_model_messages_conv ON model_messages(conversation_id, seq);
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
                 published_at: str = "", pdf_path: str = "",
                 version: int = 0, pdf_hash: str = "") -> int:
    """新增文档（未读状态；version=0 表示版本未知，pdf_hash 为入库时文件 sha256）。"""
    return _execute(
        "INSERT INTO documents(source, arxiv_id, title, authors, abstract, published_at, pdf_path,"
        " version, pdf_hash, read_status, added_at) VALUES(?,?,?,?,?,?,?,?,?, 'unread', ?)",
        (source, arxiv_id, title, authors, abstract, published_at, pdf_path,
         int(version or 0), pdf_hash or "", _now()),
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


def list_documents(read_status: Optional[str] = None) -> list[dict]:
    """按阅读状态列出文档（新→旧）。"""
    if read_status:
        return _query("SELECT * FROM documents WHERE read_status=? ORDER BY id DESC", (read_status,))
    return _query("SELECT * FROM documents ORDER BY id DESC")


def list_catalog() -> list[dict]:
    """目录卡：全部文档 + 对话数（笔记数由 api 层按笔记文件统计）。"""
    return _query(
        "SELECT d.*, COUNT(c.id) AS conversation_count"
        " FROM documents d LEFT JOIN conversations c ON c.doc_id=d.id"
        " GROUP BY d.id ORDER BY d.id DESC"
    )


def set_read_status(doc_id: int, status: str) -> None:
    """更新阅读状态（read / unread）；标记已读时刷新 read_at（最近一次标记已读时间）。"""
    if status == "read":
        _execute("UPDATE documents SET read_status='read', read_at=? WHERE id=?", (_now(), doc_id))
    else:
        _execute("UPDATE documents SET read_status='unread' WHERE id=?", (doc_id,))


def delete_document(doc_id: int) -> Optional[dict]:
    """删除文档及其对话/聊天记录/工具轨迹/模型层消息/PDF 文件。"""
    doc = get_document(doc_id)
    if doc is None:
        return None
    with _lock:
        conn = _get_conn()
        conv_ids = [r["id"] for r in conn.execute("SELECT id FROM conversations WHERE doc_id=?", (doc_id,)).fetchall()]
        for cid in conv_ids:
            conn.execute("DELETE FROM chat_messages WHERE conversation_id=?", (cid,))
            conn.execute("DELETE FROM tool_steps WHERE conversation_id=?", (cid,))
            conn.execute("DELETE FROM model_messages WHERE conversation_id=?", (cid,))
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


# ===== conversations / chat_messages =====

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


def add_chat_message(conversation_id: int, role: str, content: str, selected_text: str = "",
                     status: str = "completed") -> int:
    """写入一条聊天记录（assistant 可先建 streaming 空行，done 时 update_chat_message 定稿）。"""
    return _execute(
        "INSERT INTO chat_messages(conversation_id, role, content, selected_text, status, created_at)"
        " VALUES(?,?,?,?,?,?)",
        (conversation_id, role, content, selected_text, status, _now()),
    )


def update_chat_message(msg_id: int, content: str, status: str = "completed") -> None:
    """回答定稿 / partial 落库：一次 UPDATE 写全文与终态（尽早建行的空行在此完结）。"""
    _execute("UPDATE chat_messages SET content=?, status=? WHERE id=?",
             (content, status, msg_id))


def list_chat_messages(conversation_id: int) -> list[dict]:
    """对话消息（旧→新）。"""
    return _query("SELECT * FROM chat_messages WHERE conversation_id=? ORDER BY id", (conversation_id,))


def set_organized_seq(conv_id: int, seq: int) -> None:
    """推进整理水位到 seq（幂等：只向前推，不回退）。"""
    _execute("UPDATE conversations SET organized_seq=? WHERE id=? AND organized_seq<?",
             (int(seq), conv_id, int(seq)))


def set_last_input_tokens(conv_id: int, tokens: int) -> None:
    """直接覆盖 last_input_tokens（整理后归零用；轮末记账走 note_input_tokens）。"""
    _execute("UPDATE conversations SET last_input_tokens=? WHERE id=?", (int(tokens), conv_id))


def note_input_tokens(conv_id: int | None, tokens: int) -> None:
    """轮末 token 记账（自动整理触发判定）：会话/数值无效时跳过，写失败静默（热路径安全）。"""
    if not conv_id or tokens <= 0:
        return
    try:
        _execute("UPDATE conversations SET last_input_tokens=? WHERE id=?", (int(tokens), int(conv_id)))
    except Exception as e:
        logger.warning("last_input_tokens 记账失败（忽略）: %s", e)


# ===== 事件埋点与统计 =====

def record_event(event_type: str, event_name: str = "", conversation_id: Optional[int] = None,
                 doc_id: Optional[int] = None, ok: bool = True, latency_ms: int = 0,
                 tokens_in: int = 0, tokens_out: int = 0, detail: str = "") -> None:
    """写一条事件（失败静默，热路径安全）。"""
    try:
        _execute(
            "INSERT INTO event_log(event_type, event_name, conversation_id, doc_id, ok, latency_ms, tokens_in, tokens_out, detail, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (event_type, event_name, conversation_id, doc_id, 1 if ok else 0, int(latency_ms),
             int(tokens_in), int(tokens_out), detail[:1000], _now()),
        )
    except Exception as e:
        logger.debug("事件写入失败（已忽略）: %s", e)


def ms_since(started: float) -> int:
    """开始时间（time.monotonic()）到现在的毫秒数（埋点延迟计算）。"""
    return int((time.monotonic() - started) * 1000)


def _events_summary(since: str) -> list[dict]:
    """按 event_type 聚合事件：次数 / 成功数 / 平均延迟 / token 合计。"""
    return _query(
        "SELECT event_type, COUNT(*) AS n, SUM(ok) AS ok_n, AVG(latency_ms) AS avg_ms,"
        " SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out"
        " FROM event_log WHERE created_at >= ? GROUP BY event_type ORDER BY n DESC",
        (since,),
    )


def _events_by_name(event_type: str, since: str, limit: int = 20) -> list[dict]:
    """按 event_name 聚合某一类事件。"""
    return _query(
        "SELECT event_name, COUNT(*) AS n, SUM(ok) AS ok_n, AVG(latency_ms) AS avg_ms"
        " FROM event_log WHERE event_type=? AND created_at >= ? GROUP BY event_name ORDER BY n DESC LIMIT ?",
        (event_type, since, limit),
    )


def _rate(ok_n: int, n: int) -> float:
    """成功率（0-1，保留 3 位）；无样本返回 0。"""
    return round(ok_n / n, 3) if n else 0.0


def _with_rate(rows: list[dict]) -> list[dict]:
    """给按 event_name 聚合的结果补充成功率字段。"""
    for r in rows:
        r["rate"] = _rate(int(r["ok_n"] or 0), int(r["n"] or 0))
    return rows


def stats_summary(days: int = 7) -> dict:
    """聚合统计：totals + by_event_type + 工具/本地/API 细分（供 GET /api/stats）。"""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    by_event_type: dict = {}
    totals = {"events": 0, "api_tokens_in": 0, "api_tokens_out": 0, "local_calls": 0,
              "tool_calls": 0, "tool_rate": 0.0, "interactions": 0}
    for r in _events_summary(since):
        etype = r["event_type"]
        n = int(r["n"] or 0)
        by_event_type[etype] = {
            "count": n,
            "ok": int(r["ok_n"] or 0),
            "rate": _rate(int(r["ok_n"] or 0), n),
            "avg_latency_ms": round(float(r["avg_ms"] or 0)),
            "tokens_in": int(r["tokens_in"] or 0),
            "tokens_out": int(r["tokens_out"] or 0),
        }
        totals["events"] += n
        if etype == "api_llm":
            totals["api_tokens_in"] += by_event_type[etype]["tokens_in"]
            totals["api_tokens_out"] += by_event_type[etype]["tokens_out"]
        elif etype == "local_llm":
            totals["local_calls"] = n
        elif etype == "tool":
            totals["tool_calls"] = n
            totals["tool_rate"] = by_event_type[etype]["rate"]
        elif etype == "interaction":
            totals["interactions"] = n
    return {
        "window_days": days,
        "since": since,
        "totals": totals,
        "by_event_type": by_event_type,
        "tools": _with_rate(_events_by_name("tool", since)),
        "local_tasks": _with_rate(_events_by_name("local_llm", since)),
        "api_tasks": _with_rate(_events_by_name("api_llm", since)),
    }


# ===== 工具轨迹（tool_steps：显示层回放） =====

def add_tool_step(conversation_id: int, tool_name: str, args_summary: str,
                  ok: bool, result_summary: str, message_id: Optional[int] = None,
                  call_id: str = "", args_full: str = "") -> int:
    """记录一次工具步骤（身份链：call_id / message_id 随事件即达，无需事后回填）。"""
    return _execute(
        "INSERT INTO tool_steps(conversation_id, message_id, call_id, tool_name, args_summary,"
        " args_full, ok, result_summary, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (conversation_id, message_id, call_id, tool_name, args_summary, args_full,
         1 if ok else 0, result_summary, _now()),
    )


def list_tool_steps(conversation_id: int) -> list[dict]:
    """获取会话的工具轨迹。"""
    return _query("SELECT * FROM tool_steps WHERE conversation_id=? ORDER BY id", (conversation_id,))


# ===== 模型层消息（model_messages：追加写入 + 整理删已压段，供下轮续接） =====

def _next_seq(conn: sqlite3.Connection, conversation_id: int) -> int:
    """对话内下一个组装序号（调用方须持锁）：以「现有最大 seq 与整理水位」的较大者为基准，
    保证整理清账（条目删除而水位保留）后新条目 seq 仍严格大于水位、对话内单调不复用。"""
    row = conn.execute("SELECT COALESCE(MAX(seq),0) FROM model_messages WHERE conversation_id=?",
                       (conversation_id,)).fetchone()
    floor_row = conn.execute("SELECT COALESCE(organized_seq,0) FROM conversations WHERE id=?",
                             (conversation_id,)).fetchone()
    floor = int(floor_row[0]) if floor_row else 0
    return max(int(row[0]), floor) + 1


def append_model_message(conversation_id: int, role: str, content: str = "",
                         tool_calls: str = "", tool_call_id: str = "") -> int:
    """追加一条模型层消息（请求进入时写 user 条目等），返回 seq。"""
    with _lock:
        conn = _get_conn()
        seq = _next_seq(conn, conversation_id)
        conn.execute("INSERT INTO model_messages(conversation_id, seq, role, content, tool_calls,"
                     " tool_call_id) VALUES(?,?,?,?,?,?)",
                     (conversation_id, seq, role, content, tool_calls, tool_call_id))
        conn.commit()
        return seq


def append_model_turn(conversation_id: int, ai_content: str, tool_calls: list[dict],
                      tool_results: list[dict]) -> None:
    """一轮工具交互原子写入：assistant(含 tool_calls) + N 条 tool 结果（同事务，防断连半写）。

    tool_calls 存 [{"name","args","id"}] JSON（与 LangChain AIMessage 结构一致，round-trip 最简）；
    tool_results 为 [{"call_id","content"}]。
    """
    payload = json.dumps(tool_calls, ensure_ascii=False)
    with _lock:
        conn = _get_conn()
        try:
            seq = _next_seq(conn, conversation_id)
            conn.execute("INSERT INTO model_messages(conversation_id, seq, role, content, tool_calls,"
                         " tool_call_id) VALUES(?,?,?,?,?,'')",
                         (conversation_id, seq, "assistant", ai_content, payload))
            for tr in tool_results:
                seq += 1
                conn.execute("INSERT INTO model_messages(conversation_id, seq, role, content, tool_calls,"
                             " tool_call_id) VALUES(?,?,?,?,'',?)",
                             (conversation_id, seq, "tool", str(tr.get("content") or ""),
                              str(tr.get("call_id") or "")))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def list_model_messages(conversation_id: int) -> list[dict]:
    """模型层历史（seq 升序，全量；v0.9：组装不裁剪，条数增长由「整理」统一消化）。"""
    return _query("SELECT * FROM model_messages WHERE conversation_id=? ORDER BY seq", (conversation_id,))


def delete_model_messages_upto(conversation_id: int, seq: int) -> int:
    """整理清账：删除模型层 seq ≤ seq 的条目（单事务），返回删除条数。

    并发安全核心：整理开始后写入的新条目 seq > cut，天然不在删除范围内。
    """
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute("DELETE FROM model_messages WHERE conversation_id=? AND seq<=?",
                               (conversation_id, int(seq)))
            conn.commit()
            return int(cur.rowcount)
        except Exception:
            conn.rollback()
            raise
