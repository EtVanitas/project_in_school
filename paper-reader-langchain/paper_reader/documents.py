"""文档处理与检索：PDF 提取/分块 + 混合检索（BM25 + 词覆盖 + 稠密向量）+ 工具支持。

检索链路：三通道 RRF 融合，向量模型不可用时自动降级为纯词法检索；
跨语言由 e5 承担，零命中时均匀采样兜底。内容经 render_context 输出带页码标记。
"""

import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import config

logger = logging.getLogger(__name__)

CHUNK_TARGET_CHARS = 1200  # 目标块大小（中英混合约 500~800 token）
_MAX_SENT_CHARS = 1800     # 单句超长（无标点）时硬切

_pages_cache: dict[str, list[str]] = {}   # pdf_path -> 每页文本
_chunks_cache: dict[str, list["Chunk"]] = {}
_index_cache: dict[str, "_BM25Index"] = {}


# ===== 文本提取与分块 =====

def get_pages(pdf_path: str) -> list[str]:
    """提取 PDF 每页文本（结果常驻内存，避免重复解析）。"""
    if pdf_path in _pages_cache:
        return _pages_cache[pdf_path]
    import pymupdf

    pages: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    _pages_cache[pdf_path] = pages
    return pages


@dataclass
class Chunk:
    id: int    # 全局块序（0-based）
    page: int  # 1-based 页码
    text: str


_SENT_RE = re.compile(r"[^.!?。！？；;\n]*[.!?。！？；;\n]?")


def _split_sentences(page_text: str) -> list[str]:
    """按中英句末标点粗切句子，超长片段硬切兜底。"""
    out: list[str] = []
    for p in _SENT_RE.findall(page_text):
        p = p.strip()
        if not p:
            continue
        while len(p) > _MAX_SENT_CHARS:
            out.append(p[:_MAX_SENT_CHARS])
            p = p[_MAX_SENT_CHARS:]
        if p:
            out.append(p)
    return out


def get_chunks(pdf_path: str) -> list[Chunk]:
    """页面文本 → 段落级块（页内贪心聚合，块不跨页，页码元数据干净）。"""
    if pdf_path in _chunks_cache:
        return _chunks_cache[pdf_path]
    chunks: list[Chunk] = []
    for i, text in enumerate(get_pages(pdf_path)):
        buf: list[str] = []
        blen = 0
        for sent in _split_sentences(text):
            buf.append(sent)
            blen += len(sent)
            if blen >= CHUNK_TARGET_CHARS:
                chunks.append(Chunk(len(chunks), i + 1, " ".join(buf)))
                buf, blen = [], 0
        if buf:  # 页尾残段单独成块
            chunks.append(Chunk(len(chunks), i + 1, " ".join(buf)))
    logger.info("分块完成 %s：%d 页 → %d 块", pdf_path, len(get_pages(pdf_path)), len(chunks))
    _chunks_cache[pdf_path] = chunks
    return chunks


# ===== 轻量分词（免第三方分词库） =====

_TERM_RE = re.compile(r"[a-z][a-z0-9\-]{1,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_STOPWORDS = frozenset(
    "the and for with that this from are was were will would should could have has had not but its "
    "their there these those they then than them such also into over under between during before "
    "after above below more most other some any each both few many much own same very can may might "
    "which who whose what when where why how does did doing done using used use two one three".split()
)


def tokenize(text: str) -> list[str]:
    """英文词（小写、去停用词）+ 中文双字滑窗。"""
    tokens = [w for w in _TERM_RE.findall(text.lower()) if w not in _STOPWORDS]
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


# ===== BM25 词法检索（通道 1/2） =====

class _BM25Index:
    """单文档块级索引：词频 / 文档频率 / 块长度统计。"""

    def __init__(self, chunks: list[Chunk]):
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.df: Counter = Counter()
        for tf in self.tf:
            self.df.update(tf.keys())
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0


def _get_index(pdf_path: str) -> _BM25Index:
    if pdf_path not in _index_cache:
        _index_cache[pdf_path] = _BM25Index(get_chunks(pdf_path))
    return _index_cache[pdf_path]


_K1, _B = 1.5, 0.75  # BM25 参数


def _bm25_ranking(index: _BM25Index, terms: list[str]) -> list[int]:
    """通道 1：BM25 分数排序（返回命中块的 id 列表，从高到低）。"""
    n = len(index.tf)
    scores = [0.0] * n
    for term in terms:
        df = index.df.get(term, 0)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, tf in enumerate(index.tf):
            f = tf.get(term, 0)
            if f:
                scores[i] += idf * f * (_K1 + 1) / (f + _K1 * (1 - _B + _B * index.lengths[i] / index.avgdl))
    return [i for i in sorted(range(n), key=lambda i: (-scores[i], i)) if scores[i] > 0]


def _coverage_ranking(index: _BM25Index, terms: list[str]) -> list[int]:
    """通道 2：问题词命中种类数排序（对多词问题的召回补充）。"""
    qset = set(terms)
    cov = [len(qset & tf.keys()) for tf in index.tf]
    return [i for i in sorted(range(len(cov)), key=lambda i: (-cov[i], i)) if cov[i] > 0]


def _rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """RRF 倒数排名融合多通道排序：Σ 1/(k+rank)，k=60 衰减头部优势。"""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda i: (-fused[i], i))


# ===== 向量通道（multilingual-e5-small，通道 3） =====

class EmbeddingError(RuntimeError):
    """向量模型不可用 / 编码失败（调用方应降级）。"""


_vector_model = None
_encode_lock = threading.Lock()          # 单卡串行编码，避免并发竞争
_vec_cache: dict[str, np.ndarray] = {}   # pdf_path -> (n_chunks, dim) float32 归一化

_PREFIX_QUERY = "query: "      # e5 规范前缀：查询
_PREFIX_PASSAGE = "passage: "  # e5 规范前缀：文档


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load_vector_model():
    """加载向量模型单例；失败抛 EmbeddingError。"""
    global _vector_model
    if _vector_model is not None:
        return _vector_model
    with config.MODEL_LOAD_LOCK:  # 与本地 LLM 共用加载锁，避免 transformers 并发导入竞态
        if _vector_model is not None:
            return _vector_model
        path = config.LOCAL_MODEL_DIR / config.EMBED_MODEL_NAME
        if not path.exists():
            raise EmbeddingError(f"未找到向量模型目录: {path}")
        try:
            from sentence_transformers import SentenceTransformer

            config.silence_hf()
            _vector_model = SentenceTransformer(str(path), device="cuda" if _cuda_ok() else "cpu")
            logger.info("向量模型加载完成: %s（device=%s）", path, _vector_model.device)
            return _vector_model
        except Exception as e:
            raise EmbeddingError(f"加载向量模型失败: {e}") from e


def embed_texts(texts: list[str], kind: str = "passage") -> np.ndarray:
    """批量编码为归一化 float32 向量；kind 为 passage|query。"""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _load_vector_model()
    prefix = _PREFIX_QUERY if kind == "query" else _PREFIX_PASSAGE
    try:
        with _encode_lock:
            vecs = model.encode([prefix + t for t in texts], batch_size=32,
                                normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:
        raise EmbeddingError(f"向量编码失败: {e}") from e
    return np.asarray(vecs, dtype=np.float32)


def get_chunk_vectors(pdf_path: str, chunks: list) -> np.ndarray:
    """chunk 级向量（进程内缓存；chunks 变化时由调用方传入新对象即可，缓存键为路径）。"""
    cached = _vec_cache.get(pdf_path)
    if cached is not None:
        return cached
    vecs = embed_texts([c.text for c in chunks], "passage")
    _vec_cache[pdf_path] = vecs
    return vecs


def invalidate_vectors(pdf_path: str) -> None:
    """清除单文档向量缓存（文件更换时调用）。"""
    _vec_cache.pop(pdf_path, None)


def vector_search(pdf_path: str, query: str, chunks: list, top_k: int | None = None) -> list[tuple[int, float]]:
    """余弦检索：返回 [(chunk_id, score)] 按相似度降序（默认全量排序）。"""
    if not chunks:
        return []
    qv = embed_texts([query], "query")[0]
    dv = get_chunk_vectors(pdf_path, chunks)
    if dv.shape[0] != len(chunks):  # 缓存与当前分块不一致：重建一次
        invalidate_vectors(pdf_path)
        dv = get_chunk_vectors(pdf_path, chunks)
    scores = dv @ qv
    order = np.argsort(-scores)
    if top_k:
        order = order[:max(1, top_k)]
    return [(int(i), float(scores[i])) for i in order]


def warmup_vectors_async() -> None:
    """服务启动时后台预热（加载 + 一次微型编码）；失败静默（检索自动降级为词法）。"""

    def _job() -> None:
        try:
            embed_texts(["warmup"], "query")
            logger.info("向量模型预热完成")
        except Exception as e:
            logger.warning("向量模型预热失败（检索将降级为词法）: %s", e)

    threading.Thread(target=_job, daemon=True).start()


def _vector_ranking(pdf_path: str, question: str, chunks: list[Chunk]) -> list[int]:
    """通道 3：稠密向量余弦排序；不可用/失败时返回空（检索自动降级）。"""
    try:
        return [i for i, _ in vector_search(pdf_path, question, chunks)]
    except Exception as e:  # EmbeddingError 等一切异常都不应中断检索
        logger.warning("向量通道不可用，降级为词法检索: %s", e)
        return []


# ===== 混合检索与上下文 =====

def _retrieve(pdf_path: str, question: str) -> list[Chunk]:
    """混合检索：BM25 + 词覆盖度 + 稠密向量（可用时）→ RRF 融合，返回按相关度排序的块。"""
    chunks = get_chunks(pdf_path)
    if not chunks:
        return []
    terms = tokenize(question)
    rankings: list[list[int]] = []
    if terms:
        index = _get_index(pdf_path)
        rankings.append(_bm25_ranking(index, terms))
        rankings.append(_coverage_ranking(index, terms))
    vec = _vector_ranking(pdf_path, question, chunks)
    if vec:
        rankings.append(vec)
    if not rankings:
        return []
    return [chunks[i] for i in _rrf(rankings)]


def sample_chunks(pdf_path: str, n: int = 12) -> list[Chunk]:
    """按位置均匀采样块（检索零命中兜底：保证全文视野，防跨语言/生僻表达零召回）。"""
    chunks = get_chunks(pdf_path)
    if not chunks:
        return []
    step = max(1, len(chunks) // n)
    return chunks[::step][:n]


def _render_chunks(chunks: list[Chunk], ids: list[int], limit: int) -> str:
    """渲染块为带页码标记的上下文文本（超限截断）。"""
    context = "\n\n".join(f"【第 {chunks[i].page} 页】\n{chunks[i].text}" for i in ids)
    if len(context) > limit:
        context = context[:limit] + "\n……（内容过长已截断）"
    return context


def render_context(chunks: list[Chunk], ids: list[int], limit: int | None = None) -> str:
    """将指定块渲染为带【第 N 页】标记的上下文文本。"""
    return _render_chunks(chunks, ids, limit if limit is not None else config.MAX_CONTEXT_CHARS)


def retrieve_top_chunks(pdf_path: str, query: str, top_k: int = 12) -> list[Chunk]:
    """混合检索 Top-k 块（供图节点上下文组装与降级路径复用）。"""
    return _retrieve(pdf_path, query)[:max(1, top_k)]


def search_context(pdf_path: str, query: str, top_k: int = 6) -> tuple[str, set[int]]:
    """图外降级路径：混合检索 Top-k（零命中时均匀兜底）→ (页码标记上下文, 页码集合)。"""
    chunks = get_chunks(pdf_path)
    hits = retrieve_top_chunks(pdf_path, query, top_k) or sample_chunks(pdf_path, top_k)
    ids = [c.id for c in hits]
    text = render_context(chunks, ids)
    return text, {chunks[i].page for i in ids}


# ===== Agent 工具支持 =====

_outline_cache: dict[str, list[dict]] = {}


def search_chunks(pdf_path: str, query: str, top_k: int = 6) -> list[Chunk]:
    """单文档混合检索（Agent 工具用）：返回最相关的 Top-k 块。"""
    return _retrieve(pdf_path, query)[:max(1, top_k)]


def get_page_text(pdf_path: str, page: int) -> str:
    """读取单页文本（1-based）；越界/无文本时返回可读提示（不抛异常）。"""
    pages = get_pages(pdf_path)
    if page < 1 or page > len(pages):
        return f"页码超出范围：本文档共 {len(pages)} 页。"
    return pages[page - 1].strip() or "（本页没有可提取的文本）"


def render_page_image(pdf_path: str, page: int, max_width: int = 1200) -> bytes:
    """渲染指定页为 JPEG 字节（多模态图表解读用）；页码越界抛 ValueError。"""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        if page < 1 or page > doc.page_count:
            raise ValueError(f"页码超出范围（1-{doc.page_count}）")
        p = doc[page - 1]
        zoom = max(1.0, min(2.5, max_width / max(p.rect.width, 1.0)))
        pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("jpeg", jpg_quality=80)


def get_outline(pdf_path: str) -> list[dict]:
    """文档目录（PyMuPDF TOC）：[{level, title, page}]；无目录返回空列表。"""
    if pdf_path in _outline_cache:
        return _outline_cache[pdf_path]
    import pymupdf

    out: list[dict] = []
    try:
        with pymupdf.open(pdf_path) as doc:
            for lv, title, page in (doc.get_toc() or []):
                out.append({"level": int(lv), "title": str(title).strip(), "page": int(page)})
    except Exception as e:
        logger.warning("读取目录失败 %s: %s", pdf_path, e)
    _outline_cache[pdf_path] = out
    return out
