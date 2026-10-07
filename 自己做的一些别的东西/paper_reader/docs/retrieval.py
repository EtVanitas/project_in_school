"""混合检索：BM25 + 词覆盖度 + e5 稠密向量 → RRF 融合。

依赖方向：本模块单向导入 documents（get_chunks / Chunk）；向量模型不可用时
自动降级为纯词法检索；零命中由调用方以 sample_chunks 均匀采样兜底。
对外入口：图检索 retrieve_top_chunks(_dual)、Agent 工具 search_chunks、
图外降级 search_context、上下文渲染 render_context。
"""

import logging
import math
import re
import threading
from collections import Counter

import numpy as np

from .. import config
from .documents import Chunk, get_chunks

logger = logging.getLogger(__name__)

_REF_WEIGHT = 0.8          # 引用区块词法通道降权：轻压作者名碎片，保留引标题的实义词汇命中
_REF_VEC_WEIGHT = 0.8      # 引用区块向量通道降权：压中文问答的引用碎片淹没（R07 型污染）

_index_cache: dict[str, "_BM25Index"] = {}

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


def _rrf(rankings: list[list[int]], k: int = 60,
         weights: list[list[float]] | None = None) -> list[int]:
    """RRF 倒数排名融合多通道排序：Σ w/(k+rank)，k=60 衰减头部优势；
    weights 为按通道对齐的块级权重（每元素是逐块权重列表，None 等权；引用区块降权用）。"""
    fused: dict[int, float] = {}
    for ri, ranking in enumerate(rankings):
        w_list = weights[ri] if weights else None
        for rank, idx in enumerate(ranking, start=1):
            w = w_list[idx] if w_list else 1.0
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
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


def _invalidate_vectors(pdf_path: str) -> None:
    """清除单文档向量缓存（文件更换时调用）。"""
    _vec_cache.pop(pdf_path, None)


def vector_search(pdf_path: str, query: str, chunks: list, top_k: int | None = None) -> list[tuple[int, float]]:
    """余弦检索：返回 [(chunk_id, score)] 按相似度降序（默认全量排序）。"""
    if not chunks:
        return []
    qv = embed_texts([query], "query")[0]
    dv = get_chunk_vectors(pdf_path, chunks)
    if dv.shape[0] != len(chunks):  # 缓存与当前分块不一致：重建一次
        _invalidate_vectors(pdf_path)
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

def _retrieve(pdf_path: str, question: str, vector_query: str | None = None) -> list[Chunk]:
    """混合检索：BM25 + 词覆盖度 + 稠密向量（可用时）→ RRF 融合，返回按相关度排序的块。

    vector_query 可选：向量通道单独用改写词编码（中文原文会占满 e5 截断窗口，prejudge 改写分支用）。
    """
    chunks = get_chunks(pdf_path)
    if not chunks:
        return []
    terms = tokenize(question)
    rankings: list[list[int]] = []
    if terms:
        index = _get_index(pdf_path)
        rankings.append(_bm25_ranking(index, terms))
        rankings.append(_coverage_ranking(index, terms))
    vec = _vector_ranking(pdf_path, vector_query or question, chunks)
    if vec:
        rankings.append(vec)
    if not rankings:
        return []
    # 按通道分权：引用区块词法重降权（碎片污染）、向量轻降权（标题语义命中属实义）
    lex_w = [_REF_WEIGHT if c.kind == "reference" else 1.0 for c in chunks]
    vec_w = [_REF_VEC_WEIGHT if c.kind == "reference" else 1.0 for c in chunks]
    n_lex = len(rankings) - (1 if vec else 0)
    weights = [lex_w] * n_lex + ([vec_w] if vec else [])
    return [chunks[i] for i in _rrf(rankings, weights=weights)]


def sample_chunks(pdf_path: str, n: int = 12) -> list[Chunk]:
    """按位置均匀采样块（检索零命中兜底：保证全文视野，防跨语言/生僻表达零召回）。"""
    chunks = get_chunks(pdf_path)
    if not chunks:
        return []
    step = max(1, len(chunks) // n)
    return chunks[::step][:n]


def render_context(chunks: list[Chunk], ids: list[int]) -> str:
    """将指定块渲染为带【第 N 页】标记的上下文文本（超 CONTEXT_RENDER_LIMIT 截断；跨页块标记区间）。"""
    context = "\n\n".join(f"【{chunks[i].page_label}】\n{chunks[i].text}" for i in ids)
    if len(context) > config.CONTEXT_RENDER_LIMIT:
        context = context[:config.CONTEXT_RENDER_LIMIT] + "\n……（内容过长已截断）"
    return context


def retrieve_top_chunks(pdf_path: str, query: str, top_k: int = 12) -> list[Chunk]:
    """混合检索 Top-k 块（供图节点上下文组装与降级路径复用）。"""
    return _retrieve(pdf_path, query)[:max(1, top_k)]


def retrieve_top_chunks_dual(pdf_path: str, query: str, vector_query: str, top_k: int = 12) -> list[Chunk]:
    """双路混合检索：词法通道用「原文+改写」拼接，向量通道用改写词单独编码（prejudge 改写分支用）。"""
    return _retrieve(pdf_path, query, vector_query)[:max(1, top_k)]


def search_context(pdf_path: str, query: str, top_k: int = 6) -> tuple[str, set[int]]:
    """图外降级路径：混合检索 Top-k（零命中时均匀兜底）→ (页码标记上下文, 页码集合)。"""
    chunks = get_chunks(pdf_path)
    hits = retrieve_top_chunks(pdf_path, query, top_k) or sample_chunks(pdf_path, top_k)
    ids = [c.id for c in hits]
    text = render_context(chunks, ids)
    return text, {chunks[i].page for i in ids}


# ===== Agent 工具入口 =====


def search_chunks(pdf_path: str, query: str, top_k: int = 6) -> list[Chunk]:
    """单文档混合检索（Agent 工具入口，等价 retrieve_top_chunks；默认取 6 块）。"""
    return retrieve_top_chunks(pdf_path, query, top_k)
