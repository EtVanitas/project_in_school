"""文档解析：PDF 文本提取 + 段落级分块 + 单页/整册读取接口。

切分特性：跨页语义段合并（页末未完句 + 下页小写开头）、引用区识别与标记、
caption 归块（尾部重叠复写）、页眉页脚剔除。检索链路位于 docs/retrieval.py
（单向依赖本模块的 get_chunks / Chunk）；本模块不反向依赖。
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CHUNK_TARGET_CHARS = 1200  # 目标块大小（中英混合约 500~800 token）
_MAX_SENT_CHARS = 1800     # 单句超长（无标点）时硬切
_END_PUNCT = set(".!?。！？;；:：)】」»%")  # 页末见此集合字符视为语义完整、不跨页合并

_pages_cache: dict[str, list[str]] = {}   # pdf_path -> 每页文本
_chunks_cache: dict[str, list["Chunk"]] = {}

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
    id: int        # 全局块序（0-based）
    page: int      # 起始页码（1-based）
    text: str
    page_end: int = 0   # 结束页码（跨页块 > page；0 构造时归一为 page）
    kind: str = "text"  # text|reference，引用区块融合降权用

    def __post_init__(self):
        if not self.page_end:
            self.page_end = self.page

    @property
    def pages(self) -> set[int]:
        """块覆盖的全部页码（引用校验/used_pages 口径）。"""
        return set(range(self.page, self.page_end + 1))

    @property
    def page_label(self) -> str:
        """页码标记文本：单页「第 N 页」/ 跨页「第 N-M 页」。"""
        return f"第 {self.page} 页" if self.page_end == self.page else f"第 {self.page}-{self.page_end} 页"


_SENT_RE = re.compile(r"[^.!?。！？；;\n]*[.!?。！？；;\n]?")


def _split_sentences(page_text: str) -> list[str]:
    """按中英句末标点粗切句子（换行也断句，与页行近似 1:1），超长片段硬切兜底。"""
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


# ===== 切分辅助：页眉页脚 / 引用区 / 跨页续行 =====

_PAGENO_RE = re.compile(r"^\d{1,4}$")
_REFS_HEAD_RE = re.compile(r"^(References|Bibliography|参考文献)\s*$", re.I)
# 引用条目行特征：编号 [n]／多作者逗号链／「姓, 名缩写.」重复式／「缩写. 正文」式条目
_REF_ENTRY_RES = (
    re.compile(r"^\[\d+\]"),
    re.compile(r"(?:[A-Z][\w'’\-]+,\s){2,}"),
    re.compile(r"[A-Z][\w'’\-]+,\s+[A-Z]\..{0,60}?[A-Z][\w'’\-]+,\s+[A-Z]\."),
    re.compile(r"^[A-Z][\w'’\-]+\.\s+[A-Z].{20,}"),
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d\d\b")
_CAPTION_RE = re.compile(r"(?:Table|Figure|Fig\.|图|表)\s*\d+\s*[.:：]")  # caption 标记（非锚定，块内搜索用）


def _strip_repeated_edges(pages: list[str]) -> list[list[str]]:
    """去页眉/页脚/纯页码行：跨 ≥40% 页重复出现于页首两行或页尾两行的内容剔除（2.1.1）。"""
    from collections import Counter

    edge: Counter = Counter()
    for p in pages:
        ls = [ln.strip() for ln in p.splitlines() if ln.strip()]
        edge.update(ls[:2])
        edge.update(ls[-2:])
    thresh = max(3, int(len(pages) * 0.4))
    junk = {ln for ln, c in edge.items() if c >= thresh}
    out: list[list[str]] = []
    for p in pages:
        out.append([ln for ln in (x.strip() for x in p.splitlines())
                    if ln and not _PAGENO_RE.match(ln) and ln not in junk])
    return out


def _refs_region(page_lines: list[list[str]]) -> tuple[int, int]:
    """定位引用区 (标题页, 结束页)。从标题页向后逐页判“条目行≥3”，
    遇非引用版式（如附录伪代码/正文）即停，宁少标不误伤（护 Hit@5）。"""
    head = 0
    for i, ls in enumerate(page_lines, 1):
        if any(_REFS_HEAD_RE.match(ln) for ln in ls):
            head = i
            break
    if not head:
        return 0, 0
    end = head
    for i in range(head + 1, len(page_lines) + 1):
        ls = page_lines[i - 1]
        n_entry = sum(1 for ln in ls if any(r.search(ln) for r in _REF_ENTRY_RES))
        if n_entry >= 3 or (n_entry >= 2 and any(_YEAR_RE.search(ln) for ln in ls)):
            end = i
        else:
            break
    return head, end


def _is_continuation(prev_sent: str, next_sent: str) -> bool:
    """跨页语义续行：上页末句无句末标点 + 下页首句小写英文开头（与阶段一 C1 用例同判据）。"""
    if not prev_sent or not next_sent:
        return False
    return (prev_sent[-1] not in _END_PUNCT
            and next_sent[0].isascii() and next_sent[0].islower())


def _build_chunk(chunk_id: int, sents: list[tuple[int, str, bool]]) -> Chunk:
    """句子列表 → Chunk：页码取首尾区间，引用句子字符占多数则标 reference。"""
    ref_len = sum(len(s) for _, s, r in sents if r)
    total = sum(len(s) for _, s, _ in sents) or 1
    return Chunk(chunk_id, sents[0][0], " ".join(s for _, s, _ in sents),
                 sents[-1][0], "reference" if ref_len / total > 0.5 else "text")


def get_chunks(pdf_path: str) -> list[Chunk]:
    """页面文本 → 段落级块：去页眉页脚 + 跨页语义段合并 + 引用区标记 + caption 归块。

    块可跨页（page_end>page，页码元数据为区间）；未跨页时行为与旧版一致。
    """
    if pdf_path in _chunks_cache:
        return _chunks_cache[pdf_path]
    raw_pages = get_pages(pdf_path)
    page_lines = _strip_repeated_edges(raw_pages)
    refs_head, refs_end = _refs_region(page_lines)

    # 句子单元 (page, sent, is_ref)：References 标题页只把标题行之后的句子计入引用区
    units: list[tuple[int, str, bool]] = []
    for i, ls in enumerate(page_lines, 1):
        if i == refs_head:
            h = next((j for j, ln in enumerate(ls) if _REFS_HEAD_RE.match(ln)), -1)
            segs = [(ls[:h], False), (ls[h + 1:], True)] if h >= 0 else [(ls, False)]
        else:
            segs = [(ls, refs_head < i <= refs_end)]
        for lines, is_ref in segs:
            for s in _split_sentences("\n".join(lines)):
                units.append((i, s, is_ref))

    # 贪心聚合：页内满 1200 字切块；页界仅在语义续行处跨页累积
    chunks: list[Chunk] = []
    buf: list[tuple[int, str, bool]] = []
    blen = 0

    def _flush() -> None:
        nonlocal buf, blen
        if buf:
            chunks.append(_build_chunk(len(chunks), buf))
            buf, blen = [], 0

    for idx, (page, sent, is_ref) in enumerate(units):
        if buf and page != buf[-1][0] and not _is_continuation(buf[-1][1], sent):
            _flush()
        buf.append((page, sent, is_ref))
        blen += len(sent)
        if blen >= CHUNK_TARGET_CHARS:
            nxt = units[idx + 1] if idx + 1 < len(units) else None
            if nxt and nxt[0] != page and _is_continuation(sent, nxt[1]):
                # 切块边界恰落在页末且末句语义未完：末句携入下一块，避免跨页腰斩
                held = buf.pop()
                blen -= len(held[1])
                _flush()
                buf.append(held)
                blen = len(held[1])
            else:
                _flush()
    _flush()

    # caption 同块（2.1.2）：块尾 ≤250 字内出现 Table/图 N: 标记时，把尾部重叠复写到下一块头部，
    # 保证 caption 与其客体（表格数据/续句）同块；RRF 按名次融合，重复块影响可控
    for j in range(len(chunks) - 1):
        text = chunks[j].text
        marks = [m.start() for m in _CAPTION_RE.finditer(text)]
        if not marks:
            continue
        tail = text[marks[-1]:]
        if len(tail) > 250:
            continue
        nxt = chunks[j + 1]
        chunks[j + 1] = Chunk(nxt.id, nxt.page, tail + " " + nxt.text, nxt.page_end, nxt.kind)

    logger.info("分块完成 %s：%d 页 → %d 块（引用区 p%d-%d）",
                pdf_path, len(raw_pages), len(chunks), refs_head, refs_end)
    _chunks_cache[pdf_path] = chunks
    return chunks


# ===== 单页 / 整册读取接口（Agent 工具与 vision 共用） =====

_outline_cache: dict[str, list[dict]] = {}


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
