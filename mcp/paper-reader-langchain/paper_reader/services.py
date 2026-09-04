"""服务层（不绑定论文业务）：arXiv 检索 / 元数据 / PDF 下载。"""

from pathlib import Path

from . import config


# ===== arXiv：检索与下载 =====

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def _to_paper_dict(result) -> dict:
    """把 arxiv.Result 转成紧凑 dict。"""
    return {
        "id": result.get_short_id(),
        "title": " ".join(result.title.split()),
        "authors": [a.name for a in result.authors],
        "summary": " ".join(result.summary.split()),
        "pdf_url": result.pdf_url or f"https://arxiv.org/pdf/{result.get_short_id()}",
        "published": str(result.published.date()) if result.published else "",
        "entry_url": result.entry_id,
    }


def _query(kind: str, query: str, max_results: int = 5) -> list[dict]:
    """arXiv 查询：kind="search" 按关键词检索，否则按 ID 列表取元数据。"""
    import arxiv

    client = arxiv.Client(num_retries=3)
    search = arxiv.Search(
        **(dict(query=query, max_results=max_results) if kind == "search" else dict(id_list=[query])),
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = list(client.results(search))
    if kind != "search" and not results:
        raise RuntimeError(f"arXiv 上未找到 ID 为 {query} 的论文")
    return [_to_paper_dict(r) for r in results]


def _http_download(pdf_url: str, dest: Path) -> None:
    """requests 流式下载 PDF。"""
    import requests

    resp = requests.get(pdf_url, headers={"User-Agent": _UA}, timeout=90, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)


def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """按相关度检索 arXiv 论文。"""
    return _query("search", query, max(1, min(int(max_results), 20)))


def fetch_meta(paper_id: str) -> dict:
    """按 arXiv ID 查询论文元数据。"""
    return _query("meta", paper_id)[0]


def ensure_downloaded(paper_id: str) -> tuple[Path, dict]:
    """下载论文 PDF 到 unread 根目录（已存在则跳过），返回 (路径, 元数据)。

    先取元数据是为让调用方拿到标题+摘要做 AI 分类线索，省去重提 PDF 前文。
    """
    dest_dir = config.PAPER_ROOT / "unread"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{paper_id}.pdf"
    if dest.exists():
        try:
            return dest, fetch_meta(paper_id)
        except Exception:
            return dest, {}

    meta = fetch_meta(paper_id)
    try:
        _http_download(meta.get("pdf_url", ""), dest)
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"下载论文失败（请检查网络）: {e}")
    if dest.stat().st_size < 1024:  # 明显是错误页而非 PDF
        dest.unlink()
        raise RuntimeError(f"下载内容异常（文件过小），论文 {paper_id} 下载失败")
    return dest, meta
