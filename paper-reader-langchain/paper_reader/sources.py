"""论文数据源与发现：HF Papers 榜单（日/周/月）抓取与兜底、arXiv 检索/元数据/PDF 下载、个性化重排。

所有外部请求带重试上限（config.MAX_RETRIES）；榜单实时抓取失败时回退本地缓存（stale 标记）。
"""

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path

from . import config, db

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def normalize_arxiv_id(raw: str) -> str:
    """arXiv ID 归一：去 URL 前缀与版本号（如 2509.06942v3 -> 2509.06942）。"""
    text = (raw or "").strip().replace("https://arxiv.org/abs/", "").replace("https://arxiv.org/pdf/", "")
    m = _ARXIV_ID_RE.search(text)
    return m.group(1) if m else text.strip()


def _clean(text: str) -> str:
    """折叠空白。"""
    return " ".join((text or "").split())


# ===== 发现源：HF Papers 榜单（daily / weekly / monthly） =====

_HF_PERIODS = ("daily", "weekly", "monthly")


def _period_query(period: str) -> str:
    """榜单周期 -> API 参数：weekly 用 ISO 周（如 2026-W38），monthly 用年月（如 2026-09）。"""
    today = date.today()
    if period == "weekly":
        y, w, _ = today.isocalendar()
        return f"week={y}-W{w:02d}"
    if period == "monthly":
        return f"month={today:%Y-%m}"
    return ""


def _hf_cache_file(period: str) -> Path:
    """缓存文件路径（按日期分文件，今天只爬一次）。"""
    today = date.today().strftime("%Y-%m-%d")
    return config.CACHE_DIR / f"hf_papers_{period}_{today}.json"


def _save_hf_cache(period: str, items: list[dict]) -> None:
    """抓取成功后落盘候选摘要（按日期缓存，次日自动刷新）。"""
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _hf_cache_file(period)
        cache_file.write_text(
            json.dumps({"fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "items": items}, ensure_ascii=False),
            encoding="utf-8")
        logger.info("HF Papers(%s) 缓存已保存：%s", period, cache_file.name)
    except Exception as e:
        logger.warning("写入 HF 榜单缓存失败：%s", e)


def _load_hf_cache(period: str) -> dict | None:
    """读取当日缓存（文件不存在/过期/损坏时返回 None，触发重新爬取）。"""
    cache_file = _hf_cache_file(period)
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        # 检查是否为今日缓存（同一天内复用）
        fetched_date = datetime.strptime(data.get("fetched_at", "")[:10], "%Y-%m-%d").date()
        if fetched_date != date.today():
            logger.info("HF Papers(%s) 缓存非今日，忽略：%s", period, data.get("fetched_at"))
            return None
        return data
    except Exception:
        return None


def fetch_hf_papers(period: str = "daily") -> dict:
    """抓取 HF Papers 榜单（最多 100 条，经 hf-mirror API），返回 {items, stale, cached_at, period}。

    实时抓取失败（重试耗尽）时回退最近一次成功的本地缓存（stale=True 供前端提示）。
    """
    import requests

    period = period if period in _HF_PERIODS else "daily"
    q = _period_query(period)
    url = f"{config.HF_ENDPOINT}/api/daily_papers?limit=100" + (f"&{q}" if q else "")
    last_err: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
            resp.raise_for_status()
            items = [_hf_item_to_candidate(it) for it in resp.json()[:100]]
            if not items:
                raise RuntimeError("接口返回空列表")
            _save_hf_cache(period, items)
            return {"items": items, "stale": False, "cached_at": "", "period": period}
        except Exception as e:  # 网络抖动重试
            last_err = e
            logger.warning("HF Papers(%s) 抓取第 %d 次失败: %s", period, attempt, e)
    cached = _load_hf_cache(period)
    if cached and cached.get("items"):
        logger.warning("HF Papers(%s) 实时抓取失败，回退缓存（%s）", period, cached.get("fetched_at", ""))
        return {"items": cached["items"], "stale": True, "cached_at": str(cached.get("fetched_at", "")), "period": period}
    raise RuntimeError(f"HF Papers 抓取失败（已重试 {config.MAX_RETRIES} 次）: {last_err}")


def _hf_item_to_candidate(item: dict) -> dict:
    """HF daily_papers 单条 -> 统一候选结构（字段缺失时降级）。"""
    paper = item.get("paper") or item
    arxiv_id = normalize_arxiv_id(str(paper.get("id", "")))
    authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or [])[:8])
    return {
        "source": "hf_papers",
        "arxiv_id": arxiv_id,
        "title": _clean(str(paper.get("title", ""))),
        "authors": authors,
        "abstract": _clean(str(paper.get("summary", ""))),
        "published": str(paper.get("publishedAt", ""))[:10],
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else str(item.get("url", "")),
        "upvotes": int(paper.get("upvotes", 0) or 0),
    }


# ===== 个性化推荐：关键词兴趣画像 + 热度融合重排（本地统计，零模型/API 依赖） =====

_WORD_RE = re.compile(r"[a-z]{3,}")  # 英文关键词：小写、长度≥3（连字符/数字视为分隔）
_STOPWORDS = frozenset("""
a about above after again against all also am an and another any are as at
be because been before being below between both but by
can cannot could did do does doing done down during
each few for from further
had has have having he her here hers him his how however
into is it its itself
may me might more most much must my
no nor not now
of off on once only or other others our out over own
same shall she should so some such
than that the their them then there these they this those through to too
under until up
very was we were what when where which while who whom why will with would
you your yours
article articles paper papers work works study studies
approach approaches method methods model models result results
show shows shown propose proposes proposed present presents presenting
learning neural network networks training trained
data dataset datasets experiment experiments experimental
performance task tasks large small high low better best good
new novel first second third one two three four five
use used uses using via based upon
""".split())
_READ_BOOST = 2.0        # 已读文档权重加成（读过 > 仅下载）
_DECAY_HALF_LIFE = 30.0  # 兴趣时间衰减半衰期（天），与长期记忆一致
_INTEREST_WEIGHT = 0.65  # 综合分中兴趣匹配占比（其余为热度）


def _words(text: str) -> list[str]:
    """英文词袋提取（小写、去停用词）；标题/摘要均为英文，中文笔记不参与。"""
    return [w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS]


def _age_days(date_text: str, today: date) -> float:
    """距今天数（解析失败返回 0，视为当天发布不衰减）。"""
    try:
        return max(0.0, float((today - datetime.strptime(date_text[:10], "%Y-%m-%d").date()).days))
    except (ValueError, TypeError):
        return 0.0


def _interest_profile() -> dict[str, float]:
    """库内文档 -> 兴趣词权重：标题×3 + 摘要（去重）；已读×2、按时间 90 天半衰期衰减。"""
    profile: dict[str, float] = {}
    today = date.today()
    for doc in db.list_documents():
        base = (0.5 ** (_age_days(str(doc.get("read_at") or doc.get("created_at") or ""), today) / _DECAY_HALF_LIFE)
                * (_READ_BOOST if doc.get("status") == "read" else 1.0))
        for w in _words(str(doc.get("title") or "")):
            profile[w] = profile.get(w, 0.0) + 3.0 * base
        for w in set(_words(str(doc.get("abstract") or ""))):
            profile[w] = profile.get(w, 0.0) + 1.0 * base
    return profile


def rank_personalized(candidates: list[dict]) -> list[dict]:
    """个性化重排：综合分 = 0.65×兴趣匹配 + 0.35×热度（均为列表内归一化）。

    兴趣匹配 = 候选标题/摘要命中画像词的权重和；重排后前 3 且命中的候选附
    reco（命中词 top3，前端展示推荐理由）。无画像（冷启动）时退化为热度排序。
    """
    if not candidates:
        return candidates
    profile = _interest_profile()
    max_hit = max_up = 1.0
    for c in candidates:
        max_up = max(max_up, float(c.get("upvotes") or 0))
        words = set(_words(str(c.get("title") or ""))) | set(_words(str(c.get("abstract") or "")))
        c["_hit_words"] = sorted((w for w in words if w in profile), key=lambda w: -profile[w])[:3]
        c["_hit"] = sum(profile[w] for w in c["_hit_words"])
        max_hit = max(max_hit, float(c["_hit"]))
    for c in candidates:
        c["_score"] = (_INTEREST_WEIGHT * (c.pop("_hit") / max_hit)
                        + (1 - _INTEREST_WEIGHT) * (float(c.get("upvotes") or 0) / max_up))
    ranked = sorted(candidates, key=lambda c: -c["_score"])
    for i, c in enumerate(ranked):
        c.pop("_score")
        reco = c.pop("_hit_words")
        c["reco"] = reco if reco and i < 3 else None
    return ranked


# ===== 发现源：arXiv =====

def _to_candidate(result) -> dict:
    """arxiv.Result -> 统一候选结构。"""
    arxiv_id = normalize_arxiv_id(result.get_short_id())
    return {
        "source": "arxiv",
        "arxiv_id": arxiv_id,
        "title": _clean(result.title),
        "authors": ", ".join(a.name for a in result.authors[:8]),
        "abstract": _clean(result.summary),
        "published": str(result.published.date()) if result.published else "",
        "url": result.entry_id,
        "upvotes": 0,
    }


def arxiv_search(query: str, limit: int = 10) -> list[dict]:
    """按关键词检索 arXiv（相关度排序）。"""
    import arxiv

    client = arxiv.Client(num_retries=config.MAX_RETRIES)
    search = arxiv.Search(query=query, max_results=max(1, min(int(limit), 50)),
                          sort_by=arxiv.SortCriterion.Relevance)
    return [_to_candidate(r) for r in client.results(search)]


def arxiv_meta(arxiv_id: str) -> dict:
    """按 arXiv ID 取元数据（标题/作者/摘要/日期）。"""
    import arxiv

    client = arxiv.Client(num_retries=config.MAX_RETRIES)
    results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
    if not results:
        raise RuntimeError(f"arXiv 上未找到 ID 为 {arxiv_id} 的论文")
    return _to_candidate(results[0])


# ===== PDF 下载 =====

def download_pdf(arxiv_id: str, dest: Path) -> None:
    """下载论文 PDF 到 dest（重试上限内流式下载并校验完整性）。"""
    import requests

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    last_err: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": _UA}, timeout=90, stream=True)
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            if tmp.stat().st_size < 10240 or not tmp.read_bytes()[:5].startswith(b"%PDF"):
                tmp.unlink(missing_ok=True)
                raise RuntimeError("下载内容不是有效 PDF（可能为错误页）")
            tmp.replace(dest)
            return
        except Exception as e:
            last_err = e
            logger.warning("下载 %s 第 %d 次失败: %s", arxiv_id, attempt, e)
    raise RuntimeError(f"论文 {arxiv_id} 下载失败（已重试 {config.MAX_RETRIES} 次）: {last_err}")


def download_into_library(candidate: dict) -> dict:
    """把候选论文下载到论文库并入库（未读区）；已存在则跳过。

    返回 {status: downloaded|skipped, doc_id, message}。
    """
    arxiv_id = normalize_arxiv_id(candidate.get("arxiv_id", ""))
    if not arxiv_id:
        raise RuntimeError("缺少有效的 arXiv ID")
    existing = db.find_document_by_arxiv(arxiv_id)
    if existing is not None:
        return {"status": "skipped", "doc_id": existing["id"], "message": f"《{existing['title']}》已在库中（未读区/已读区）"}

    # 元数据：候选自带优先，缺失时向 arXiv 补
    meta = dict(candidate)
    if not meta.get("title"):
        meta = arxiv_meta(arxiv_id)

    # 标题级查重兜底（同一论文不同 arXiv 版本/ID 时仍可识别）
    by_title = db.find_document_by_title(str(meta.get("title") or ""))
    if by_title is not None:
        return {"status": "skipped", "doc_id": by_title["id"],
                "message": f"《{by_title['title']}》已在库中（标题相同，arXiv ID {arxiv_id}）"}

    config.ensure_dirs()
    dest = config.DOCS_DIR / f"{arxiv_id}.pdf"
    download_pdf(arxiv_id, dest)
    doc_id = db.add_document(
        source=meta.get("source", "arxiv"), arxiv_id=arxiv_id,
        title=meta.get("title", arxiv_id), authors=meta.get("authors", ""),
        abstract=meta.get("abstract", ""), published=meta.get("published", ""),
        url=meta.get("url") or f"https://arxiv.org/abs/{arxiv_id}", pdf_path=str(dest),
    )
    return {"status": "downloaded", "doc_id": doc_id, "message": f"《{meta.get('title', arxiv_id)}》已下载入库"}
