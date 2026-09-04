"""Agent 工具集：7 个 @tool（检索 / 下载 / 内容 / 书架 / 研究）+ 统一异常守卫。"""

import functools
import json
import logging
import webbrowser

from langchain_core.tools import tool

from . import ai, config, services, storage

logger = logging.getLogger(__name__)

# ===== 统一异常守卫（业务异常透传，未知异常转中文提示）=====

def guard_tool(fn):
    """工具边界统一异常处理：工具始终返回可读字符串而非未处理异常。"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except storage.PaperError as e:  # 业务异常消息已面向用户
            return str(e)
        except Exception as e:
            logger.exception(f"工具 {fn.__name__} 执行异常")
            return f"执行 {fn.__name__} 时出错: {type(e).__name__}: {e}"

    return wrapper


# ===== 论文检索与下载 =====

def _paper_listing(papers: list[dict], max_summary_chars: int = 500) -> str:
    """论文列表压缩为紧凑 JSON（截断作者与摘要，节省上下文）。"""
    return json.dumps([
        {
            "id": p["id"],
            "title": p["title"],
            "authors": p["authors"][:6] + (["..."] if len(p["authors"]) > 6 else []),
            "summary": p["summary"][:max_summary_chars],
            "pdf_url": p["pdf_url"],
            "published": p.get("published", ""),
        }
        for p in papers
    ], ensure_ascii=False)


def _meta_hint(meta: dict) -> str:
    """用 arXiv 元数据（标题+摘要）构造 AI 分类线索。"""
    return f"{meta.get('title', '')}\n{meta.get('summary', '')[:800]}"


def _open_pdf_if_enabled(path) -> None:
    """按 AUTO_OPEN_PDF 配置自动打开 PDF。"""
    if config.AUTO_OPEN_PDF:
        try:
            webbrowser.open(str(path.resolve()))
        except Exception as e:
            logger.warning(f"自动打开 PDF 失败: {e}")


@tool
@guard_tool
def search_papers(query: str, max_results: int = 5) -> str:
    """按关键词在 arXiv 检索论文（相关度排序）。query 建议英文，支持引号与 AND/OR 语法；max_results 默认 5 最多 20。"""
    papers = services.arxiv_search(query, max_results)
    if not papers:
        return f"在 arXiv 上未搜索到与「{query}」相关的论文，可尝试更换关键词。"
    return f"已在 arXiv 上搜索「{query}」，找到 {len(papers)} 篇相关论文：\n{_paper_listing(papers)}"


@tool
@guard_tool
def download_paper(paper_id: str) -> str:
    """下载 arXiv 论文 PDF 到论文库并 AI 自动分类归档；paper_id 如 1706.03762（可含版本号）；已存在则跳过。"""
    paper_id = paper_id.strip().replace("/", "")
    existing = storage.find_paper(f"{paper_id}.pdf")
    if existing is not None:
        if existing.parent == config.PAPER_ROOT / "unread":
            categorize_msg = storage.categorize_in_unread(existing.name)
        else:
            categorize_msg = "（该论文已在库中，无需重新下载）"
        _open_pdf_if_enabled(existing)
        return f"论文 {paper_id} 已存在于库中: {existing}\n{categorize_msg}"

    path, meta = services.ensure_downloaded(paper_id)
    try:
        categorize_msg = storage.categorize_in_unread(f"{paper_id}.pdf", hint_text=_meta_hint(meta))
    except Exception as e:
        categorize_msg = f"注意：自动分类失败: {e}（论文已保留在 unread 根目录）"
    _open_pdf_if_enabled(path)
    # 分类会把文件移入 unread/{分类}，重新定位展示真实路径
    final_path = storage.find_paper(f"{paper_id}.pdf") or path
    return f"已成功下载论文至: {final_path}\n{categorize_msg}"


# ===== 论文内容 =====

@tool
@guard_tool
def extract_paper_content(file_path: str, max_chars: int = 12000) -> str:
    """提取论文 PDF 文本；file_path 为路径或文件名（自动在论文库定位），max_chars 限制返回长度。"""
    path = storage.find_paper(file_path)
    if path is None:
        return f"找不到文件: {file_path}"
    return storage.extract_text(str(path), max_chars=int(max_chars))


# ===== 书架管理 =====

@tool
@guard_tool
def auto_categorize_paper(filename: str) -> str:
    """把 unread 区未分类的论文 AI 分类到 unread/{分类} 子目录。"""
    return storage.categorize_in_unread(filename)


@tool
@guard_tool
def mark_as_read(filename: str) -> str:
    """把论文标记为已读并归档：从未读区移动到 read/{分类}（AI 判定领域分类）。"""
    return storage.move_to_read(filename)


@tool
@guard_tool
def organize_papers() -> str:
    """整理本地论文库：删除 read 根目录散落的未分类 PDF，清理与已读重复的未读论文。"""
    return storage.organize_paper_library()


# ===== 一键研究（单工具函数内顺序完成，无 LangGraph 图）=====

MAX_PAPERS = 3  # 深度精读的论文数


def _refine_query(topic: str) -> str:
    """把主题（可能为中文口语）改写成 arXiv 友好的英文检索式。"""
    llm = ai.get_llm(temperature=0.2)
    if llm is None:
        return topic.strip()
    try:
        resp = llm.invoke([
            ("system", "你是论文检索专家。把用户的研究主题改写成适合 arXiv API 的英文检索式"
             "（可用引号与 AND/OR 语法），只输出检索式本身。"),
            ("human", f"研究主题: {topic}"),
        ])
        return str(resp.content).strip().strip('"')[:200] or topic.strip()
    except Exception as e:
        logger.warning(f"主题改写失败，使用原文检索: {e}")
        return topic.strip()


def _analyze_one(paper: dict) -> str:
    """分析单篇论文：本地定位或下载 → AI 总结。"""
    pid, title = paper["id"], paper["title"]
    lines = [f"## {title}", f"- arXiv: {paper.get('entry_url') or f'https://arxiv.org/abs/{pid}'}"]
    notes: list[str] = []

    path = storage.find_paper(f"{pid}.pdf")
    if path is None:
        try:
            path, meta = services.ensure_downloaded(pid)
            hint = f"{meta.get('title', title)}\n{meta.get('summary', '')[:600]}"
            storage.categorize_in_unread(f"{pid}.pdf", hint_text=hint)
        except Exception as e:
            notes.append(f"下载失败（改用 arXiv 摘要分析）: {type(e).__name__}")
    if path is not None:
        lines.append(f"- 本地路径: {path}")

    try:
        text = storage.extract_text(str(path), max_chars=15000) if path is not None else paper.get("summary", "")
        summary = ai.summarize_text(title, text)
        if summary:
            lines.append(f"- AI 总结:\n{summary}")
    except Exception as e:
        notes.append(f"总结失败: {e}")
    if notes:
        lines.append("- 说明: " + "; ".join(notes))
    return "\n".join(lines)


@tool
@guard_tool
def research_assistant(topic: str) -> str:
    """一键深度研究：自动检索 arXiv 并精读最相关的几篇论文（下载入库、AI 总结），生成 Markdown 研究报告。"""
    query = _refine_query(topic)
    try:
        papers = services.arxiv_search(query, max_results=8)
    except Exception as e:
        return f"arXiv 检索失败: {e}"
    if not papers:
        return f"arXiv 上未检索到与「{topic}」相关的论文，可换个主题试试。"

    draft = "\n\n".join([f"# 研究主题：{topic}\n\n检索式: {query}"] + [_analyze_one(p) for p in papers[:MAX_PAPERS]])
    llm = ai.get_llm(temperature=0.3)
    if llm is None:
        return draft
    try:
        resp = llm.invoke([
            ("system", "你是资深学术研究助理。请把下面的论文分析整理成结构清晰的中文 Markdown 研究报告："
             "概览、逐篇要点与综合评述。保留论文链接与本地路径。"),
            ("human", f"研究主题: {topic}\n\n分析素材:\n{draft[:16000]}"),
        ])
        return str(resp.content).strip() or draft
    except Exception as e:
        logger.warning(f"报告生成失败，返回原始分析: {e}")
        return draft


ALL_TOOLS = [
    search_papers, download_paper, extract_paper_content,
    auto_categorize_paper, mark_as_read, organize_papers, research_assistant,
]

__all__ = ["ALL_TOOLS", "guard_tool"]
