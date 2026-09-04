"""论文库：文件定位 / AI 分类 / 移动 / 整理，以及 PDF 文本提取。"""

import logging
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


class PaperError(Exception):
    """论文库业务异常（消息面向用户）。"""


def find_paper(filename: str, kind: str | None = None) -> Path | None:
    """全库（或限定 unread/read 区）定位论文文件，找不到返回 None。"""
    target = Path(filename).name
    dirs = config.all_paper_dirs() if kind is None else config._paper_dirs(kind)
    for d in dirs:
        candidate = d / target
        if candidate.exists():
            return candidate
    return None


def _state_label(kind: str) -> str:
    """状态中文名：read→已读，unread→未读。"""
    return "已读" if kind == "read" else "未读"


def _category_of(path: Path) -> str | None:
    """所在目录若是分类目录则返回分类名。"""
    parent = path.parent
    return parent.name if parent.name in config.CATEGORIES else None


def _content_hint(path: Path) -> str:
    """取 PDF 开头 2000 字符（标题+摘要）作 AI 分类线索。"""
    try:
        return extract_text(str(path), max_chars=2000)
    except Exception as e:
        logger.warning(f"提取 {path.name} 内容失败: {e}")
        return ""


def _classify(path: Path, hint_text: str | None = None) -> str:
    """AI 判定领域分类；无 LLM / 失败 / 无内容时回退「其它」。"""
    from .ai import classify_text

    content = hint_text or _content_hint(path)
    if not content.strip():
        return "其它"
    return classify_text(content)


def _move(source: Path, target_dir: Path) -> Path:
    """把文件移入目标目录；目标已有同名文件则删源（视为重复），源==目标则跳过。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if source != target:
        if target.exists():
            source.unlink()
        else:
            source.rename(target)
    return target


def _classify_and_move(filename: str, to_kind: str, hint_text: str | None = None) -> str:
    """定位 unread 区论文 → AI 分类 → 移动到 to_kind/{分类}；已在目标区则幂等返回。"""
    label = _state_label(to_kind)
    source = find_paper(filename, "unread")
    if source is None:
        if to_kind == "read":  # 标记已读的幂等场景：文件已在已读区
            target = find_paper(filename, "read")
            if target is not None:
                cat = _category_of(target)
                return f"论文 {filename} 已在{label}目录" + (f"的「{cat}」分类中" if cat else "中")
        raise PaperError(f"找不到论文文件: {filename}")
    if source.parent.name in config.CATEGORIES and source.parent.parent.name == to_kind:
        return f"论文 {filename} 已在{label}目录的「{source.parent.name}」分类中"
    category = _classify(source, hint_text)
    target = _move(source, config.PAPER_ROOT / to_kind / category)
    logger.info(f"论文 {filename} -> {target}")
    return f"已将论文 {filename} 标记为{'已读' if to_kind == 'read' else '已分类'}，放入「{category}」目录"


def categorize_in_unread(filename: str, hint_text: str | None = None) -> str:
    """下载后自动分类：把 unread 区论文归类到 unread/{分类}。"""
    return _classify_and_move(filename, "unread", hint_text)


def move_to_read(filename: str, hint_text: str | None = None) -> str:
    """标记已读：从未读区移动到 read/{分类}（AI 判定分类）。"""
    return _classify_and_move(filename, "read", hint_text)


def organize_paper_library() -> str:
    """整理论文库：删除 read/ 根目录散落 PDF，清理与已读重复的未读论文。"""
    messages: list[str] = []
    read_root, unread_root = config.PAPER_ROOT / "read", config.PAPER_ROOT / "unread"

    if read_root.exists():
        for f in read_root.iterdir():
            if f.is_file() and f.suffix.lower() == ".pdf":
                f.unlink()
                messages.append(f"已删除 read 目录中未分类的论文: {f.name}")

    read_names = {f.name for d in config._paper_dirs("read") if d.exists() for f in d.iterdir() if f.is_file()}
    if read_names and unread_root.exists():
        for d in config._paper_dirs("unread"):
            if d.exists():
                for f in d.iterdir():
                    if f.is_file() and f.name in read_names:
                        f.unlink()
                        messages.append(f"已删除 {d.relative_to(config.PAPER_ROOT)} 中的重复论文: {f.name}")
    return "\n".join(messages) if messages else "论文库整理完成，未发现需要处理的文件。"


def extract_text(pdf_path: str, max_chars: int = 0) -> str:
    """提取 PDF 全文文本；max_chars > 0 时截断并附加提示。"""
    import pymupdf  # PyMuPDF >= 1.24 官方命名

    parts: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    text = "\n".join(parts)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + f"\n……（内容过长，仅展示前 {max_chars} 字符）"
    return text
