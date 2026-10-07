"""工具层收口（四段管线）：声明 → 校验 → 执行 → 归一。

设计：
- 声明：SPECS（name/description/JSON Schema/超时档/executor 一体定义，杜绝声明与实现漂移）；
- 校验：validate_args 白名单类型强转 + 必填 + 范围（失败 → param 错误观测喂回自愈）；
- 执行：executor 同步（进线程池）或 async（原生 await）均可，统一按超时档包裹；
- 归一：ToolResult（ok / observation / pages / summary / error_kind），错误即数据、绝不抛异常。

启动自检 self_check()：声明完整性 + schema 合法性；不过 → logger.error（调用方决定是否拒绝启动）。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

from .. import config
from ..docs import documents, retrieval
from . import llm as llm_module

logger = logging.getLogger(__name__)

_MAX_TOOL_OUTPUT = config.MAX_TOOL_OUTPUT


@dataclass
class ToolResult:
    """工具执行归一结果（喂回模型与轨迹展示的统一结构）。"""
    ok: bool
    observation: str                 # 喂回模型的观测文本（成功为业务文本，失败为统一错误格式）
    pages: set[int] = field(default_factory=set)
    summary: str = ""                # 轨迹摘要（前端步骤条展示）
    error_kind: str = ""             # 失败时：param / timeout / tool
    snippets: dict[int, str] = field(default_factory=dict)  # 页码 → 原文块片段（引用回溯）


# ===== 执行器（参数已通过校验：类型强转与必填均已保证） =====

def _run_search_in_paper(args: dict, doc: dict) -> tuple[str, set[int], str, dict[int, str]]:
    """检索论文段落（词法+向量混合，带页码）。"""
    hits = retrieval.search_chunks(doc.get("pdf_path") or "", args["query"], top_k=4)
    if not hits:
        return "未检索到相关段落（可换用更具体的英文关键词，或用 read_page 浏览指定页）。", set(), "未命中", {}
    pages: set[int] = set()
    snippets: dict[int, str] = {}
    for c in hits:
        pages |= c.pages  # 跨页块覆盖整个页码区间，与上下文标记口径一致
        for p in c.pages:
            snippets.setdefault(p, c.text[:300])
    text = "\n\n".join(f"【{c.page_label}】{c.text[:800]}" for c in hits)
    return text, pages, f"命中 {len(hits)} 个段落（第 " + "、".join(str(p) for p in sorted(pages)) + " 页）", snippets


def _run_read_page(args: dict, doc: dict) -> tuple[str, set[int], str, dict[int, str]]:
    """读取指定页完整文本。"""
    page = args["page"]
    text = documents.get_page_text(doc.get("pdf_path") or "", page)
    if text.startswith("页码超出范围"):
        return text, set(), "页码越界", {}
    return text[:_MAX_TOOL_OUTPUT], {page}, f"读取第 {page} 页（{len(text)} 字符）", {page: text[:300]}


def _run_get_outline(args: dict, doc: dict) -> tuple[str, set[int], str, dict[int, str]]:
    """获取论文章节目录（标题 + 页码）。"""
    toc = documents.get_outline(doc.get("pdf_path") or "")
    if not toc:
        return "该 PDF 没有内置章节目录。", set(), "无目录", {}
    lines = "\n".join(
        f"{'  ' * (t['level'] - 1)}{t['title']}（第 {t['page']} 页）" for t in toc[:60])
    return lines, {t["page"] for t in toc}, f"{len(toc)} 个章节", {}


async def _run_look_at_page(args: dict, doc: dict) -> tuple[str, set[int], str, dict[int, str]]:
    """多模态解读指定页图表（vision 客户端原生异步）。"""
    page = args["page"]
    text = await llm_module.describe_page_text(str(doc.get("pdf_path") or ""), page)
    return text, {page}, f"多模态解读第 {page} 页图表（{len(text)} 字符）", {}


def _run_flag_note_issue(args: dict, doc: dict) -> tuple[str, set[int], str, dict[int, str]]:
    """提交笔记勘误建议（不直接改笔记，交用户确认卡片）。"""
    page = args.get("evidence_page")
    pages = {page} if isinstance(page, int) and page > 0 else set()
    return ("勘误建议已提交，系统将展示更正卡片供用户确认（不会直接修改笔记）。请继续完成回答。",
            pages, "提交勘误建议", {})


# ===== 声明（与执行同文件一体，杜绝漂移） =====

SPECS: dict[str, dict] = {
    "search_in_paper": {
        "name": "search_in_paper",
        "description": "在当前打开的论文中检索与查询最相关的段落，返回带页码的原文片段。"
                       "查找具体方法、实验结果、术语定义时使用。建议用英文关键词。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索关键词（英文效果更佳）"}},
            "required": ["query"],
        },
        "timeout_kind": "text",
        "executor": _run_search_in_paper,
    },
    "read_page": {
        "name": "read_page",
        "description": "读取当前论文指定页码的完整文本。已知目标内容所在页码、需要该页全部细节时使用。",
        "parameters": {
            "type": "object",
            "properties": {"page": {"type": "integer", "description": "页码（1 起）"}},
            "required": ["page"],
        },
        "timeout_kind": "text",
        "executor": _run_read_page,
    },
    "get_outline": {
        "name": "get_outline",
        "description": "获取当前论文的章节目录（标题 + 页码）。用于了解论文整体结构或定位主题所在章节。",
        "parameters": {"type": "object", "properties": {}},
        "timeout_kind": "text",
        "executor": _run_get_outline,
    },
    "look_at_page": {
        "name": "look_at_page",
        "description": "用多模态模型查看当前论文指定页的图/表/公式截图并解读。"
                       "当问题涉及图表内容、图像细节、公式排版（文本提取难以还原）时使用。",
        "parameters": {
            "type": "object",
            "properties": {"page": {"type": "integer", "description": "页码（1 起）"}},
            "required": ["page"],
        },
        "timeout_kind": "vision",
        "executor": _run_look_at_page,
    },
    "flag_note_issue": {
        "name": "flag_note_issue",
        "description": "提交「历史讨论记忆（笔记）」的勘误建议：核实后发现笔记中某条结论与论文原文矛盾、明显有误时使用。"
                       "仅在已用工具核实后调用；不会直接修改笔记，系统将把建议展示给用户确认。",
        "parameters": {
            "type": "object",
            "properties": {
                "original_text": {"type": "string", "description": "笔记中可能有误的原句（尽量逐字摘录，便于定位替换）"},
                "correction": {"type": "string", "description": "建议的修正文本（以论文事实为准）"},
                "reason": {"type": "string", "description": "问题说明：与论文何处矛盾 / 无依据 / 自相矛盾"},
                "evidence_page": {"type": "integer", "description": "核实的依据页码（论文页，1 起；可选）"},
            },
            "required": ["original_text", "correction", "reason"],
        },
        "timeout_kind": "text",
        "executor": _run_flag_note_issue,
    },
}


# ===== 校验与摘要 =====

def validate_args(name: str, args: dict | None) -> tuple[bool, dict | str]:
    """轻量校验：白名单类型强转（"3"→3）+ 必填 + 范围；返回 (ok, 清洗后参数 | 错误原因)。"""
    spec = SPECS.get(name)
    if spec is None:
        return False, f"未知工具：{name}"
    params = spec.get("parameters") or {}
    props: dict = params.get("properties") or {}
    required: list = params.get("required") or []
    src = args if isinstance(args, dict) else {}
    cleaned: dict = {}
    for key, rule in props.items():
        val = src.get(key)
        is_blank = val is None or (isinstance(val, str) and not val.strip())
        if is_blank:
            if key in required:
                return False, f"缺少必填参数：{key}"
            continue
        t = (rule or {}).get("type")
        if t == "integer":
            try:
                val = int(val)
            except (TypeError, ValueError):
                return False, f"参数 {key} 应为整数"
            if val < 1:
                return False, f"参数 {key} 超范围（应 ≥1）"
        elif t == "string":
            val = str(val).strip()
        cleaned[key] = val
    return True, cleaned


def args_summary(name: str, args: dict) -> str:
    """工具调用参数的一句话摘要（前端轨迹展示用）。"""
    q = str(args.get("query", "")).strip()[:50]
    if name == "search_in_paper":
        return f"检索「{q}」"
    if name == "read_page":
        return f"读取第 {args.get('page', '?')} 页"
    if name == "get_outline":
        return "获取论文目录"
    if name == "look_at_page":
        return f"查看第 {args.get('page', '?')} 页图表"
    if name == "flag_note_issue":
        return "提交笔记勘误建议"
    return f"调用 {name}"


# ===== 管线与导出 =====

def _fail(kind: str, tool: str, reason: str, summary: str) -> ToolResult:
    """失败归一（统一观测文本：错误即数据，不抛异常）。"""
    return ToolResult(ok=False, observation=config.observation(kind, tool, reason),
                      pages=set(), summary=summary, error_kind=kind)


async def _invoke(executor: Callable, args: dict, doc: dict, timeout: float):
    """统一调用执行器：同步进线程池、async 原生 await，按超时档 wait_for。"""
    coro = executor(args, doc) if asyncio.iscoroutinefunction(executor) else asyncio.to_thread(executor, args, doc)
    return await asyncio.wait_for(coro, timeout=timeout)


async def execute_tool(name: str, args: dict, doc: dict) -> ToolResult:
    """四段管线：查找 → 校验 → 执行（超时档）→ 归一。错误即数据，绝不抛异常。"""
    spec = SPECS.get(name)
    if spec is None:
        return _fail(config.KIND_PARAM, name, "未知工具", "未知工具")
    ok, payload = validate_args(name, args)
    if not ok:
        return _fail(config.KIND_PARAM, name, str(payload), "参数校验失败")
    timeout = config.VISION_TOOL_TIMEOUT if spec["timeout_kind"] == "vision" else config.TOOL_TIMEOUT
    try:
        obs, pages, summary, snippets = await _invoke(spec["executor"], payload, doc, timeout)
        return ToolResult(ok=True, observation=obs, pages=pages or set(), summary=summary,
                          snippets=snippets or {})
    except TimeoutError:
        logger.warning("工具 %s 执行超时（>%ss）", name, timeout)
        return _fail(config.KIND_TIMEOUT, name, f"执行超时（>{timeout}s）", "执行超时")
    except Exception as e:  # 执行器抛错：转 tool 观测喂回（用户可见文本不暴露栈）
        logger.exception("工具 %s 执行失败", name)
        return _fail(config.KIND_TOOL, name, str(e)[:200] or e.__class__.__name__,
                     f"执行出错：{e.__class__.__name__}")


def openai_tools() -> list[dict]:
    """OpenAI tools 格式声明（llm.bind_tools 使用）。"""
    return [{"type": "function",
             "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
            for s in SPECS.values()]


def self_check() -> list[str]:
    """启动自检：声明完整性 + schema 合法性；返回问题列表（空 = 通过）。"""
    problems: list[str] = []
    for name, s in SPECS.items():
        if not name or s.get("name") != name:
            problems.append(f"{name}: name 不一致")
        if not callable(s.get("executor")):
            problems.append(f"{name}: 缺少 executor")
        if not str(s.get("description") or "").strip():
            problems.append(f"{name}: 缺少 description")
        if s.get("timeout_kind") not in ("text", "vision"):
            problems.append(f"{name}: timeout_kind 非法")
        params = s.get("parameters")
        if not isinstance(params, dict):
            problems.append(f"{name}: parameters 非法")
            continue
        props = params.get("properties") or {}
        for req in params.get("required") or []:
            if req not in props:
                problems.append(f"{name}: required 参数 {req} 未在 properties 中声明")
    if problems:
        logger.error("工具自检未通过：%s", "; ".join(problems))
    return problems
