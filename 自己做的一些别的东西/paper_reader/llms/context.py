"""模型上下文组装（纯函数）：历史 + 检索块 + 记忆 + 画像 + 划词 → LLM 消息序列。

契约：
- 纯函数：无 IO、无副作用、确定性（数据由调用方采集，可脱离服务单测；不抛异常）；
- 全量组装：不做预算裁剪、不改写历史条目——已写入模型的条目逐字节稳定，
  才能命中云端前缀缓存；上下文增长由「整理」统一消化；
- 两种形态：kind="agent"（ReAct 主循环，含画像与记忆注入，全量回放含工具消息）/ "answer"（降级直达问答，仅文本轮次）。

history 为模型层条目（model_messages 行）：role ∈ user / assistant（可含 tool_calls JSON）/ tool。
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from . import prompts


def _render_head(kind: str, doc_title: str, profile_block: str) -> str:
    """渲染 system 文本（agent 模板含标题与画像；answer 为固定提示）。"""
    if kind == "answer":
        return prompts.SYSTEM_ANSWER
    return prompts.SYSTEM_AGENT.format(title=doc_title or "未命名", profile=profile_block)


def _render_user_text(kind: str, doc_title: str, context_text: str, memory_text: str,
                      selected_text: str, question: str) -> str:
    """渲染当前轮用户消息（资料定界注入：检索块 / 记忆 / 划词 + 问题）。"""
    if kind == "answer":
        text = f"论文标题：{doc_title}\n\n论文内容：\n{context_text}\n\n"
        if selected_text:
            text += f"我在论文中选中的片段（重点参考）：\n{selected_text}\n\n"
        return text + f"我的问题：{question}"
    text = f"当前论文的预检索上下文：\n{context_text}\n\n" if context_text else ""
    if memory_text:
        text += memory_text + "\n\n"
    if selected_text:
        text += f"我选中的片段（重点参考）：\n{selected_text}\n\n"
    return text + f"我的问题：{question}"


def _parse_tool_calls(m: dict) -> list[dict]:
    """解析 assistant 条目的 tool_calls JSON（容错返回 []）。"""
    raw = m.get("tool_calls")
    if not raw:
        return []
    try:
        tcs = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return [tc for tc in (tcs or []) if isinstance(tc, dict) and tc.get("id")]


def _to_messages(history: list[dict], kind: str) -> list:
    """模型层历史条目 → LangChain 消息序列（容错，不抛异常）。

    角色映射：user → Human；assistant → AI（含 tool_calls 时带工具调用）；tool → ToolMessage。
    agent 路径做配对清洗：tool_calls 未全部配对的 assistant 降级为纯文本，其孤立的 tool 条目丢弃
    （保证 OpenAI 工具配对约束）；answer 路径仅回放文本轮次，跳过工具交互细节。
    """
    entries = [m for m in (history or []) if isinstance(m, dict)]
    if kind == "answer":
        out: list = []
        for m in entries:
            role = m.get("role")
            content = str(m.get("content") or "")
            if role == "tool":
                continue  # 工具观测不进入降级直达问答历史（保持纯文本轮次）
            if content.strip():
                out.append(HumanMessage(content) if role == "user" else AIMessage(content))
        return out
    paired = {str(m.get("tool_call_id") or "") for m in entries
              if m.get("role") == "tool" and m.get("tool_call_id")}
    retained: set[str] = set()        # 配对完整的助手调用 id（其 tool 结果才允许回放）
    parsed: dict[int, list] = {}      # 条目下标 → 保留的 tool_calls
    for i, m in enumerate(entries):
        if m.get("role") != "assistant":
            continue
        tcs = _parse_tool_calls(m)
        if tcs and all(str(tc["id"]) in paired for tc in tcs):
            parsed[i] = tcs
            retained |= {str(tc["id"]) for tc in tcs}
    out = []
    for i, m in enumerate(entries):
        role = m.get("role")
        content = str(m.get("content") or "")
        if role == "tool":
            tcid = str(m.get("tool_call_id") or "")
            if tcid in retained:
                out.append(ToolMessage(content, tool_call_id=tcid))
            continue
        if role == "assistant":
            if i in parsed:
                out.append(AIMessage(content, tool_calls=parsed[i]))
            elif content.strip():
                out.append(AIMessage(content))
            continue
        if content.strip():          # user 及其他未知角色按 user 处理
            out.append(HumanMessage(content))
    return out


def build_model_context(*, kind: str = "agent", doc_title: str, history: list[dict],
                        context_text: str, memory_text: str = "", profile_text: str = "",
                        selected_text: str = "", question: str) -> list:
    """组装发给 LLM 的消息序列：[system] + 历史 + [当前轮]（全量，不裁剪）。

    history 为模型层条目（user/assistant/tool；当前轮问题条目已由调用方过滤）；
    已写入条目逐字节稳定是缓存命中前提（G7），本函数不做任何丢弃与截断。
    """
    doc_title = str(doc_title or "")
    question = str(question or "")
    selected_text = str(selected_text or "")
    memory_text = str(memory_text or "")
    profile_text = str(profile_text or "")
    hist = _to_messages(history, kind)
    profile_block = (profile_text + "\n") if (kind != "answer" and profile_text) else ""
    head = _render_head(kind, doc_title, profile_block)
    user_text = _render_user_text(kind, doc_title, str(context_text or ""),
                                  memory_text, selected_text, question)
    return [SystemMessage(head)] + hist + [HumanMessage(user_text)]
