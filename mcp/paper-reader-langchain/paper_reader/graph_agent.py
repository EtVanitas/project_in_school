"""对话 Agent：langchain.agents.create_agent 构建（LangGraph 托管思考-工具循环与多轮记忆）。"""

import logging

from .ai import get_llm
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是「智能论文阅读助手」，服务于学术论文的检索、下载、阅读与管理。"
    "论文库 papers/ 分 unread（未读）与 read（已读），各含六个分类子目录：机器学习、强化学习、图像生成、大语言模型、多模态、其它。\n"
    "工作方式：1) search_papers 找论文（建议英文关键词），用户确认后 download_paper 下载并自动分类；"
    "2) extract_paper_content 提取内容后回答论文问题；"
    "3) 阅读完毕按用户要求 mark_as_read 归档；4) 深度研究用 research_assistant。\n"
    "规则：始终用中文回答；不确定文件名时先检索探测，不要凭空猜测；"
    "每次工具调用后根据结果判断是否需要继续，完成后给出清晰总结。"
)

_agent = None


def get_agent():
    """懒加载对话 Agent；未配置 LLM 时返回 None。"""
    global _agent
    if _agent is None:
        llm = get_llm(temperature=0.3)
        if llm is None:
            return None
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import MemorySaver

        _agent = create_agent(
            model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT,
            checkpointer=MemorySaver(), name="paper-reader-agent",
        )
    return _agent
