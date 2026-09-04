"""交互式命令行（替代旧 client.py）：自然语言对话 + /research 快捷命令。

Agent 由 LangGraph 托管思考-工具循环，CLI 用 stream 实时呈现工具调用与流式回复。
"""

import logging
import sys

from . import config
from .ai import has_llm
from .graph_agent import get_agent

logger = logging.getLogger(__name__)

THREAD_ID = "paper-reader-cli"
QUIT_WORDS = {"quit", "exit", "退出", "q"}

HELP_TEXT = """直接输入自然语言对话，例如：
  帮我找几篇关于 transformer 的论文 / 下载 1706.03762 并分析它的核心思想
/research <主题>  一键深度研究，自动生成报告
/help            显示帮助
quit / exit / 退出  结束会话"""


def _summarize(text: str, limit: int = 200) -> str:
    """折叠换行并截断，用于控制台摘要展示。"""
    one_line = " ".join((text or "").split())
    return one_line if len(one_line) <= limit else one_line[:limit] + "…"


class PaperReaderCLI:
    def chat(self, user_input: str) -> None:
        """驱动 Agent 对话：流式展示工具调用与模型回复。"""
        print("助手: ", end="", flush=True)
        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": THREAD_ID}},
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                for msg in (update.get("messages", []) if isinstance(update, dict) else []):
                    content = getattr(msg, "content", None)
                    if node_name == "tools" and getattr(msg, "name", ""):
                        print(f"\n  ↳ 工具[{msg.name}] {_summarize(str(content))}", flush=True)
                    elif content:
                        print(str(content), end="", flush=True)
        print()

    def run(self) -> None:
        """主循环：欢迎 → 校验 LLM → 对话 → 退出。"""
        config.setup_logging(logging.WARNING)
        config.ensure_paper_library()
        if not has_llm():
            print("未检测到 DEEPSEEK_API_KEY。请在项目根目录创建 .env 文件填入 Key")
            print("（申请地址：https://platform.deepseek.com），或设置同名系统环境变量后重试。")
            return
        self.agent = get_agent()
        if self.agent is None:
            print("Agent 初始化失败（请检查 langchain/langgraph 依赖）。")
            return
        print("智能论文阅读助手（LangChain 重构版）—— arXiv 检索/下载/归档 · AI 总结 · 一键研究\n")
        print(HELP_TEXT)
        while True:
            try:
                user_input = input("\n您: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
            if user_input.lower() in QUIT_WORDS:
                break
            if user_input.startswith("/research"):
                self._run_research(user_input[len("/research"):].strip())
            elif user_input in ("/help", "/?"):
                print(HELP_TEXT)
            elif user_input.startswith("/"):
                print(f"未知命令: {user_input}（输入 /help 查看帮助）")
            else:
                try:
                    self.chat(user_input)
                except Exception as e:
                    logger.exception("对话执行异常")
                    print(f"\n执行过程中出错: {e}（可重试或输入 quit 退出）")
        print("再见！希望我的帮助对您有用。")

    def _run_research(self, topic: str) -> None:
        """/research 快捷命令：直接调用研究工具并打印报告。"""
        if not topic:
            print("用法: /research <主题>，例如 /research diffusion model 的最新进展")
            return
        from .tools import research_assistant

        print(f"正在研究：{topic}（需要几分钟，请稍候）…")
        print("\n" + str(research_assistant.invoke({"topic": topic})))


def main() -> None:
    """CLI 入口。"""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows 控制台兼容
    PaperReaderCLI().run()


if __name__ == "__main__":
    main()
