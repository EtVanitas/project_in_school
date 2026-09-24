"""智能论文阅读助手（网页版）入口。

用法：python main.py 启动服务（默认 http://127.0.0.1:8000）。
"""

from paper_reader import config, documents, llm
from paper_reader.db import init_db


def main() -> None:
    """启动服务。"""
    config.setup_logging()
    init_db()

    if not config.DEEPSEEK_API_KEY:
        print("警告：未配置 DEEPSEEK_API_KEY，AI 问答功能不可用（请在 .env 中配置）。")

    documents.warmup_vectors_async()  # 后台预热向量模型（失败静默：检索自动降级为词法）

    import uvicorn

    print(f"智能论文阅读助手已启动: http://127.0.0.1:{config.PORT}")
    print("开发模式前端: http://localhost:5173 （需先 cd frontend && npm run dev）")
    uvicorn.run("paper_reader.api:app", host="127.0.0.1", port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
