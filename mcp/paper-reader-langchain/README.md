# 智能论文阅读助手（LangChain 重构版）

基于 **LangChain 1.x + LangGraph** 重构原 MCP 论文阅读应用：用自然语言对话驱动 Agent，完成
arXiv 论文的**检索 → 下载 → AI 分类归档 → 精读（AI 总结）→ 管理**全流程。
重构保持原系统功能可用，同时**代码量降至原实现一半以下**（约 710 行、9 个 .py，原约 1990 行、20+ 文件）。

## 技术栈

| 组件 | 选型 | 说明 |
| --- | --- | --- |
| Agent 框架 | `langchain.agents.create_agent` + LangGraph | 托管思考-工具循环、多轮对话记忆 |
| 主模型 | DeepSeek（OpenAI 兼容，`langchain-openai` 接入） | 对话 / 分类 / 总结 / 报告 |
| 论文数据源 | `arxiv` 官方库 + `requests` 流式下载 | 免 Key |
| PDF 解析 | `pymupdf` | 单一文本提取引擎 |

## 功能一览（Agent 可自主调用的 7 个工具）

| 类别 | 工具 | 作用 |
| --- | --- | --- |
| 论文获取 | `search_papers` | 按关键词检索 arXiv（相关度排序） |
| | `download_paper` | 下载 PDF 入库并 AI 自动分类 |
| 精读理解 | `extract_paper_content` | 提取论文 PDF 文本内容 |
| 书架管理 | `auto_categorize_paper` | 给未读论文 AI 分类归档 |
| | `mark_as_read` | 标记已读，移入 read/{分类} |
| | `organize_papers` | 整理去重（清理 read 散落 PDF、未读重复件） |
| 深度研究 | `research_assistant` | 一键生成 Markdown 研究报告 |

> 说明：LLM 不可用（未配置 Key / 调用失败）时，AI 功能自动降级——
> 分类回退「其它」、总结返回空、报告返回原文分析，离线仍可用。

## 论文库结构

```
papers/
├── unread/          # 未读
│   ├── 机器学习/ 强化学习/ 图像生成/ 大语言模型/ 多模态/ 其它/
└── read/            # 已读（目录结构同上）
```

> 兼容旧系统：可把 `PAPER_ROOT` 指向旧论文库目录沿用存量 PDF；
> 旧库的 `translated/` 目录也会参与检索定位。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（DeepSeek，申请：https://platform.deepseek.com）
# 在项目根目录创建 .env 文件，填入 DEEPSEEK_API_KEY（详见下方「环境变量」表）；
# 也可不创建 .env，直接设置同名系统环境变量

# 3. 启动对话
python main.py
```

## 使用示例

启动后直接输入自然语言（多轮对话，Agent 会自动决策调用工具）：

```
您: 帮我找几篇关于 diffusion model 的论文
您: 下载 1706.03762 并分析它的核心思想
您: 我读完了，帮我把 1706.03762 标记为已读
您: 我的论文库现在有哪些没读的？帮我整理一下
您: /research 大语言模型推理加速的最新进展     ← 一键深度研究报告
您: quit                                     ← 退出
```

## 项目结构（仅 9 个 .py：1 入口 + 8 模块）

```
paper-reader-langchain/
├── main.py                  # 入口（python main.py）
├── requirements.txt / .env
└── paper_reader/
    ├── __init__.py
    ├── cli.py               # 交互式命令行（流式输出）
    ├── config.py            # 全局配置与论文库初始化
    ├── ai.py                # LLM 工厂 + AI 分类/总结
    ├── graph_agent.py       # Agent 构建与系统提示词
    ├── storage.py           # 论文库文件操作 + PDF 文本提取
    ├── tools.py             # 7 个 @tool + 一键研究 + 统一异常守卫
    └── services.py          # arXiv 检索/下载
```

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 无（必填） | DeepSeek Key，也可用系统环境变量 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名 |
| `AUTO_OPEN_PDF` | `true` | 下载论文后自动打开 PDF |
| `PAPER_ROOT` | `./papers` | 论文库根目录（可指向旧库） |
