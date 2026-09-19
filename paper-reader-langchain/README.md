# 智能论文阅读助手（paper-reader-langchain）

> v0.5.4 ｜ 本地优先的 LLM 论文阅读 Agent（面试项目）

三栏网页阅读器：**LangGraph 状态图编排 + 大小模型协作（本地 Qwen3-1.7B / 云端 DeepSeek）+ Agentic RAG + 对话总结记忆**。

核心闭环：**发现（HF 日/周/月热门榜 + arXiv 搜索 + 个性化重排）→ 勾选下载 → 网页阅读 → 选中段落问 AI（混合检索 + 多步 Agent + 流式）→ 图表解读 → 对话总结整理成笔记（下次讨论自动作背景记忆）**

## 亮点速览

| # | 亮点 | 说明 |
| --- | --- | --- |
| 1 | LangGraph 状态图编排 | 4 节点线性架构（guard→retrieve→agent→log），AsyncSqliteSaver 每步落盘，失败可断点续跑 |
| 2 | 大小模型分工（零 Token 总结） | 本地 1.7B 专责「信息压缩」任务（对话总结），零 API token；所有生成（问答/图表解读）均由 DeepSeek 承担 |
| 3 | Agentic RAG | BM25 + 词覆盖度 + e5 稠密向量三通道 → RRF 融合 Top-12 → 直接注入 Agent（删除精排评分，响应更快） |
| 4 | 护栏式 ReAct Agent | 5 工具（4 只读查论文 + 1 笔记勘误提议）自主多步推理；6 轮 / 文本工具 20s・看图 90s / 整响应 300s 三重护栏；工具连续失败自动弃用、超限强制收敛；轨迹事件流式可见 |
| 5 | 对话总结记忆 + 勘误闭环 | 一键「整理记忆」：对话太长或结束时压缩成结构化笔记（可编辑），问答时自动注入作背景；Agent 核实发现笔记与论文矛盾时提交勘误建议，用户确认后更正（原句留痕） |
| 6 | 全链路降级保底 | 每个节点、每次模型/网络故障都有兜底路径（见「降级矩阵」），服务不中断 |
| 7 | 多模态图表解读 | PDF 单页渲染 → DeepSeek vision 流式解读；工具栏按钮 + Agent `look_at_page` 工具双入口 |
| 8 | 代码精简优化 | v0.5 模块合并（16→10：图三合一 / 笔记 + 记忆合并 / 向量并入检索 / 本地模型并入 llm）+ 精排评分合一；v0.5.1 删除废弃数据库表及死代码；无闲聊分流/跨文档检索等过度设计 |
| 9 | 响应速度优化 | 删除 1.7B 相关性评分节点（延迟高、收益低），改为纯 RRF 排序；本地模型专用于对话总结（离线任务，延迟可接受） |

## 功能一览（按使用流程）

### 1. 发现 —— 找论文

- **HF Papers 热门榜**：日/周/月三档（hf-mirror 镜像 API）；**按日期缓存**（今天只爬一次，次日自动刷新），实时抓取失败自动回退当日缓存（前端提示 stale 与缓存时间）
- **arXiv 搜索**：关键词检索（相关度排序），最多 50 条
- **个性化推荐**：由库内文档标题（×3）+ 摘要提取英文关键词生成兴趣画像（已读×2 加权、90 天半衰期衰减），候选按 `0.65×兴趣匹配 + 0.35×热度` 重排；前 3 篇标注推荐命中词（纯本地统计，零 token）
- **勾选批量下载** / 直接输入 arXiv ID；arXiv ID + 标题双查重（重复自动跳过）；下载后校验 PDF 完整性

### 2. 阅读 —— 读论文

- 三栏布局：左（对话历史 + 文档库）、中（阅读与浏览）、右（问答对话）
- 文档库分区：**未读区 / 已读区 / 目录页**（摘要卡片，标注笔记数与对话数，支持筛选）
- 浏览器内 PDF 翻页缩放（react-pdf）；**选中文字自动生成引用条**，提问时锚定该段落
- **标记已读 / 退回未读 / 删除**（级联清理对话、消息、工具轨迹）

### 3. 问答 —— 问 AI（核心链路）

- SSE 流式输出；**工具轨迹时间线**实时可见（agent_step / agent_observation 事件配对展开与折叠）
- 每次提问进入 LangGraph 状态图（见下方架构图）：混合检索（RRF 排序）→ Agent 多步推理（统一由 DeepSeek 生成，无意图分流）
- **引用护栏**：回答后校验页码引用是否越界（区间引用 ≥50% 宽容），越界时前端提示
- **历史回放**：切回旧对话时，工具轨迹按归属消息复位展示

### 4. 图表解读 —— 看图

- PDF 工具栏「解读本页」：当前页渲染为截图 → DeepSeek vision 流式解读图 / 表 / 公式
- Agent 工具 `look_at_page`：问答中涉及图像细节时由模型自主调用

### 5. 总结与记忆 —— 沉淀讨论

- **整理记忆**：对话太长或结束时点右栏「整理记忆」→ DeepSeek 把整段对话压缩为结构化笔记（讨论主题 / 核心结论 / 待跟进问题）→ 追加到该论文笔记文件（`data/notes/{arxiv_id}.md`，`## 日期 对话整理` 小节）
- **笔记 = 记忆文件**：每篇论文一个 Markdown 文件，「笔记」视图可查看 / 编辑 / 保存（用户可修改整理结果，也可手写补充）
- **自动注入**：问答时读取该论文笔记（最新部分，限 4000 字符）注入 Agent 作历史讨论背景——下次聊同一篇论文时自然承接上次结论
- **笔记勘误**：注入的记忆可能出错——Agent 发现其与论文原文矛盾时先核实、以论文为准，并调用 `flag_note_issue` 提交勘误建议（只提议不改文件）；对话区渲染勘误卡片，用户点「更正笔记」后经现有保存通道更正（保留原句留痕，下轮问答生效）
- **专注当前论文**：不做跨对话 / 跨文档记忆召回（场景少、引入杂音，有意去掉）

### 6. 可观测性 —— 复盘

- 全链路埋点：交互 / 本地调用 / API token / 工具成败 / 图级降级原因，经 `GET /api/stats`（7/30 天窗口）聚合查询
- 工具轨迹单独落库（`agent_steps`），支持按对话回放

## 技术架构

### 技术栈

| 层 | 选型 |
| --- | --- |
| 后端 | FastAPI + uvicorn + SQLite（stdlib，WAL 模式）+ PyMuPDF |
| 编排 | LangGraph 1.x（StateGraph + AsyncSqliteSaver 断点续跑）|
| 检索 | BM25（自实现）+ 词覆盖度 + multilingual-e5-small 稠密向量 → RRF → 直接注入 Agent |
| 模型 | 云端：DeepSeek `deepseek-chat`（问答）/ vision（图表解读）；本地：Qwen3-1.7B（对话总结，零 token）|
| 数据源 | HF Papers（hf-mirror 镜像，日/周/月榜）+ arXiv |
| 前端 | Vite + React 18 + TypeScript + Tailwind v4 + Zustand + react-pdf |

### LangGraph 状态图（v0.5：单文件 `graph.py`）

每次聊天请求进入状态图（`paper_reader/graph.py`），每步状态由 checkpointer 落盘（thread_id = conversation_id，失败后传 None 输入即可从断点续跑）：

```
START → guard(规则护栏) → retrieve(三通道 RRF Top-12；零命中均匀采样)
      → agent(ReAct 内核，5 工具（含勘误提议）；注入检索上下文 + 该论文笔记记忆) → log → END
```

- **4 节点线性图**：删除相关性评分节点（grade），RRF 排序后直接注入 Agent，单次问答节省 2-5 秒
- 所有节点带降级保底（检索异常→转 Agent 工具自救；agent 异常→预检索直达问答），正常全流程 ≤5 步
- 节点经 `get_stream_writer()` 外发 `agent_step / agent_observation / delta / note_issue` 事件，前端轨迹时间线与流式协议不变

### 大小模型分工（token 净减）

| 任务 | 执行者 | API token |
| --- | --- | --- |
| 相关性排序（RRF） | 本地算法（BM25 + 词覆盖度 + e5 向量） | 0 |
| 向量编码（检索） | 本地 e5-small | 0 |
| **对话总结整理** | **本地 Qwen3-1.7B（bf16 CUDA，单例 + 串行锁）** | **0** |
| 问答最终生成 / 图表解读 | DeepSeek API | 有 |

> 设计原则：**本地模型专责「信息压缩」**（对话总结），延迟敏感度低（用户主动触发）、成本优势明显（0 Token）；所有文字生成都由 DeepSeek 承担（闲聊/问候也走同一条问答管线直答，不做本地分流）。

### Agentic RAG

1. **三通道检索**：BM25 + 词命中覆盖度 + e5 向量（跨语言：中文问 ↔ 英文论文）→ RRF 融合 Top-12；向量不可用时自动降级纯词法，词法 + 向量均零命中时均匀采样全文块兜底
2. **RRF 排序直接注入**：对 Top-12 按 RRF 分数排序，取前 6 块注入 Agent（删除 1.7B 相关性评分，响应更快）；引用护栏校验页码引用越界（前端提示）
3. **护栏**：ReAct Agent 另有 6 轮上限 / 文本工具 20s・看图 90s 超时 / 300s 整响应超时

### ReAct Agent（v4 内核，被图调用）

- **4 只读工具（全部围绕当前论文）**：`search_in_paper`（文内检索）/ `read_page`（读整页）/ `get_outline`（章节目录）/ `look_at_page`（多模态看图）；另有 **`flag_note_issue`**（笔记勘误提议）：发现注入记忆与论文矛盾且已核实时提交，只提议不写文件（用户确认后前端经现有保存通道更正）
- **注入**：短期记忆（最近消息窗口原文）+ 该论文笔记记忆 + 预检索上下文（带【第 N 页】标记）随消息提供，简单问题零工具一轮即答
- **护栏**：单工具连续失败 2 次自动弃用；达到步数/超时上限后强制基于已收集信息收敛最终回答；每步观察回填自愈

### 记忆体系（v4.2 重构）

- **短期**：每对话取最近 8 条消息原文；构建失败回退最近 10 条
- **总结整理**：右栏「整理记忆」把当前对话压缩为结构化 Markdown（讨论主题 / 核心结论 / 待跟进问题），追加到该论文笔记文件
- **论文记忆注入**：问答时读该论文笔记（最新 ≤4000 字符）注入 Agent 作背景；「总结整理」产物即下次对话的长期记忆，可随时在笔记视图编辑
- **有意不做**：跨对话/跨文档记忆召回（噪音大、场景少）；偏好画像由库内文档天然覆盖（下载入库即计入画像），无独立偏好表

### 多模态图表解读

PDF 单页渲染（PyMuPDF，最长边 ≈1200px JPEG）→ base64 data URL → DeepSeek vision 流式解读；两个入口共用同一管线：前端「解读本页」按钮（`POST /api/vision_page`）与 Agent `look_at_page` 工具。

### 降级矩阵（工程护栏）

| 环节 | 失败场景 | 兜底行为 |
| --- | --- | --- |
| 护栏 guard | 空消息 | 按问候处理 |
| 整体检索 | 异常 | 转 Agent 工具自救（标记 failed） |
| 检索零命中 | 词法 + 向量均无命中 | 均匀采样全文块兜底（防跨语言零召回） |
| Agent | 多步失败 / 超时 | 降级预检索上下文直达问答 |
| 图表解读 | 渲染或模型失败 | 转为提示文本，不中断 SSE |
| 笔记记忆读取 | 文件不存在 / 目录异常 | 返回空串，跳过注入不阻断问答 |
| 总结整理 | 模型 / 写入失败 | 返回错误提示，不影响对话 |

## 快速开始

```bash
# 1. Python 依赖（含 torch/transformers，建议 Python 3.10+）
pip install -r requirements.txt

# 2. 本地模型放入 models/（可从 HF 镜像下载；缺失时自动降级为纯云端 + 词法检索）
#    models/Qwen3-1.7B/（本地 LLM）与 models/multilingual-e5-small/（向量）
#    本地 LLM 可通过 LOCAL_LLM_NAME 切换为 models/下的其他模型目录

# 3. 配置 .env（模板已带默认值，核心是 DEEPSEEK_API_KEY）

# 4. 构建前端并启动（自动托管，访问 http://127.0.0.1:8000）
cd frontend && npm install && npm run build && cd ..
python main.py
```

- 前端开发模式：`cd frontend && npm run dev`（访问 5173，`/api` 自动代理到 8000）
- 数据持久化：文档/对话/埋点存于 `data/db/app.db`（WAL，重启不丢）；**笔记存于 `data/notes/{arxiv_id}.md`**；PDF 存 `data/docs/`；检索索引为进程内缓存，重启重建
- 未配置本地模型或 `LOCAL_LLM_ENABLED=0` 时，全部控制任务走保底路径（功能可用，token 消耗增加）

## 使用指南（首次体验路径）

1. **发现**：进入「发现」视图 → 切换 HF 日/周/月榜或搜索 arXiv → 勾选候选 → 「下载入库」
2. **阅读**：打开「文档库」→ 目录页/未读区点击论文进入阅读器；翻页缩放浏览
3. **提问**：右栏输入问题；或先在 PDF 中选中一段文字再提问（自动锚定引用）→ 观察轨迹时间线与流式回答
4. **看图**：「解读本页」按钮解读当前页图表；或在提问中要求「看看第 N 页的图」，Agent 会自主调用工具
5. **沉淀**：聊到一定程度点右栏「整理记忆」→ 对话压缩为笔记（可去「笔记」视图修改）；下次就同一篇论文提问时，该笔记自动作为背景注入

## 目录结构

```
paper-reader-langchain/
├── main.py                     # 入口：python main.py
├── requirements.txt / .env
├── models/                     # 本地模型（Qwen3-1.7B / multilingual-e5-small）
├── paper_reader/               # 后端（v0.5 合并后 10 个模块）
│   ├── __init__.py             # 版本号
│   ├── config.py               # 配置、路径与限额
│   ├── db.py                   # SQLite 存储 + 埋点统计
│   ├── sources.py              # HF 榜单（日/周/月 + 兜底缓存）/ arXiv / 个性化重排 / 下载
│   ├── documents.py            # PDF 解析分块 + 三通道检索（BM25/覆盖度/e5 向量）+ Agent 工具支持
│   ├── llm.py                  # DeepSeek 问答/图表解读 + 引用校验 + 本地 1.7B 对话总结
│   ├── notes.py                # 笔记文件存储（data/notes/*.md）+ 对话记忆（窗口/整理/注入）
│   ├── graph.py                # LangGraph：状态 + 4 节点 + 组装（checkpointer 断点续跑）
│   ├── agent.py                # ReAct 内核（4 工具 + 护栏，被图调用）
│   └── api.py                  # FastAPI 路由 + SSE + 托管前端
├── frontend/src/               # api.ts / store.ts / components（三栏 + 发现/目录/笔记视图）
├── data/                       # 运行时：db/app.db + docs/*.pdf（全部论文 PDF）+ notes/*.md + cache/（榜单兜底）
│                               # （另含 LangGraph checkpoints / writes 表：断点续跑状态）
└── 设计文档/                   # 需求与设计记录（个人资料，保留）
```

## API 一览

| 端点 | 作用 |
| --- | --- |
| `POST /api/discover` | 获取候选：`{source: hf_papers, period: daily\|weekly\|monthly}`（榜单 + 个性化重排，抓取失败回退缓存并给 `stale` 标记）或 `{source: arxiv, query}` |
| `POST /api/download` | 按候选列表或 arXiv ID 下载入库（未读区），含重试与 PDF 完整性校验 |
| `GET /api/docs?status=` | 未读 / 已读文档列表 |
| `GET /api/docs/{id}` `/file` | 文档详情 / PDF 文件流 |
| `POST /api/docs/{id}/mark_read` `/mark_unread` | 标记已读 / 退回未读 |
| `DELETE /api/docs/{id}` | 删除文档（文件 + 记录，级联对话/消息/轨迹） |
| `GET /api/catalog` | 目录卡：标题 + 摘要 + 状态 + 笔记数（按笔记文件统计）+ 对话数 |
| `POST /api/chat` | SSE 流式问答（图执行）：`{doc_id, message, selected_text?, conversation_id?}`，事件 `meta/agent_step/agent_observation/delta/done` |
| `POST /api/conversations/{id}/summarize` | 总结整理：压缩对话为结构化笔记并追加到该论文（下次讨论自动作背景） |
| `GET /api/conversations?doc_id=` / `.../{id}/messages` | 对话历史 |
| `GET /api/conversations/{id}/steps` | 会话工具轨迹（agent_steps） |
| `GET /api/notes` `/api/notes/{arxiv_id}` | 笔记列表（可 ?doc_id=）/ 详情 |
| `POST /api/notes/{arxiv_id}` / `DELETE /api/notes/{arxiv_id}` | 保存笔记（不存在自动创建）/ 删除笔记 |
| `POST /api/vision_page` | SSE 图表解读：页面渲染 → vision 流式（`meta/delta/done`） |
| `GET /api/stats?days=` | 埋点聚合（事件/模型调用/token/工具与任务细分） |
| `GET /api/health` | 健康检查（云端 LLM + 本地模型 + 多模态状态） |

## 环境变量（.env）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 无（必填） | DeepSeek Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名 |
| `DEEPSEEK_VISION_MODEL` | `deepseek-v4-flash-vision-exp` | 多模态图表解读模型 |
| `DATA_DIR` | `data/`（项目根） | 运行时数据目录（db / docs / notes / cache） |
| `LOCAL_LLM_ENABLED` | `1` | 置 0 关闭本地模型（全部走保底路径） |
| `LOCAL_LLM_NAME` / `EMBED_MODEL_NAME` | `Qwen3-1.7B` / `multilingual-e5-small` | models/ 下目录名 |
| `LOCAL_LLM_TIMEOUT` | `30` | 单次本地推理超时（秒） |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HF 镜像 |
| `PORT` | `8000` | 服务端口 |
| `MAX_RETRIES` | `3` | 网络请求重试上限 |
| `CHAT_HISTORY_TURNS` | `10` | 问答携带的历史消息条数 |
| `MAX_CONTEXT_CHARS` | `50000` | 注入上下文截断上限（字符），超限截断 |

## 版本路线

- **v2**：段落级分块 + BM25/RRF 混合检索；引用校验护栏
- **v3**：护栏式 ReAct Agent（多工具多步循环、轨迹时间线、异常降级）
- **v4**：LangGraph 编排（单轮检索评分 / 断点续跑）；1.7B 控制层 + e5 混合检索
- **v4 一期**：删除 QA 缓存系统；inline 检索上下文构建逻辑
- **v4 二期（v0.4.0）**：HF 周/月热门榜 + 抓取失败兜底缓存；关键词画像个性化推荐；笔记文件化存储
- **v4 二期修订（v0.4.1）**：修复笔记链路（路由/创建/查询/计数）；看图工具超时放宽（90s）；死代码清理
- **v4 三期（v0.4.2）**：删除闲聊分流与意图分类（5 节点线性图，小模型只做相关性评分）；Agent 工具收敛为单篇论文 4 工具；记忆体系重构（「总结整理」→ 笔记 → 自动注入，移除跨对话召回与偏好表）
- **v0.5.0**：模块合并 16→10（图三合一 graph.py；笔记 + 记忆 → notes.py；e5 向量并入 documents.py；本地 1.7B 并入 llm.py）；精排与评分合并为一次「打分即排序」（省一次本地推理）；删除 pick_context 等约 90 行旧逻辑
- **v0.5.1**：清理数据库废弃表（notes/conv_summaries/qa_cache/memories）及级联死代码，不再兼容 v4 历史数据
- **v0.5.2**：删除 1.7B 相关性评分功能（延迟高、收益低），改为纯 RRF 排序；本地模型专用于对话总结（离线任务，零 Token）；LangGraph 从 5 节点简化为 4 节点（响应更快）
- **v0.5.3**：笔记勘误闭环——注入记忆升级为「可能有误、以论文为准」；新增 `flag_note_issue` 工具（核实矛盾后提交勘误建议，只提议不写文件）+ `note_issue` 事件 + 前端勘误卡片；用户确认后经现有保存通道更正（保留原句留痕）
- **v0.5.4（当前）**：端到端测试修复——「解读本页」不再因普通问答误显「解读中…」（streamKind 区分生成类型）；笔记页「返回阅读」失效修复（改挂 store 视图切换）；ReAct 工具轮过渡语泄漏修复（新增 `retract` 事件回退误转发文本）；窄窗口布局防挤压（根容器最小宽度 + 工具栏单行横向滚动）
