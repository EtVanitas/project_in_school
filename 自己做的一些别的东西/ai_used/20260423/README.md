# Qwen3.5-2B 动态路径微调系统

基于强化学习的动态 Transformer 层选择系统，通过智能跳过不必要的计算层来提升推理效率。

---

## 📋 目录

- [架构概述](#架构概述)
- [阶段 0：数据准备](#阶段-0数据准备)
- [阶段 1A：批量路径搜索](#阶段-1a批量路径搜索)
- [阶段 1B：模型训练](#阶段-1b模型训练)
- [阶段 2：在线联合训练](#阶段-2在线联合训练)
- [快速开始](#快速开始)
- [文件结构](#文件结构)

---

## 架构概述

### 核心思想

传统 Transformer 必须顺序经过所有层，但某些样本可能只需要部分层就能得到准确预测。本系统通过强化学习动态选择最优的计算路径：

```
输入文本 → 前9层(固定) → RL策略网络选择 → 动态跳过中间层 → 输出
```

### 三阶段训练流程

```
Stage 0: 文本分块 + 基准计算
    ↓
Stage 1A: 批量路径搜索（无梯度探索）
    ↓  
Stage 1B: LoRA微调 + RL网络训练（有梯度优化）
    ↓
Stage 2: 策略引导的在线联合训练（DFS + 动态分支）
```

---

## 阶段 0：数据准备

### 目标

将原始文本转换为模型可处理的 token 序列，并预计算每个位置的基准概率分布。

### 输入

- **原始文本**: `wiki_zh_2019/wiki_zh_2019.txt`
- **模型**: Qwen3.5-2B (本地路径)

### 处理流程

#### Step 1: 文本分块 (`utils/text_splitter.py`)

```python
# 按长度分组存储
trajectory_dir/texts/
├── len_128/
│   ├── sample_0.pt    # [128] token IDs
│   ├── sample_1.pt
│   └── ...
├── len_256/
│   ├── sample_0.pt    # [256] token IDs
│   └── ...
└── ...
```

**特点:**
- 相同长度的文本放在同一文件夹
- 便于后续批处理（无需 padding）
- 文件名格式: `sample_{n}.pt`

#### Step 2: 基准计算 (`trainer/stage0.py`)

对每个文本块，使用完整模型前向传播，记录最后一个 token 的 top-100 logits：

```python
trajectory_dir/bases/
├── len_128/
│   ├── sample_0.pt    # {'top_indices': [100], 'top_logits': [100]}
│   └── ...
└── ...
```

**用途:**
- 作为"教师分布"，用于计算交叉熵损失
- Stage1A 中评估路径质量的基准

### 运行

```bash
python main.py --stage0
```

或使用批量脚本（支持断点续传）：

```bash
python run_stage0_batch.py
```

---

## 阶段 1A：批量路径搜索

### 目标

为每个样本找到最优的层选择路径，并收集训练数据。

### 核心创新：真正的 GPU 批处理

**旧方法（慢）:**
```python
for sample in batch:          # 4次循环
    for path in paths:         # 10次循环
        loss = forward(sample, path)  # 单样本前向
```

**新方法（快 3-4x）:**
```python
batch_input_ids = stack(samples)  # [4, seq_len]
for path in paths:                 # 10次循环
    losses = forward(batch, path)  # 批处理前向 [4]
```

### 处理流程

#### 1. 随机采样批次

从某个长度文件夹（如 `len_128/`）中随机选择 `batch_size=4` 个样本：

```python
samples = random.sample(all_files, batch_size)
# 例如: ['sample_5.pt', 'sample_12.pt', 'sample_28.pt', 'sample_35.pt']
```

#### 2. 多路径探索

对每个样本采样 `num_exploration_paths=10` 条路径：

```python
paths = [
    [0,1,...,8, 9,10,11,...,23, 15],   # path_0
    [0,1,...,8, 9,12,15,...,23, 15],   # path_1
    ...
]
```

**路径格式:**
- 前 9 层固定: `[0,1,...,8]`
- 动态层: 从 15 个可选层中选择
- 结束动作: `15` (输出)

#### 3. 批处理前向传播

对每条 path，4 个样本**同时**经过模型：

```python
# 嵌入层批处理
hidden = embed_tokens(batch_input_ids)  # [4, seq_len, 2048]

# 逐层批处理
for layer_idx in path:
    hidden = layers[layer_idx](hidden)  # [4, seq_len, 2048]

# 计算每个样本的 loss
for sample_idx in range(4):
    logits = lm_head(hidden[sample_idx, -1, :])
    ce_loss = cross_entropy(logits, base_probs[sample_idx])
```

**关键优化:**
- Embedding、Transformer 层、lm_head 全部批处理
- 只在最后计算 loss 时逐个样本（因为每个样本的 base_probs 不同）

#### 4. 独立最优路径选择

每个样本**独立**找出自己的最小 loss path：

```python
# sample_0: losses = [0.5, 0.4, 0.6, ...] → best_path = path_1
# sample_1: losses = [0.6, 0.9, 0.5, ...] → best_path = path_2
# sample_2: losses = [0.7, 0.6, 0.8, ...] → best_path = path_1
# sample_3: losses = [0.8, 0.7, 0.9, ...] → best_path = path_1
```

**注意:** 不同样本的最优 path 可能不同！

#### 5. 三元组收集

为**所有路径**（不只是最优）计算 (state, action, value) 三元组：

```python
for each sample:
    for each path (10 paths):
        trajectories = collect_states_actions(path)
        triples = compute_values(trajectories, ce_loss, penalty)
        save(triples)  # 保存到 trajectories/
```

**为什么收集所有路径？**
- 高奖励路径：告诉策略网络什么动作是好的
- 低奖励路径：告诉策略网络什么动作是不好的
- 对比学习需要正负样本

#### 6. 路径记录

保存每个 path 对应的样本列表：

```json
// trajectory_dir/path_records.json
{
  "(0,1,...,8,9,10,...,23,15)": [
    {"len_key": "len_128", "sample_name": "sample_5"},
    {"len_key": "len_128", "sample_name": "sample_28"},
    {"len_key": "len_256", "sample_name": "sample_3"}
  ],
  "(0,1,...,8,9,12,...,23,15)": [
    {"len_key": "len_128", "sample_name": "sample_12"}
  ]
}
```

**用途:** Stage1B 可以直接按 path 分组进行批处理训练

#### 7. 轨迹批量保存

所有三元组先存入内存缓冲区，Epoch 结束时一次性保存：

```
trajectory_dir/trajectories/
├── batch_000000.pt    # 包含多个三元组
├── batch_000001.pt
└── ...
```

**优势:**
- I/O 速度提升 10-50x
- 减少文件系统压力
- Stage1B 自动兼容新旧格式

### 输出

1. **path_records.json**: `{path_tuple: [sample_info, ...]}`
2. **trajectories/**: 批量保存的三元组文件（`batch_*.pt`）

### 运行

```bash
python main.py --stage1a
```

---

## 阶段 1B：模型训练

### 目标

1. 使用 path_records 训练 LoRA（微调模型以适应动态路径）
2. 使用 trajectories 训练 Actor-Critic（学习路径选择策略）

### 训练流程

#### Part 1: LoRA 微调

**数据组织:**
```python
# 按 path 分组
for path, samples in path_records.items():
    # 再按长度分组（相同长度才能批处理）
    for length, sample_names in group_by_length(samples):
        # 加载该组的所有样本
        batch_data = load_samples(length, sample_names)
        
        # 批处理前向+反向传播
        loss = forward_batch(batch_data, path)
        loss.backward()
        optimizer.step()
```

**关键优势:**
- 相同 path + 相同长度的样本一起训练
- 真正的 GPU 批处理（无需 padding）
- 更高效地利用计算资源

#### Part 2: Actor-Critic 训练

**状态表示改进:**

旧方法（信息不足）:
```python
state = hidden_state  # [2048] 抽象语义表示
```

新方法（信息丰富）:
```python
# Stage1B 中预处理
logits = lm_head(hidden_state)           # [vocab_size]
vocab_probs = softmax(logits)            # [248320] 完整的词汇分布

# Networks 中处理
vocab_proj = MLP(vocab_probs)            # 248320 → 2048 → 1024 → 1024
combined = concat(vocab_proj, layer_onehot, step_idx, context)  # [1041]
features = fusion_network(combined)      # 1041 → 256 → 64
```

**优势:**
- 词表概率包含完整的预测置信度信息
- 比 2048 维隐藏状态更容易学习策略

**训练过程:**
```python
# 从 trajectories/ 读取三元组
triples = load_all_trajectories()

# 分批训练
for batch in batches(triples):
    # 预处理：hidden → vocab_probs
    hidden_states = stack([t['state']['hidden'] for t in batch])
    with torch.no_grad():
        logits = lm_head(hidden_states)
        vocab_probs = softmax(logits)
    
    # Actor-Critic 前向
    action_logits, values = actor_critic(
        vocab_probs=vocab_probs,
        layer_index=batch.layer_indices,
        context_length=batch.context_lengths,
        step_idx=batch.step_idxs
    )
    
    # 计算损失
    actor_loss = policy_gradient(action_logits, batch.actions, advantages)
    critic_loss = MSE(values, batch.target_values)
    total_loss = actor_loss + 0.5 * critic_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
```

### 模型保存

始终保存两个版本：

```
models/stage1a/
├── lora_latest.pt      # 最新模型
├── lora_latest.json
├── lora_best.pt        # 最优模型（loss 最低）
└── lora_best.json

models/stage1b/
├── rl_latest.pt        # 最新 Actor-Critic
├── rl_latest.json
├── rl_best.pt          # 最优 Actor-Critic
└── rl_best.json
```

### 数据清理

训练完成后自动删除：
- `trajectory_dir/path_records.json`
- `trajectory_dir/trajectories/`

**原因:** 数据已过时，下次 Stage1A 会生成新的

### 运行

```bash
python main.py --stage1b
```

---

## 阶段 2：在线联合训练

### 目标

在 Stage1 的基础上，使用策略网络引导的动态路径搜索，进一步优化模型和策略。

### 核心特点

**与 Stage1 的区别:**
- Stage1: 离线训练（预先生成所有路径）
- Stage2: 在线训练（实时探索 + 即时优化）

**工作流程:**
1. **DFS 探索**: 使用 Actor-Critic 动态采样动作，产生分支
2. **两阶段前向**:
   - 阶段1: 无梯度探索所有路径，找到最优路径
   - 阶段2: 沿最优路径有梯度前向+反向传播
3. **联合优化**: 同时更新 LoRA 和 Actor-Critic

### 处理流程

#### 1. 加载样本

随机选择一个文本块（不需要基准数据）

#### 2. DFS 路径探索

```python
# 使用策略网络动态选择动作
action_probs = actor_critic(vocab_probs, ...)
action = sample(action_probs)

# 判断是否分支
if should_branch(action_probs):
    # 为每个分支创建新路径
    for branch_action in branch_actions:
        create_new_path(branch_action)
```

**分支条件:**
- 第二大概率 > 0.2
- 第二大概率 / 最大概率 > 0.8

#### 3. 选择最优路径

从所有完成的路径中，选择交叉熵损失最小的路径

#### 4. 有梯度训练

沿最优路径重新进行前向传播（带梯度），然后反向传播更新 LoRA

#### 5. 收集轨迹并训练 RL

计算三元组价值，分批训练 Actor-Critic

#### 6. 数据保存

每个文本处理完后，将所有路径的三元组批量保存到一个文件：

```
trajectory_dir/stage2_trajectories/
├── text_000000.pt    # 包含该文本的所有路径三元组
├── text_000001.pt
└── ...
```

### 模型存储

Stage2 有独立的存储目录，不覆盖 Stage1：

```
models/
├── stage1a/          # Stage1 LoRA
├── stage1b/          # Stage1 RL
├── stage2a/          # Stage2 LoRA (未来拆分)
└── stage2b/          # Stage2 RL (未来拆分)
```

**当前实现:**
- Stage2 读取 Stage1 的最优模型作为起点
- Stage2 保存自己的最优和最新模型
- 完全隔离，互不影响

### 运行

```bash
python main.py --stage2
```

---

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+ with CUDA
- Transformers 4.35+

### 安装依赖

```bash
pip install torch transformers packaging
```

### 完整流程

```bash
# 1. 数据准备（一次性）
python main.py --stage0

# 2. 路径搜索
python main.py --stage1a

# 3. 模型训练
python main.py --stage1b

# 4. 在线联合训练（可选）
python main.py --stage2
```

### 测试模型加载

```bash
python main.py --test
```

---

## 文件结构

```
c:\Users\35201\.vscode\ai\
├── config.py                    # 配置类（ModelConfig, LoRAConfig, RLConfig）
├── main.py                      # 主入口
├── run_stage0_batch.py          # Stage0 批量处理脚本
│
├── models/
│   ├── __init__.py
│   ├── model_loader.py          # LayeredModelLoader（分层加载模型）
│   └── networks.py              # LoRA + MLPActorCritic 网络定义
│
├── trainer/
│   ├── __init__.py
│   ├── data_manager.py          # 🆕 统一数据管理器（TrainingDataManager）
│   ├── stage0.py                # Stage0: 基准计算
│   ├── stage1a.py               # Stage1A: 批量路径搜索
│   └── stage1b.py               # Stage1B: LoRA+RL训练
│
├── utils/
│   ├── __init__.py
│   ├── text_splitter.py         # 文本分块工具
│   ├── path_generator.py        # 路径生成器
│   └── reward.py                # 奖励计算（三元组生成）
│
├── trajectory_dir/              # 训练数据目录
│   ├── texts/                   # Token IDs（Stage0 输出）
│   │   ├── len_128/
│   │   │   ├── sample_0.pt
│   │   │   └── ...
│   │   └── ...
│   ├── bases/                   # 基准数据（Stage0 输出）
│   │   ├── len_128/
│   │   │   ├── sample_0.pt
│   │   │   └── ...
│   │   └── ...
│   ├── path_records.json        # 路径记录（Stage1A 输出）
│   ├── trajectories/            # Stage1A 三元组（批量保存）
│   │   ├── batch_000000.pt
│   │   └── ...
│   └── stage2_trajectories/     # Stage2 三元组（按文本保存）
│       ├── text_000000.pt
│       └── ...
│
├── models/                      # 训练模型存储
│   ├── stage1a/                 # Stage1 LoRA 权重
│   │   ├── lora_latest.pt
│   │   └── lora_best.pt
│   ├── stage1b/                 # Stage1 Actor-Critic 权重
│   │   ├── rl_latest.pt
│   │   └── rl_best.pt
│   ├── stage2a/                 # Stage2 LoRA (预留)
│   └── stage2b/                 # Stage2 RL (预留)
│
└── wiki_zh_2019/                # 原始数据
    └── wiki_zh_2019.txt
```

---

## 🆕 TrainingDataManager 数据管理器

### 概述

`TrainingDataManager` 是一个统一的数据管理组件，用于管理 Stage1A/Stage1B/Stage2 的所有数据存取操作。

**核心优势：**
- ✅ **统一管理**：所有数据存取逻辑集中在一个文件中
- ✅ **目录隔离**：各阶段使用独立的输出目录，避免冲突
- ✅ **向后兼容**：自动检测并兼容新旧数据格式
- ✅ **智能加载**：支持回退机制，优先加载当前阶段模型

### 主要功能

#### 1. 数据加载
```python
from trainer import TrainingDataManager

data_manager = TrainingDataManager(model_config, lora_config, rl_config)

# 批量加载样本
batch_input_ids, batch_base_data, sample_names = \
    data_manager.load_batch_data(sample_files, text_folder, base_folder, device)

# 加载所有轨迹
all_trajectories = data_manager.load_all_trajectories()
```

#### 2. LoRA 模型管理
```python
# 保存最优模型
data_manager.save_lora_model(layered_loader, loss, stage='stage1a', is_best=True)

# 智能加载（优先当前阶段，回退到上一阶段）
data_manager.try_load_best_lora_model(
    layered_loader,
    current_stage='stage2a',
    fallback_stage='stage1b'
)
```

#### 3. RL 网络管理
```python
# 保存最优模型
data_manager.save_rl_model(actor_critic, optimizer, epoch, loss,
                          stage='stage1b', is_best=True)

# 智能加载
checkpoint = data_manager.try_load_best_rl_model(
    actor_critic, optimizer,
    current_stage='stage2b',
    fallback_stage='stage1b'
)
```

#### 4. 轨迹和路径记录
```python
# 保存轨迹批次
data_manager.save_trajectory_batch(all_triples, batch_idx)

# 保存路径记录
data_manager.save_path_records(path_records)
```

### 目录结构

```
models/
├── stage1a/              # Stage1A LoRA 模型
│   ├── lora_latest.pt
│   └── lora_best.pt
├── stage1b/              # Stage1B RL 模型
│   ├── rl_latest.pt
│   └── rl_best.pt
├── stage2/               # Stage2A LoRA 模型
│   ├── lora_latest.pt
│   └── lora_best.pt
└── stage2b/              # Stage2B RL 模型
    ├── rl_latest.pt
    └── rl_best.pt
```

### 文档

- 📖 [使用指南](trainer/DATA_MANAGER_USAGE.md) - 详细的 API 文档
- 📋 [迁移计划](trainer/MIGRATION_PLAN.md) - 分步骤迁移指南
- ⚡ [快速参考](trainer/QUICK_REFERENCE.md) - 常用 API 速查
- 📊 [总结文档](trainer/TRAINING_DATA_MANAGER_SUMMARY.md) - 完整概述

---

## 关键技术细节

### 批处理维度

| 场景 | Batch 组成 | 是否需要 Padding |
|------|-----------|-----------------|
| Stage1A 路径搜索 | 同长度文件夹随机采样 4 个 | ❌ 不需要（长度相同） |
| Stage1B LoRA 训练 | 同 path + 同长度的样本 | ❌ 不需要（已分组） |
| Stage1B RL 训练 | 固定 batch_size=4 的三元组 | ❌ 不需要（张量堆叠） |

### 显存优化

1. **梯度检查点**: LoRA 训练时启用
2. **无梯度探索**: Stage1A 路径搜索时 `torch.no_grad()`
3. **及时清理**: Stage1B 训练完立即删除临时数据
4. **bfloat16**: 全程使用 bf16 精度

### 性能加速

- **Stage1A**: 3-4x 加速（真批处理 vs 伪批处理）
- **Stage1B LoRA**: 3x 加速（同 path 批处理）
- **lm_head 优化**: 1.5-2x 加速（批量计算 logits）

---

## 常见问题

### Q: Stage1A 为什么要收集所有路径的三元组？

A: Actor-Critic 需要通过对比好坏样本来学习。如果只保留最优路径，网络无法知道哪些动作是不好的。

### Q: 为什么状态表示要用词表概率而不是隐藏状态？

A: 248320 维的词表概率包含了模型对每个 token 的完整预测置信度，比 2048 维的抽象隐藏状态包含更多信息，更容易学习有效的策略。

### Q: Stage1B 训练完后为什么要删除数据？

A: 这些数据是基于旧的 LoRA 权重生成的，继续使用会导致分布偏移。下次运行 Stage1A 会生成与新权重匹配的数据。

---

## 下一步

- [ ] Stage 2: 在线联合训练（待实现）
- [ ] 推理部署：使用训练好的策略网络动态选择路径
- [ ] 性能评估：对比固定路径 vs 动态路径的效率和准确率

---

## 许可证

本项目仅供研究使用。
