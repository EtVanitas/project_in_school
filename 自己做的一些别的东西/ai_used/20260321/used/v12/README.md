# Alice 模型 - v12.0 代码简化与文档优化

## 📋 项目概述

Alice 是一个基于**动态批处理**和**向量化操作**的神经网络模型，采用独特的**Activate-Reason 双阶段架构**和**双记忆系统（Remain 短期记忆 + Trajectory 长期记忆）**。

---

## 📁 文件结构

```plaintext
c:/Users/35201/.vscode/ai/
├── alice_main.py            # 主模型（MainModel）- v12.0 简化优化
├── alice_components.py      # 核心组件（Activate, Reason, BatchActivateTester）- v12.0 简化
├── batch_pool.py           # 动态批处理池（BatchPool，GPU 存储）- v12.0 简化
├── remain_manager.py       # Remain 管理器（短期记忆，CPU 存储）- v12.0 简化
├── trajectory_recorder.py  # 轨迹记录器（长期记忆，CPU 存储）- v12.0 简化
├── config.py              # 配置管理（三阶段阈值优化）
│
├── text_embedding_lite.py # 文本嵌入预处理（BERT-base-chinese）
├── train_pretrain.py      # 预训练 Trainer（带 MUON 优化器）
│
└── README.md              # 本文档

测试数据：
├── test_data/
│   ├── test_0.txt
│   ├── test_0.jsonl
│   ├── test_1.txt
│   └── test_1.jsonl
```

---

## 🏗️ 架构设计

### 1. 核心组件关系

```
alice_main.py (MainModel)
    ↓ 使用
alice_components.py
    ├── Activate (72 个) - 计算激活分数 s = sigmoid(A * silu(X @ B))
    └── Reason (72 个) - 三种类型处理
        ├── Type1 (simple): 直接返回
        ├── Type2 (transform): 三路径融合
        └── Type3 (forget): 双输出（已移除遗忘门）
    ↓ 使用
batch_pool.py (BatchPool)
    └── 动态管理 x 数据（xs_list, labels_list, activate_indices_list）
    ↓ 使用
remain_manager.py (RemainManager)
    ├── Remain 存储（CPU 上，按 label 分组）
    └── TrajectoryRecorder（自动记录轨迹）
```

### 2. 数据流（v12.0）

```
输入 (batch_size, m, n) [GPU]
    ↓
初始化 activated_pool（act_idx=-1 表示新输入）[GPU]
    ↓
主循环（最多 MAX_ITERATIONS 轮）
    ├── Step 1: 提取新输入（act_idx=-1）[GPU]
    ├── Step 2: Activate 批量检测（复制 72 份测试所有 activate）[GPU]
    ├── Step 3: 筛选 x1（高阈值）和 x2（低阈值）[GPU]
    ├── Step 4: x1 进入激活池（带 s 值排序）[GPU]
    ├── Step 5: x2 + pending_pool → Remain 辅助
    │           ├─→ get_remains_batched() [GPU→CPU→GPU]
    │           ├─→ short_remains.detach() [切断梯度]
    │           ├─→ query_long_term_memory() [GPU→CPU→GPU]
    │           ├─→ long_remains.detach() [切断梯度]
    │           └─→ x4（成功）+ x5（失败）+ x6（长期记忆验证）[GPU]
    ├── Step 6: x4/x6 进入激活池，x5 进入待激活池 [GPU]
    ├── Step 7: update_remains_batched() [CPU 更新]
    ├── Step 8: decay_all_remains() 指数衰减 e^(-0.3) [CPU]
    ├── Step 9: Reason 批量处理（按 act_idx 分组）[GPU]
    │   ├── Type1/Type2: out → next_xs（下一轮输入）
    │   └── Type3: out → output_lists（输出），next_input → next_xs（下一轮输入）
    └── Step 10: record() 记录轨迹（x4 对应的 x2 和 remain）[GPU→CPU]
                └─→ 如果超出容量，剪枝低频记忆
    ↓
返回 output_lists（按 label 分组）和 stats
```

---

## 🔧 核心模块详解

### 1. alice_components.py - 核心组件

**三个核心组件**：

**Activate** - 计算激活分数
```python
s = sigmoid(A * silu(X @ B))
# A: (1, m), B: (n, 1) - 可学习参数
# x: (..., m, n) → s: (...)
```

**Reason** - 三种类型处理
```python
Type1 (0-7):   return x                           # 直接返回
Type2 (8-67):  三路径融合 + 残差连接              # 变换处理
Type3 (68-71): return out, next_input             # 双输出（产生输出）
```

**BatchActivateTester** - 批量激活测试
```python
forward(xs, labels, act_indices, r_high, r_low) 
→ x1_xs, x1_labels, x1_act_idxs, s_x1, 
  x2_xs, x2_labels, x2_act_idxs, s_x2
# 返回高激活 (x1) 和低激活 (x2) 两组数据
```

---

### 2. batch_pool.py - 动态批处理池

**数据结构**：
```python
xs_list: List[Tensor]        # GPU 上的 x 数据
labels_list: List[int]       # x 的归属标签  
activate_indices_list: List[int]  # activate 索引（-1 表示新输入）
```

**核心接口**：
```python
extend_with_scores(xs, labels, act_idxs, scores)  # 带 s 值排序添加
extend(xs, labels, act_idxs)                       # 直接添加
clear(device=None) → xs, labels, act_idxs          # 清空并返回
```

**特点**：
- ✅ 容量限制（max_size=5），防止显存爆炸
- ✅ s 值排序筛选（溢出时保留高质量 x）
- ✅ 完全向量化操作

---

### 3. remain_manager.py - Remain 管理器（短期记忆）

**数据结构**：
```python
remain_groups_cpu: {label: [remain_0, ..., remain_71]}  # CPU 存储
# 每个 label 独立维护 72 个 Remain 矩阵 [m, n]
```

**核心接口**：
```python
initialize_remains_for_label(label: int)           # 初始化 72 个 Remain
get_remains_batched(labels, act_idxs) → Tensor     # 批量查询 [GPU→CPU→GPU]
update_remains_batched(activated_data: Dict)       # 批量更新 [CPU]
decay_all_remains()                                # 指数衰减 e^(-0.3)
clear()                                            # 清空所有
```

**特点**：
- ✅ CPU 存储，按需传输到 GPU
- ✅ detach() 切断梯度（统计规律不参与训练）
- ✅ 每轮自动衰减（类似短期记忆的遗忘）

---

### 4. trajectory_recorder.py - 轨迹记录器（长期记忆）

**数据结构**：
```python
x2_memory_bank: List[Tensor]  # x2 特征 [m, n]
remain_bank: List[Tensor]      # 对应的 remain [m, n]
usage_count: List[int]         # 使用次数（强化机制）
# CPU 存储，最多 10 个配对
```

**核心接口**：
```python
record(x2_features, remains)                    # 记录成功的 (x2, remain) 配对
query_long_term_memory(x2_query, topk=1) → Tensor  # 基于 L2 距离检索 [GPU]
_prune_low_usage_memories()                     # 剪枝低频记忆
clear()                                         # 清空
```

**工作原理**：
1. **记录**：当 x2 + remain 成功激活为 x4 时，保存配对
2. **查询**：计算 L2 距离 + 使用次数加权，返回最相似的 remain
3. **强化**：每次被检索，usage_count+1（常用记忆被强化）

**特点**：
- ✅ 基于相似度检索（语义匹配）
- ✅ 频率驱动优化（常用优先）
- ✅ 类脑学习机制（重复强化）

---

### 5. alice_main.py - 主模型

**核心流程**：
```python
forward(initial_xs, epoch=None):
    1. 初始化 activated_pool 和 pending_pool
    2. 主循环（最多 10 轮）：
       ├── Activate 批量检测（复制 72 份测试）
       ├── x1（高激活）→ 进入激活池
       ├── x2（低激活）+ pending → Remain 辅助
       │   ├── get_remains_batched() → short_remains
       │   ├── query_long_term_memory() → long_remains
       │   └── x4（成功）/ x5（失败）/ x6（长期验证）
       ├── update_remains_batched() + decay_all_remains()
       └── Reason 批量处理（按 act_idx 分组）
           ├── Type1/Type2 → next_xs（下一轮输入）
           └── Type3 → output_lists（输出）+ next_xs
    3. 返回 output_lists 和 stats
```

**关键方法**：
```python
_process_with_remain_vectorized(...)  # x2+pending→Remain→x4,x5,x6
_process_reasons_batched(...)         # Reason 批处理
```

**梯度管理**：
```python
short_remains.detach()  # 短期记忆不携带梯度
long_remains.detach()   # 长期记忆不携带梯度
with torch.no_grad():   # 长期记忆分支完全阻断梯度
```

---

### 6. config.py - 配置管理

**配置分层**：
```python
Config.data       # 数据处理（CHUNK_SIZE, MASK_PROBABILITY 等）
Config.model      # 模型架构（M,N,P,Q, 阈值，Reason 类型分配）
Config.remain     # Remain 机制（DECAY_RATE=0.3）
Config.flow       # 推理流程（MAX_ITERATIONS, 池子容量，长期记忆超参数）
Config.loss       # 损失函数（W_OUT_INIT, W_OUT_FINAL）
Config.optimizer  # 优化器（LR, WEIGHT_DECAY）
Config.special_output  # 特殊输出（Type3 indices）
Config.train      # 训练配置（DEVICE, NUM_EPOCHS）
```

**关键配置**：
```python
# 维度
Config.model.M = 1024, N = 1024, P = 512, Q = 512

# 三阶段阈值
Stage1: r_high=0.5, r_low=0.2   # 宽松
Stage2: r_high=0.7, r_low=0.5   # 中等
Stage3: r_high=0.9, r_low=0.8   # 严格

# 流程控制
MAX_ITERATIONS = 10
MAX_ACTIVATED_X = 5
MAX_UNACTIVATED_X = 5
MAX_OUTPUT_PER_LABEL = 3

# 长期记忆
LONG_TERM_MEMORY_TOPK = 1
LONG_TERM_MEMORY_MAX_SIZE = 10
LONG_TERM_MEMORY_MIN_USAGE = 3
LONG_TERM_MEMORY_USAGE_BONUS = 0.1
```

---

## 🚀 快速开始

### 1. 基本使用

```python
import torch
from alice_main import MainModel
from config import Config

# 创建模型
model = MainModel().to('cuda')

# 准备输入
batch_size = 4
initial_xs = torch.randn(batch_size, 1024, 1024).to('cuda')

# 前向传播
output_lists, stats = model(initial_xs)

# 查看输出
print(f"迭代次数：{stats['iterations']}")
print(f"输出总数：{stats['total_outputs']}")
print(f"Label 分布：{stats['output_counts']}")
```

---

### 2. 完整训练流程

```python
from train_pretrain import PretrainTrainer, create_train_dataloader

# ========== 创建训练器 ==========
trainer = PretrainTrainer()

# ========== 准备数据 ==========
train_files = ['test_data/test_0.txt', 'test_data/test_1.txt']
train_loader = create_train_dataloader(
    file_paths=train_files,
    batch_size=2,  # 小批次，避免显存溢出
    max_chunks=None  # 全部加载
)

# 验证集（可选）
val_loader = create_train_dataloader(
    file_paths=['test_data/test_0.txt'],
    batch_size=2,
    max_chunks=10  # 只用 10 个 chunks
)

# ========== 开始训练 ==========
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=100,
    resume_from=None  # 可以指定检查点路径恢复训练
)
```

---

### 3. 自定义配置

```python
from config import Config

# 修改阈值（三阶段）
Config.model.ACTIVATE_THRESHOLDS_STAGE2 = torch.ones(72) * 0.6

# 修改当前使用阶段
Config.model.CURRENT_STAGE = 2  # 1, 2, or 3

# 修改批次大小
Config.data.BATCH_SIZE = 8

# 修改最大迭代次数
Config.flow.MAX_ITERATIONS = 50

# 修改池子容量
Config.flow.MAX_ACTIVATED_X = 10
Config.flow.MAX_UNACTIVATED_X = 10

# 修改长期记忆参数
Config.flow.LONG_TERM_MEMORY_TOPK = 3
Config.flow.LONG_TERM_MEMORY_MAX_SIZE = 100
```

---

## 📊 性能特性

- ✅ **完全向量化** - 所有操作使用掩码和批处理，无 Python for 循环
- ✅ **动态批处理池** - 按需分配显存，容量限制防止溢出
- ✅ **CPU 存储优化** - Remain 和 Trajectory 存储在 CPU，节省 GPU 显存
- ✅ **梯度隔离** - detach() 切断记忆梯度，统计规律不参与训练
- ✅ **智能检索** - 长期记忆基于 L2 距离 + 使用频率加权

---

## 📝 API 参考

### MainModel

```python
class MainModel(nn.Module):
    def __init__(self)
    def forward(self, initial_xs: torch.Tensor, epoch: int = None)
```

### BatchPool

```python
class BatchPool:
    def __init__(self, max_size: int = 5, m: int = 1024, n: int = 1024, device: str = 'cuda')
    def __len__(self) -> int
    def extend_with_scores(self, xs: torch.Tensor, labels: torch.Tensor, 
                          act_idxs: torch.Tensor, scores: torch.Tensor)
    def extend(self, xs: torch.Tensor, labels: torch.Tensor, act_idxs: torch.Tensor)
    def clear(self, device=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

### RemainManager

```python
class RemainManager:
    def __init__(self, m: int = 1024, n: int = 1024)
    def initialize_remains_for_label(label: int)
    def get_remains_batched(self, labels: torch.Tensor, act_idxs: torch.Tensor) -> torch.Tensor
    def update_remains_batched(self, activated_data: Dict)
    def decay_all_remains()
    def save_trajectory_and_clear(filepath: str)
    def clear()
```

---

## 🧪 测试与验证

### 接口兼容性测试

```bash
python quick_interface_check.py
```

### 梯度传播测试

```bash
python test_gradient_flow.py
```

### 完整功能测试

```bash
python test_interface_compatibility.py
```

---

## ⚠️ 已知问题与持续优化

#### **1. Phase 3 异步优化** ⭐⭐⭐⭐

**严重性**：🟡 **性能瓶颈** - GPU-CPU 串行执行

**现状**：
```
GPU: 计算 → 等待 Remain 更新
CPU: 等待 GPU 数据 → 更新 Remain
```

**目标**：
```
GPU: 计算 ──┬── 继续下一轮
            │
CPU: ───────┴── 后台更新 Remain
```

**实现方案**：
```python
# 使用后台线程
import threading

class AsyncRemainManager(RemainManager):
    def update_remains_async(self, data):
        thread = threading.Thread(target=self._update_worker, args=(data,))
        thread.start()
```

**预期收益**：
- 加速 30-50%
- GPU 利用率提升

**计划版本**：v13.0

---

#### **2. 混合精度训练 (AMP)** ⭐⭐⭐⭐

**严重性**：🟡 **性能瓶颈** - FP32 计算较慢

**目标**：
```python
from torch.cuda.amp import autocast, GradScaler

with autocast():
    outputs, stats = model(inputs)
    loss = compute_loss(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**注意事项**：
- Remain 保持 FP32（保证长期记忆质量）
- 轨迹记录保持 FP32

**预期收益**：
- 提速 2-3 倍
- 显存占用减少 50%

**计划版本**：v13.0

---

### 🟢 **持续监控（低优先级）**

#### **3. 梯度检查点** ⭐⭐⭐

**状态**：🟢 **监控中** - 按需实现

**场景**：当 batch_size 需要进一步扩大时启用

**实现**：
```python
from torch.utils.checkpoint import checkpoint

def custom_forward(model, x):
    return model(x)

outputs = checkpoint(custom_forward, model, inputs)
```

**权衡**：
- ✅ 节省显存，可处理更大 batch
- ❌ 时间换空间（慢 20-30%）

**触发条件**：batch_size > 8 或显存不足

---

#### **4. 动态阈值调整** ⭐⭐

**状态**：🟢 **监控中** - Curriculum Learning 思想

**方案**：
```python
# 根据训练进度自动调整
progress = epoch / total_epochs
r_high = 0.5 + 0.4 * progress  # 从 0.5 逐渐增加到 0.9
```

**效果**：
- 早期宽松，后期严格
- 类似 Curriculum Learning

**触发条件**：发现训练不稳定或收敛困难

---

#### **5. 可视化与监控** ⭐⭐

**状态**：🟢 **规划中** - 提升可解释性

**功能**：
- TensorBoard 集成
- Remain 矩阵可视化
- 激活分布统计
- 轨迹分析工具

**触发条件**：需要深入分析模型行为

---

### 🔮 **探索性方向**

#### **6. Remain 压缩** ⭐⭐

**方案**：
- 低秩分解（PCA/随机投影）
- 量化（FP16/INT8）
- 稀疏化

**预期收益**：
- 显存占用减少 50-75%
- 传输速度提升 2-4 倍

**风险**：
- 可能影响长期记忆质量

---

#### **7. 分层衰减** ⭐⭐

**方案**：
```python
# 不同 act_idx 使用不同衰减率
decay_rates = {
    'important': 0.1,  # 重要记忆衰减慢
    'normal': 0.3,     # 普通记忆
    'transient': 0.6   # 次要记忆衰减快
}
```

**依据**：
- 类脑记忆巩固机制
- 重要记忆衰减慢，次要记忆衰减快

---

#### **8. 元学习** ⭐

**方案**：
- Learn to Initialize Remains
- 从历史轨迹学习初始化策略

**长期愿景**：
- 新 label 的 Remain 不再是随机初始化
- 基于相似 label 的历史数据智能初始化

---

## 🎯 总结

### v12.0 核心成果

✅ **已完成** - 13 项优化全部完成（详见「⚠️ 已知问题与持续优化」）
- Reason 遗忘门简化
- 代码简化与文档优化
- output_lists 容量限制
- RemainManager 自动初始化
- BatchPool 设备转换优化
- 池子容量动态调整
- 阈值配置优化
- Remain 衰减优化
- 激活次数惩罚优化
- 长期记忆梯度路径明确
- **GPU 硬编码修复**（v12.0 新增）
- **Activate 可训练化**（v12.0 新增）
- **多 GPU 数据并行**（v12.0 新增）

🟡 **待优化** - 2 个中优先级方向
- Phase 3 异步优化（v13.0）
- 混合精度训练（v13.0）

🟢 **持续监控** - 6 个低优先级方向
- 梯度检查点、动态阈值调整、可视化与监控
- Remain 压缩、分层衰减、元学习

### 核心优势

1. **高性能**：完全向量化，无 Python for 循环
2. **低显存**：动态批处理池 + 小批次训练，适配消费级显卡
3. **可训练**：Reason 和 Activate 都可以正常训练
4. **易扩展**：模块化设计，接口清晰，支持多 GPU 并行
5. **自动化**：轨迹自动记录，智能保存策略
6. **稳定性**：13 项优化完成，系统稳定可靠
7. **兼容性**：支持 CPU/GPU 自动降级，适应不同环境

---
