# RNN策略引导的动态路径微调系统

## 使用说明

只打包了文件，没有打包模型和数据集，如果有这两个并且更改一下config里的路径地址，理论上运行main.py就可以正常训练了。为了显示出运行信息，我在代码中添加了一些调试信息，用以输出当前的最佳路径，这样就可以看看模型走了哪条路。要运行这个文件还需要数据集和模型，对于数据集，由于data_loader文件只提取input和output对，所以换成别的数据集应该用不了，除非它也是jsonl格式的文件并且包含input和output两个集合项。对于模型，只要是Qwen3系列的模型应该都能用，因为使用的是Qwen3的模型加载器和RoPE嵌入。原本是打算用Qwen3.5-2B做这个大作业的，但是训练后发现效果并不是很好，因为Qwen3.5-2B还有视觉层和5层mtp，使用的注意力大部分还是线性注意力，且只有24层网络，这导致跳跃一些层时可能会出现问题，最后还是换成了现在的Qwen3-1.7B，它有28层transformer解码器，而且使用的都是标准的注意力机制，训练起来就不容易出错，而且跳跃1-2层带来的损失也更小。由于是用词表作为强化学习网络的输入，词表维度很大，虽然加入了1%的概率过滤使得每次最多只有100个非零输入，但还是很难收敛，而且由于没有用批训练，可能要训练很久。即使中间用了梯度检查点，在我的笔记本上训练时单个问题和回答的上下文长度也不能超过8000个token，否则显存依旧会超过12GB，如果是24GB的显卡应该能好很多。数据集内有很多output很长的问题，最长的能有二十几万的token长度，我只筛选了那些长度小于8000的来训练，如果要用长的，可能要进行手动裁剪。最后就是为了运行时不中断，我设置了断点续训，每训练完几个token就保存一次LoRA和强化学习网络的参数，具体模型会保存在在trainer文件夹下的models里。下面的README文件内容都是AI帮忙写的，基本涵盖了代码的主要逻辑，可以当作word报告的简化模式看，为了简洁，word里只写了伪代码，而且去掉了清理显存，输出信息这种逻辑，只保留了主要的逻辑，word里没写的其它代码逻辑基本都写在下面了。

## 📋 项目概述

本项目实现了一个创新的**RNN策略引导的动态路径微调框架**，将强化学习（RL）与LoRA微调相结合，用于优化Qwen3-1.7B大语言模型的推理路径选择。

### 核心思想

传统的Transformer模型按顺序执行所有层，而本系统引入**动态路径选择机制**：
- **前9层固定执行**：提取基础语义特征
- **后19层动态跳转**：由LSTM策略网络决定跳到哪一层或直接输出
- **Actor-Critic架构**：通过强化学习训练策略网络，优化路径选择以最大化预测准确率

---

## 🏗️ 系统架构

### 整体流程图

```
输入文本 → Tokenization → 前9层固定执行 → LSTM策略网络决策 → 动态路径执行 → 输出预测
                                    ↓
                              Actor: 选择动作(跳层/输出)
                              Critic: 评估状态价值
                                    ↓
                              RL训练更新策略网络
                                    ↓
                              LoRA训练更新模型权重
```

### 三层架构设计

#### 1. **基础模型层 (Qwen3-1.7B)**
- **模型**: Qwen3-1.7B (28层Transformer)
- **词表大小**: 151,936
- **隐藏层维度**: 2048
- **注意力机制**: GQA (16个Q头, 8个KV头)
- **位置编码**: RoPE (Rotary Position Embedding)

#### 2. **LoRA适配层**
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **LoRA秩 (r)**: 8
- **Alpha**: 16.0
- **Dropout**: 0.1
- **可训练参数占比**: ~0.5%

#### 3. **RL策略层 (LSTM Actor-Critic)**
- **输入**: 过滤后的词表概率分布 [151,936] + 层索引one-hot [20]
- **网络架构**: 
  - Prob投影: [151936] → [256]
  - 残差分割: [256] → [128前] + [128后]
  - 拼接降维: [256+20] → [128]
  - LSTM处理: [128] → [128] (单层单向)
  - 特征融合: [128 LSTM + 128残差 + 20层索引] = [276]
  - Actor头: [276] → [64] → [20] (动作logits)
  - Critic头: [276] → [64] → [20] → [1] → Sigmoid (状态价值)
- **动作空间**: 20个动作 (19个跳层动作 + 1个输出动作)

---

## 🔄 训练流程

### 训练阶段详解

#### Stage 1: 路径探索 (无梯度)
```python
for each token in output:
    1. 执行前9层固定层
    2. while path_length < max_steps:
        a. 计算当前层的词表概率分布
        b. 过滤低概率token (<1%)
        c. LSTM策略网络采样动作
        d. 记录rollout选项 (如果现在输出会怎样)
        e. 执行动作 (跳层或输出)
    3. 选择reward最大的路径作为最优路径
```

**关键特性**:
- **Rollout机制**: 在每个时间步记录"如果现在输出"的reward
- **概率过滤**: 只保留概率>1%的token作为RNN输入，减少噪声
- **Reward定义**: target_token的预测概率 (越高越好)

#### Stage 2: LoRA训练 (有梯度)
```python
沿最优路径执行前向传播:
    for action in best_path:
        if action == OUTPUT:
            计算CE Loss并返回
        else:
            使用梯度检查点执行目标层
    
反向传播更新LoRA参数
```

**优化技术**:
- **梯度检查点**: 不保存中间激活值，节省显存
- **即时更新**: 每个token处理后立即更新LoRA

#### Stage 3: RL训练 (异步更新)
```python
每处理N个token (默认100):
    for trajectory in buffer:
        重置LSTM hidden state
        for t in trajectory:
            # Actor更新
            log_prob = log_softmax(action_logits)
            advantage = target_value - predicted_value
            actor_loss = -log_prob * advantage
            
            # Critic更新
            critic_loss = MSE(predicted_value, target_value)
        
        反向传播更新Actor-Critic网络
```

**价值计算**:
```python
target_value = final_reward - step_penalty × remaining_steps
target_value = clip(target_value, 0, 1)  # 裁剪到[0,1]
```

---

## 📊 数据流与维度

### 状态表示
```
vocab_probs (softmax输出)     [151936]
    ↓ 过滤 (<1%设为0)
vocab_probs_filtered          [151936] (稀疏)
    ↓ 投影
prob_features                 [256]
    ↓ 分割
prob_front [128] + prob_back [128]

layer_index (Python int)      标量
    ↓ one-hot编码
layer_onehot                  [20]

拼接: prob_features + layer_onehot = [276]
    ↓ 降维
lstm_input                    [128]
    ↓ LSTM
lstm_context                  [128]

融合: lstm_context + prob_back + layer_onehot = [276]
    ↓ Actor/Critic头
action_logits                 [20]
state_value                   [1] (Sigmoid, 范围[0,1])
```

### 动作映射
```
动作索引 0-18: 跳到 layer 9-27
动作索引 19:   输出 (ACTION_OUTPUT)

示例:
  Layer 8  → one-hot[0] → action 0  (跳到layer 9)
  Layer 9  → one-hot[1] → action 1  (跳到layer 10)
  ...
  Layer 26 → one-hot[18] → action 18 (跳到layer 27)
  Layer 27 → one-hot[19] → action 19 (输出)
```

---

## ⚙️ 配置参数

### PathConfig (路径配置)
```python
project_root: "c:\Users\35201\.vscode\ai"
qwen_model_path: "Qwen3-1.7B"
models_dir: "trainer/models"
jsonl_dataset_path: "Superior-Reasoning-SFT-gpt-oss-120b-stage1-train-data.jsonl"
checkpoint_dir: "trainer/checkpoints"
```

### ModelConfig (模型配置)
```python
num_layers: 28              # 总层数
fixed_layers: 9             # 固定层数 (前9层)
dynamic_layers: 19          # 动态层数 (后19层)
hidden_size: 2048           # 隐藏层维度
vocab_size: 151936          # 词表大小
max_steps: 30               # 最大推理步数
dtype: "bfloat16"           # 数据类型
batch_size: 1               # 批大小 (逐样本训练)
```

### LoRAConfig (LoRA配置)
```python
r: 8                        # LoRA秩
alpha: 16.0                 # 缩放系数
dropout: 0.1                # Dropout概率
target_modules: [           # 目标模块列表
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
lr: 1e-4                    # 学习率
betas: (0.9, 0.95)          # AdamW优化器beta参数
weight_decay: 0.01          # 权重衰减
grad_clip: 1.0              # 梯度裁剪阈值
```

### RLConfig (强化学习配置)
```python
# 状态表示
vocab_size: 151936          # 词表大小
layer_encoding_dim: 20      # 层索引one-hot维度

# 网络架构
prob_proj_dim: 256          # 概率向量投影维度
lstm_hidden_dim: 128        # LSTM隐藏层维度
num_layers: 1               # LSTM层数（单层）
bidirectional: False        # 单向LSTM

# 动作空间
action_dim: 20              # 动作数量 (19跳层 + 1输出)

# Actor-Critic头
fusion_dim: 276             # 融合维度 (128+128+20)
actor_hidden_dim: 64        # Actor隐藏层
critic_hidden_dim: 64       # Critic隐藏层

# 优化器
lr: 1e-4                    # 学习率
betas: (0.9, 0.95)          # AdamW优化器beta参数
weight_decay: 0.01          # 权重衰减
grad_clip: 1.0              # 梯度裁剪阈值

# 奖励计算
step_penalty: 0.01          # 每步惩罚系数

# 状态过滤
prob_threshold: 0.01        # 词表概率过滤阈值 (1%)

# 训练控制
rl_update_interval: 100     # RL更新间隔 (每N个token)
```

---

## 📁 项目结构

```
ai/
├── config.py                    # 配置管理 (PathConfig, ModelConfig, LoRAConfig, RLConfig)
├── main.py                      # 主入口 (显示配置/启动训练)
├── trainer.py                   # 训练器 (Trainer类，整合LoRA+RL训练)
│
├── models/
│   ├── __init__.py
│   ├── model_loader.py         # LayeredModelLoader (加载Qwen3-1.7B并添加LoRA)
│   ├── model_manager.py        # ModelManager (模型保存/加载)
│   └── networks.py             # LoRALinear, LoRAModuleWrapper, RNNActorCritic
│
├── utils/
│   ├── __init__.py
│   ├── components.py           # RL训练函数、梯度前向传播、RoPE计算
│   └── data_loader.py          # DataLoader (JSONL数据集加载)
│
├── trainer/
│   └── models/                 # 模型保存目录
│       ├── lora_latest.pt      # 最新LoRA模型
│       └── rl_best.pt          # 最优RL模型
│
├── Qwen3-1.7B/                 # Qwen3-1.7B模型文件
└── Superior-Reasoning-SFT-gpt-oss-120b-stage1-train-data.jsonl  # 训练数据集
```

---

## 🚀 使用方法

### 1. 查看配置信息
```bash
python main.py
```

输出示例:
```
======================================================================
RNN策略引导的动态路径微调 (Actor-Critic)
======================================================================

模型配置:
  路径：c:\Users\35201\.vscode\ai\Qwen3-1.7B
  层数：28 (前9固定 + 后19动态)
  隐藏层：2048
  词表：151,936

LoRA 配置:
  秩：8
  Alpha: 16.0
  Target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

RL 配置:
  状态维度: vocab_probs(151936→256投影) + LSTM(128) + layer(20) = 276融合
  架构: LSTM (128->128, 1层单向)
  动作空间: 20 (19个跳转目标9-27 + 1个输出)
  Actor头: 276 -> 64 -> 20
  Critic头: 276 -> 64 -> 20 -> 1
  学习率：0.0001
  奖励计算：交叉熵损失反向推导，步数惩罚 -0.01/步

训练配置:
  设备：cuda
  Epochs: 10 (默认)
  Batch size: 1 (逐样本训练)
```

### 2. 开始训练
```bash
python main.py --train
```

训练流程:
1. 加载Qwen3-1.7B模型并添加LoRA
2. 初始化RNN Actor-Critic网络
3. 加载训练数据集 (JSONL格式)
4. 逐个样本进行Teacher Forcing训练
5. 每100个token更新一次RL网络
6. 每5个样本保存checkpoint

### 3. 断点续训
系统会自动检测已保存的模型：
- 优先加载 `trainer/models/rl_best.pt` (最优RL模型)
- 其次加载 `trainer/models/lora_latest.pt` (最新LoRA模型)
- 如果没有找到模型，则使用随机初始化

---

## 🔬 核心技术细节

### 1. RoPE位置编码
- **实现**: `transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding`
- **Position IDs**: 2D `[batch, seq_len]`
- **Cos/Sin Shape**: `[batch, seq_len, head_dim]`
- **RoPE Theta**: 1,000,000 (默认)

### 2. 梯度检查点
```python
hidden = torch.utils.checkpoint.checkpoint(
    layer_forward,
    hidden, next_layer, cos, sin,
    use_reentrant=False  # 非重入模式
)
```
- **作用**: 不保存中间激活值，显著降低显存占用
- **代价**: 需要重新计算前向传播 (时间换空间)

### 3. 概率过滤机制
```python
threshold = rl_config.prob_threshold  # 0.01
mask = vocab_probs > threshold
vocab_probs_filtered = vocab_probs * mask.float()
```
- **目的**: 去除低概率噪声，让RNN关注高概率token
- **效果**: 稀疏化输入，提高策略网络的鲁棒性

### 4. 价值裁剪
```python
target_value = final_reward - step_penalty * remaining_steps
target_value = max(0.0, min(1.0, target_value))  # 裁剪到[0,1]
```
- **必要性**: 防止步数惩罚导致负价值，与Critic的Sigmoid输出保持一致
- **范围**: [0, 1]

### 5. Advantage计算
```python
advantage = target_value - predicted_value.detach()
```
- **Detach**: 防止梯度回传到Critic
- **意义**: 衡量当前动作比预期好多少

---

## 📈 训练监控

### 日志输出示例
```
[Sample 0] input_len=45, output_len=12
    Token 0: reward=0.8234, path_len=3 → 最优路径: path=[5, 12, 19]
    Token 1: reward=0.7891, path_len=2 → 最优路径: path=[8, 19]
  [Sample 0] tokens=12, avg_loss=2.3456, traj_buffer=12

已处理 5 个样本, 平均CE=2.1234

  [Checkpoint] 保存前更新剩余 60 个三元组

开始RL网络更新...
轨迹缓冲区大小: 60 条完整轨迹
    Trajectory 10/60: actor=0.0234, critic=0.0156
    Trajectory 20/60: actor=0.0198, critic=0.0142
    ...
Actor-Critic 训练完成：actor loss=0.0215, critic loss=0.0148, 总 loss=0.0363
成功训练 60/60 条轨迹
RL网络更新完成
```

### 关键指标
- **CE Loss**: 交叉熵损失，衡量预测准确性
- **Actor Loss**: 策略网络损失，越低越好
- **Critic Loss**: 价值网络损失，越低越好
- **Reward**: target_token的预测概率，越高越好
- **Path Length**: 平均路径长度，反映跳转效率

---

## 🎯 创新点总结

1. **动态路径选择**: 打破传统顺序执行，允许跳跃式推理
2. **RNN时序建模**: LSTM捕捉路径选择的时序依赖
3. **残差连接架构**: 部分信息跨过LSTM，避免信息丢失
4. **概率过滤**: 稀疏化输入，提高鲁棒性
5. **梯度检查点**: 显存优化，支持更长路径
6. **联合训练**: LoRA和RL交替更新，相互促进
7. **Rollout机制**: 实时评估不同路径的质量
8. **价值裁剪**: 确保数值稳定性

---

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.0+
- **模型库**: HuggingFace Transformers
- **基础模型**: Qwen3-1.7B
- **精度**: bfloat16 (混合精度训练)
- **优化器**: AdamW
- **硬件要求**: NVIDIA GPU (推荐12GB+显存)

---

## 📝 注意事项

1. **显存管理**: 
   - 使用梯度检查点降低显存占用
   - 每处理完一个token清理缓存
   - 建议使用12GB+显存的GPU

2. **训练速度**:
   - 逐样本训练 (batch_size=1)
   - 路径探索需要多次前向传播
   - 预计训练速度较慢，适合小规模精调

3. **超参数调优**:
   - `prob_threshold`: 影响RNN输入的稀疏程度
   - `step_penalty`: 控制路径长度的偏好
   - `rl_update_interval`: 平衡LoRA和RL的训练频率

4. **断点续训**:
   - 自动保存checkpoint
   - 重启时自动恢复训练进度
   - 优先加载最优模型 (rl_best.pt)

---