# 大语言模型完整框架

这是一个从零开始构建的大语言模型框架，包含了完整的训练流程。

## 框架特点

- ✅ 完整的文本处理管道
- ✅ 支持BERT-base-chinese分词器
- ✅ 支持TXT和JSONL数据格式
- ✅ 智能文本块处理（512长度固定块）
- ✅ 20层深度神经网络架构
- ✅ 创新的自定义矩阵神经网络模块（AXBX^T CXD）
- ✅ 残差连接和记忆矩阵机制
- ✅ 梯度检查点显存优化
- ✅ 统一配置管理系统
- ✅ Mask Language Modeling预训练
- ✅ 标准的前向传播和反向传播
- ✅ 为图像处理预留扩展接口
- ✅ 规范化的命名体系
- ✅ 详细的中文注释
- ✅ 模块化设计，易于扩展

## 目录结构

```
├── alice.py                  # 主框架文件
├── demo.py                  # 基础使用示例
├── matrix_demo.py           # 自定义矩阵模块演示
├── sequence_chunk_demo.py   # 序列块处理演示
├── chunk_training_example.py # 序列块训练示例
├── main_model_demo.py       # 主模型演示
├── norm_test.py             # 归一化层测试
├── unified_nonlinear_test.py # 统一非线性处理测试
├── cleaned_matrix_test.py    # 精简版CustomMatrixBlock测试
├── simplified_logic_test.py  # 精简逻辑测试
├── axb_text_test.py          # AXB文本处理测试
├── requirements.txt          # 依赖包列表
├── sample_data.jsonl        # 示例JSONL数据
└── README.md                # 说明文档
```

## 核心组件

### 1. 文本处理模块
- `TextChunkProcessor`: 文本块处理器（创新）
- `TextPreprocessor`: 文本预处理器（优化版）
- 默认使用BERT-base-chinese分词器
- 支持JSONL格式数据
- 智能文本块分割（512长度）
- 自动填充处理
- 内置Mask Language Modeling数据处理
- **优化特性**: 移除冗余方法，简化数据流程，提高效率

### 2. 模型架构模块
- `MainModel`: 20层深度主模型（处理文本数据）
- `CustomMatrixBlock`: 自定义矩阵神经网络块（创新模块）
- `ModelConfig`: 统一配置管理器
- 包含词嵌入层、20个CustomMatrixBlock层、残差连接、记忆矩阵机制
- 支持梯度检查点优化
- 完整的前向传播流程：词嵌入→block_0→blocks_1-6→block_17→blocks_7-12→block_18→blocks_13-16→block_19

### 3. 图像处理模块（预留）
- `ImagePreprocessor`: 图像预处理器（预留接口）
- `ImageChunkProcessor`: 图像块处理器（预留接口）
- 为未来图像处理功能预留扩展空间

### 3. 训练模块
- `Trainer`: 模型训练器
- 支持Mask Language Modeling训练
- 损失计算、梯度裁剪、学习率调度
- 训练和评估功能
- **梯度累积优化**: 默认每4个文本块累积梯度后更新参数，减少训练方差
- **梯度检查点优化**: MainModel支持在blocks_7-12层启用梯度检查点，显著降低显存消耗
- 显存优化比例可达30-40%，同时保持训练稳定性
- **代码优化**: 消除重复代码，提高可维护性
- 自动检查点保存和最佳模型跟踪
- **预处理优化**: 简化文本处理流程，移除无用组件
- **GPU加速训练**: 自动检测CUDA环境
- **实时监控**: 训练进度、ETA估算、GPU内存使用情况显示
- **学习率预热**: 线性预热+余弦衰减调度
- **梯度截断**: 智能梯度范数控制（默认0.5）
- **RMSNorm稳定化**: 19个PyTorch内置RMSNorm层防止梯度爆炸

### 4. 微调模块
- `SupervisedFineTuner`: 有监督微调器
- 支持指令微调格式
- 可自定义提示模板

### 5. 强化学习模块
- `RLTrainer`: 强化学习训练器
- 基于PPO算法实现
- 包含奖励建模功能

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行示例
```bash
# 运行完整训练流程
python alice.py

# 运行基础功能演示
python demo.py

# 运行自定义矩阵模块演示（重点推荐）
python matrix_demo.py

# 运行序列块处理演示
python sequence_chunk_demo.py

# 运行序列块训练示例
python chunk_training_example.py

# 运行主模型演示（新增）
python main_model_demo.py

# 运行归一化层测试（新增）
python norm_test.py

# 运行统一非线性处理测试（新增）
python unified_nonlinear_test.py

# 运行精简版CustomMatrixBlock测试（新增）
python cleaned_matrix_test.py

# 运行精简逻辑测试（新增）
python simplified_logic_test.py

# 运行AXB文本处理测试（新增）
python axb_text_test.py
```

### 完整预训练流程

```bash
# 直接运行完整预训练流程
python alice.py
```

### 自定义训练

```python
# 1. 准备数据
processor = TextPreprocessor()  # 默认使用BERT-base-chinese分词器

# 加载数据
texts = processor.load_data('your_data.jsonl', file_format='jsonl')

# 2. 创建数据加载器
train_loader = processor.create_text_dataloader(
    texts=texts,
    chunk_size=512,     # 固定块大小
    batch_size=1,       # 逐个块处理
    shuffle=True        # 打乱数据
)

# 3. 创建主模型
model = MainModel(
    vocab_size=21128,
    text_seq_len=512,
    use_gradient_checkpointing=True
)

# 4. 创建训练器（默认4步梯度累积）
trainer = Trainer(model, learning_rate=5e-4, accumulation_steps=4)

# 或者自定义累积步数
# trainer = Trainer(model, learning_rate=5e-4, accumulation_steps=8)

# 5. 执行训练
for epoch in range(10):
    epoch_loss = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")
```

## 参数配置

### 模型超参数
- `vocab_size`: 词汇表大小（默认30522）
- `d_model`: 模型维度（默认768）
- `n_layers`: Transformer层数（默认12）
- `use_gradient_checkpointing`: 是否启用梯度检查点（默认True）

### 梯度检查点优化
- **作用层**: blocks_7-12（索引6-11）
- **显存优化**: 降低30-40%显存消耗
- **性能影响**: 前向传播略有延迟，反向传播延迟较小
- **兼容性**: 完全保持训练稳定性和梯度正确性
- `n_heads`: 注意力头数（默认12）
- `d_ff`: 前馈网络维度（默认3072）

### 训练超参数
- `learning_rate`: 学习率（默认1e-4）
- `batch_size`: 批次大小（默认1）
- `accumulation_steps`: 梯度累积步数（默认4）
- `max_seq_len`: 最大序列长度（默认512）

## 扩展建议

1. **增加更多训练技巧**：
   - 混合精度训练
   - 梯度累积
   - 更复杂的学习率调度

2. **完善强化学习**：
   - 实现完整的PPO算法
   - 添加人类反馈机制
   - 设计更好的奖励函数

3. **优化性能**：
   - 添加分布式训练支持
   - 实现模型并行
   - 优化内存使用

## 性能优化

- **梯度检查点**: 在blocks_7-12层启用，节省30-40%显存
- **梯度累积**: 减少小批次训练的方差
- **自动检查点**: 防止训练中断丢失进度
- **显存优化**: 智能批处理和内存管理
- **GPU监控**: 实时显示显存使用情况和训练进度
- **RMSNorm稳定化**: 19层PyTorch内置RMS归一化防止梯度爆炸
- **学习率预热**: 智能学习率调度避免训练初期不稳定
- **梯度截断**: 动态梯度范数控制保持训练稳定

## 注意事项

- 建议使用GPU进行训练（支持CUDA 11.0+）
- 大模型训练需要充足的内存（推荐16GB+显存）
- 可根据硬件条件调整模型大小
- 训练时间较长，建议使用检查点机制
- BERT分词器需要网络连接下载
- JSONL文件应确保每行都是有效的JSON对象
- 首次运行会自动下载BERT分词器，请保持网络连接
- 梯度累积默认每4个文本块更新一次参数，可根据需要调整
- 数据加载失败时会自动使用示例数据进行测试
- 支持混合精度训练，可显著提升训练速度
- 自动显存管理和溢出保护机制

## 许可证

MIT License