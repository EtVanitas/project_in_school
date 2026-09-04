# config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    vocab_size: int = 151670                   # Qwen3-1.7B 完整词表
    dim: int = 1024                            # 模型维度
    n_layers: int = 24                         # 总层数
    n_heads: int = 16                          # 头数
    ffn_hidden_dim: int = 1024                 # 特征维度为 1024
    use_swiglu: bool = True                    # SwiGLU 激活
    norm_eps: float = 1e-5
    max_seq_len: int = 1024                    # 训练最大长度
    rope_theta: float = 10000.0
    num_cycles: int = 2                        # 默认循环次数
    use_gradient_checkpointing: bool = False   # 默认开启梯度检查点
    mask_token_id: int = 151669                 # <mask> 在 Qwen3 中的 ID
    pad_token_id: int = 151643                 # <|endoftext|> 作为 pad


@dataclass
class TrainArgs:
    # 数据
    data_dir: str = "./data"
    tokenizer_path: str = "./Qwen3-1.7B"
    # 训练
    batch_size: int = 1
    epochs: int = 3
    lr: float = 1e-4                           # AdamW 学习率
    weight_decay: float = 0.01                 # AdamW 权重衰减
    muon_lr: float = 0.002                     # Muon 学习率
    muon_weight_decay: float = 0.01            # Muon 权重衰减
    warmup_steps: int = 1000                   # 学习率预热步数
    max_steps: int = 0                         # 0 = 无限制（训完所有 epoch），>0 则达到后提前停止
    save_steps: int = 5000                     # 每 N 步保存一次 checkpoint（用于断点续训）
    log_steps: int = 100                       # 每 N 步打印一次日志
    # 系统
    output_dir: str = "./checkpoints"
    device: str = "cuda"
    seed: int = 42
    # 混合精度
    use_bf16: bool = True
