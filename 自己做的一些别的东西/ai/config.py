# config.py
from dataclasses import dataclass


@dataclass
class ModelArgs:
    vocab_size: int = 21128                    # BERT 中文词表大小（与 tokenizer_path 保持一致）
    d_model: int = 512                         # 模型维度
    num_layers: int = 24                       # LayerBlock 层数
    first_gauss: int = 256                     # 首层高斯基个数
    first_wave: int = 128                      # 首层小波基个数
    layer_ff_dim: int = 1024                   # 每层权重维度
    layer_gauss: int = 128                     # 每层高斯基个数
    layer_wave: int = 64                       # 每层小波基个数
    max_input_len: int = 256                   # token 窗口长度
    max_mem_len: int = 24                      # 历史权重记忆长度


@dataclass
class TrainArgs:
    # 数据
    data_dir: str = "./data"
    tokenizer_path: str = "./bert-base-chinese"  # 本地 BERT 分词器目录
    # 训练
    lr: float = 1e-4                           # AdamW 学习率
    weight_decay: float = 0.01                 # AdamW 权重衰减
    muon_lr: float = 0.002                     # Muon 学习率
    muon_weight_decay: float = 0.01            # Muon 权重衰减
    muon_momentum: float = 0.95                # Muon 动量
    muon_nesterov: bool = True                 # Muon Nesterov 动量
    muon_ns_steps: int = 5                     # Muon 牛顿-舒尔茨迭代步数
    use_muon: bool = True                      # True=Muon+AdamW 双优化器；False=全部 AdamW
    max_steps: int = 0                         # 0 = 无限制（训完全部数据），>0 则达到后提前停止
    save_steps: int = 5000                     # 每 N 步保存一次 checkpoint（含数据位置，用于断点续训）
    log_steps: int = 500                       # 每 N 步记一个 loss 平均值到 train_loss.csv（供画曲线）
    # 系统
    output_dir: str = "./checkpoints"
    device: str = "cuda"
    # 混合精度
    use_bf16: bool = True
