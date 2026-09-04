"""
配置
简洁版配置，只包含训练所需的参数
"""

from dataclasses import dataclass, field
import os
import torch


PROJECT_ROOT = r"c:\Users\35201\.vscode\ai"
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.bfloat16
ACTION_OUTPUT = 19  # 输出动作（最后一个动作，索引19）


@dataclass
class PathConfig:
    """路径配置 - 统一管理所有路径信息"""
    
    project_root: str = PROJECT_ROOT  # 项目根目录
    qwen_model_path: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "Qwen3-1.7B"))  # Qwen基础模型路径
    models_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "trainer", "models"))  # 模型保存目录 (LoRA + RL网络)
    jsonl_dataset_path: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "stage2_small_dataset.jsonl"))  # JSONL数据集路径
    checkpoint_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "trainer", "checkpoints"))  # Checkpoint保存目录


@dataclass
class ModelConfig:
    """Qwen3-1.7B 模型配置"""
    
    num_layers: int = 28  # 总层数
    fixed_layers: int = 9  # 固定层数 (前9层不参与RL决策)
    dynamic_layers: int = 19  # 动态层数 (后19层由RL控制)
    
    hidden_size: int = 2048  # 隐藏层维度
    vocab_size: int = 151936  # 词表大小
    
    max_steps: int = 30  # 最大推理步数
    dtype: str = "bfloat16"  # 数据类型
    batch_size: int = 1  # 批大小 (暂时为逐样本训练)


@dataclass
class LoRAConfig:
    """LoRA 配置"""
    
    r: int = 8  # LoRA秩
    alpha: float = 16.0  # LoRA缩放系数
    dropout: float = 0.1  # Dropout概率
    target_modules: list = field(default_factory=lambda: [  # 目标模块列表
        "q_proj", "k_proj", "v_proj", "o_proj",  # 标准注意力层
        "gate_proj", "up_proj", "down_proj"  # MLP 层
    ])
    
    lr: float = 1e-4  # 学习率
    betas: tuple = (0.9, 0.95)  # AdamW优化器beta参数
    weight_decay: float = 0.01  # 权重衰减
    grad_clip: float = 1.0  # 梯度裁剪阈值


@dataclass
class RLConfig:
    """强化学习配置 (RNN Actor-Critic)"""
    
    # 状态表示
    vocab_size: int = 151936  # 词表大小（直接使用所有token概率）
    layer_encoding_dim: int = 20  # 层索引one-hot编码维度 (19个动态层 + 1个输出动作)
    
    # 网络架构配置
    prob_proj_dim: int = 256  # 概率向量投影维度
    lstm_hidden_dim: int = 128  # LSTM隐藏层维度
    num_layers: int = 1  # LSTM层数（单层）
    bidirectional: bool = False  # 单向LSTM
    
    # 动作空间
    action_dim: int = 20  # 19个动态层 + 1个输出动作
    
    # Actor-Critic头
    fusion_dim: int = 276  # 融合维度: 128(LSTM) + 128(残差) + 20(layer)
    actor_hidden_dim: int = 64  # Actor: 276->64->20
    critic_hidden_dim: int = 64  # Critic: 276->64->20->1
    
    # 优化器
    lr: float = 1e-4  # 学习率
    betas: tuple = (0.9, 0.95)  # AdamW优化器beta参数
    weight_decay: float = 0.01  # 权重衰减
    grad_clip: float = 1.0  # 梯度裁剪阈值
    
    # 奖励计算
    step_penalty: float = 0.01  # 每步惩罚系数
    
    # 状态过滤
    prob_threshold: float = 0.01  # 词表概率过滤阈值 (只保留概率>阈值的token)
    
    # 训练控制
    rl_update_interval: int = 100  # RL更新间隔，每N个token更新一次策略网络


# 默认配置实例
path_config = PathConfig()
model_config = ModelConfig()
rl_config = RLConfig()
