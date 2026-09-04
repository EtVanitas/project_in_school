"""
配置类模块
集中管理所有超参数配置

重构说明：
- PathConfig: 统一管理所有路径
- ModelConfig: 模型架构和推理配置
- LoRAConfig: LoRA 微调超参数
- RLConfig: 强化学习超参数
"""

from dataclasses import dataclass, field
from typing import List
import os
import torch


PROJECT_ROOT = r"c:\Users\35201\.vscode\ai"  # 项目根目录
ACTION_OUTPUT = 15  # 输出动作（最后一个动作）
DEFAULT_DTYPE = torch.bfloat16


@dataclass
class PathConfig:
    """统一路径配置 - 管理所有数据和模型存储路径"""
    
    # 根目录
    project_root: str = PROJECT_ROOT
    
    # 数据目录（trajectory_dir）
    trajectory_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "trajectory_dir"))
    text_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "trajectory_dir", "texts"))
    base_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "trajectory_dir", "bases"))
    trajectory_files_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "trajectory_dir", "trajectories"))
    path_records_file: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "trajectory_dir", "path_records.json"))
    stage1a_path_pool_file: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "trajectory_dir", "stage1a_path_pool.json"))
    
    # 模型目录（简化为 stage1 和 stage2）
    models_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "models", "training_models"))
    stage1_model_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "models", "training_models", "stage1"))
    stage2_model_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "models", "training_models", "stage2"))


@dataclass
class ModelConfig:
    """Qwen3.5-2B 模型架构配置"""
    
    # 路径配置
    model_path: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "Qwen3.5-2B"))
    path_config: PathConfig = field(default_factory=PathConfig)
    
    # 模型架构
    num_layers: int = 24  # 总层数
    fixed_layers: int = 9  # 前 N 层固定顺序
    dynamic_layers: int = 15  # 后 N 层动态选择
    
    # 模型维度
    hidden_size: int = 2048  # 隐藏层维度
    vocab_size: int = 248320  # 词表大小
    intermediate_size: int = 6144  # MLP 中间层维度
    num_attention_heads: int = 8  # 注意力头数
    num_key_value_heads: int = 2  # GQA KV 头数
    
    # 推理配置
    max_steps: int = 30  # 最大推理步数
    dtype: str = "bfloat16"  # 模型精度
    batch_size: int = 4  # 批次大小


@dataclass
class LoRAConfig:
    """LoRA 微调配置"""
    
    # LoRA 超参数
    r: int = 16  # LoRA 秩
    alpha: float = 32.0  # 缩放因子
    dropout: float = 0.1  # Dropout 概率
    
    # 目标模块
    target_modules: List[str] = field(default_factory=lambda: [
        # 标准注意力层 (full_attention)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # 线性注意力层 (linear_attn)
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        # MLP 层 (所有层共有)
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 优化器配置
    lr: float = 1e-4  # 学习率
    betas: tuple = (0.9, 0.95)  # AdamW betas
    weight_decay: float = 0.01  # 权重衰减
    grad_clip: float = 1.0  # 梯度裁剪


@dataclass
class RLConfig:
    """强化学习配置 (Actor-Critic 网络)"""
    
    # 状态表示
    token_prob_dim: int = 399  # token概率向量维度
    layer_encoding_dim: int = 15  # 层索引 one-hot 编码
    
    @property
    def state_raw_dim(self) -> int:
        """原始状态维度 = token_probs(399) + layer(15) + step(1) + context(1) = 416"""
        return self.token_prob_dim + self.layer_encoding_dim + self.step_idx_dim + self.context_dim
    
    # LSTM配置
    lstm_input_dim: int = 128  # 投影后维度
    lstm_hidden_dim: int = 128  # 单向LSTM隐藏层维度
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = False  # 单向,确保训练-推理一致性
        
    # 动作空间
    action_dim: int = 16  # 15 个动态层 + 1 个输出动作
    
    # Actor-Critic头配置
    actor_hidden_dims: list = None  # type: ignore  # 128->64->32->16
    critic_hidden_dims: list = None  # type: ignore  # 128->64->16->1
    
    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [64, 32]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [64, 16]
    
    # 优化器配置
    lr: float = 1e-4  # 学习率
    betas: tuple = (0.9, 0.95)  # AdamW betas
    weight_decay: float = 0.01  # 权重衰减
    grad_clip: float = 1.0  # 梯度裁剪
    
    # 奖励计算
    step_penalty: float = 0.01  # 每步惩罚系数
    
    # Stage1A 探索配置
    num_exploration_paths: int = 4  # 每个样本探索的路径数
    
    # Stage2 分支配置
    branch_prob_threshold: float = 0.2  # 第二大概率阈值
    branch_prob_ratio: float = 0.8  # 第二大概率/最大概率比值
