"""
模型组件模块 - LoRA 适配器、网络结构等
全部使用 bfloat16 精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import math
from config import LoRAConfig, ModelConfig, RLConfig, DEFAULT_DTYPE

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== LoRA 适配器 ====================

class LoRALinear(nn.Module):
    """带 LoRA 的线性层 - 原始权重冻结，LoRA 矩阵可训练"""
    def __init__(self, in_features: int, out_features: int, 
                 lora_r: int = 16, lora_alpha: float = 32.0,  # 提高秩和 alpha
                 lora_dropout: float = 0.1, frozen_weight: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        # 使用全局默认设备和 dtype
        device = device or DEFAULT_DEVICE
        self.lora_dtype = DEFAULT_DTYPE  # LoRA 也使用 bf16
        base_dtype = dtype or DEFAULT_DTYPE
        
        # 原始权重（冻结）- 使用传入的 dtype
        if frozen_weight is not None:
            # 确保权重在正确的设备上
            device = frozen_weight.device
            base_dtype = frozen_weight.dtype
            self.register_buffer("weight", frozen_weight, persistent=False)
            self.register_buffer("bias", torch.zeros(out_features, device=device, dtype=base_dtype), persistent=False)
        else:
            self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=base_dtype))
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=base_dtype))
        
        # LoRA 矩阵（可训练）
        self.lora_A = nn.Linear(in_features, lora_r, bias=False, device=device, dtype=self.lora_dtype)
        self.lora_B = nn.Linear(lora_r, out_features, bias=False, device=device, dtype=self.lora_dtype)
        
        # 缩放因子
        self.scaling = lora_alpha / lora_r
        
        # Dropout
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # 初始化 LoRA 权重
        self._init_lora_weights(self.lora_dtype)
    
    def _init_lora_weights(self, dtype: torch.dtype):
        """初始化 LoRA 权重"""
        # kaiming_uniform_ 在正确的 dtype 上执行
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        
        # 初始化为零
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始前向传播（冻结）
        base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA 分支（可训练）
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        
        return base_output + lora_output


class LoRAModuleWrapper(nn.Module):
    """为 Transformer 层的所有线性层添加 LoRA"""
    def __init__(self, transformer_layer: nn.Module):
        super().__init__()
        # 直接使用全局配置
        self.lora_config = LoRAConfig()
        self.original_layer = transformer_layer
        
        # 递归查找所有目标线性层并替换
        self._replace_linear_layers(transformer_layer)
    
    def _replace_linear_layers(self, module: nn.Module):
        """递归替换所有目标线性层为 LoRALinear"""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(
                target in name for target in self.lora_config.target_modules
            ):
                # 获取原始设备的 dtype 和设备信息
                original_device = child.weight.device
                original_dtype = child.weight.dtype
                
                # 创建带 LoRA 的线性层
                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    lora_r=self.lora_config.r,
                    lora_alpha=self.lora_config.alpha,
                    lora_dropout=self.lora_config.dropout,
                    frozen_weight=child.weight.data.clone().to(device=original_device, dtype=original_dtype)
                )
                
                if child.bias is not None:
                    lora_layer.bias.data.copy_(child.bias.data)
                
                # 替换
                setattr(module, name, lora_layer)
            else:
                # 递归处理子模块
                self._replace_linear_layers(child)
    
    def forward(self, *args, **kwargs):
        return self.original_layer(*args, **kwargs)
    
    def get_trainable_params(self):
        """获取所有可训练参数（LoRA 部分）"""
        params = []
        for name, param in self.named_parameters():
            if "lora_" in name:
                params.append(param)
        return params


# ==================== RL 网络 ====================

class MLPActorCritic(nn.Module):
    """基于 MLP 的 Actor-Critic 网络 (简化版)"""
    def __init__(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        # 直接使用全局配置
        self.rl_config = RLConfig()
        self.model_config = ModelConfig()
        
        # 使用全局默认设备，bfloat16 精度
        device = device or DEFAULT_DEVICE
        dtype = DEFAULT_DTYPE  # RL 网络使用 bfloat16
        
        # 1. 词表概率投影：vocab_size -> 256 -> 64 (两层压缩)
        self.vocab_projection = nn.Sequential(
            nn.Linear(self.model_config.vocab_size, 256, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(256, 64, device=device, dtype=dtype),
            nn.SiLU()
        )
        
        # 2. 策略头（Actor）：81 → 32 → 16
        #    输入维度 = vocab_proj(64) + layer_onehot(15) + step_idx(1) + context(1) = 81
        enhanced_input_dim = 64 + self.rl_config.layer_encoding_dim + self.rl_config.step_idx_dim + self.rl_config.context_dim
        self.actor_head = nn.Sequential(
            nn.Linear(enhanced_input_dim, self.rl_config.actor_hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.rl_config.actor_hidden_dim, self.rl_config.action_dim, device=device, dtype=dtype)
        )
        
        # 3. 价值头（Critic）：81 → 16 → 1
        self.critic_head = nn.Sequential(
            nn.Linear(enhanced_input_dim, self.rl_config.critic_hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(self.rl_config.critic_hidden_dim, 1, device=device, dtype=dtype)
        )
    
    def _build_state_features(self, vocab_probs: torch.Tensor, layer_index: torch.Tensor,
                              context_length: torch.Tensor, step_idx: torch.Tensor) -> torch.Tensor:
        """构建状态特征向量"""
        batch_size = vocab_probs.shape[0]
        device = vocab_probs.device
        
        # 1. 词表概率投影到 64 维 (经过两层压缩: 248320 -> 256 -> 64)
        vocab_proj = self.vocab_projection(vocab_probs)  # [batch, 64]
        
        # 2. 层数 one-hot 编码（只针对后 15 层）
        layer_onehot = torch.zeros(batch_size, 15, device=device, dtype=DEFAULT_DTYPE)
        
        # 确保 layer_index 是 [batch] 形状
        if layer_index.dim() == 0:
            layer_index = layer_index.expand(batch_size)
        
        # 批量处理层索引
        mask = layer_index >= self.model_config.fixed_layers
        if mask.any():            # [关键修复] scatter_ 需要 Long 类型索引，先计算再转换
            local_indices = (layer_index[mask] - self.model_config.fixed_layers).long()
            layer_onehot[mask].scatter_(1, local_indices.unsqueeze(1), 1)
        
        # 3. 其他特征拼接 - 全部已经是 Tensor
        if context_length.dim() == 0:
            context_length = context_length.expand(batch_size)
        context_tensor = context_length.view(-1, 1).to(DEFAULT_DTYPE)
        
        if step_idx.dim() == 0:
            step_idx = step_idx.expand(batch_size)
        step_idx_tensor = step_idx.view(-1, 1).float().to(DEFAULT_DTYPE)
        
        # 4. 拼接所有特征：[batch, 81]
        combined = torch.cat([
            vocab_proj,              # [batch, 64]
            layer_onehot,            # [batch, 15]
            step_idx_tensor,         # [batch, 1]
            context_tensor,          # [batch, 1]
        ], dim=-1)  # Total: 64 + 15 + 1 + 1 = 81
        
        return combined
    
    def forward(self, vocab_probs: torch.Tensor, 
                layer_index: Union[int, torch.Tensor], 
                context_length: Union[int, torch.Tensor],
                step_idx: Union[int, torch.Tensor] = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 - 支持批量和单样本"""
        # 确保输入是 bfloat16
        if vocab_probs.dtype != DEFAULT_DTYPE:
            vocab_probs = vocab_probs.to(DEFAULT_DTYPE)
            
        # 确保是二维 [batch, vocab_size]
        if vocab_probs.dim() == 1:
            vocab_probs = vocab_probs.unsqueeze(0)
            
        # 构建状态特征
        combined = self._build_state_features(
            vocab_probs=vocab_probs,
            layer_index=layer_index,
            context_length=context_length,
            step_idx=step_idx
        )
        
        # combined = [vocab_proj(64), layer_onehot(15), step_idx(1), context(1)] = 81维
        enhanced_features = combined
        
        # Actor-Critic 输出
        action_logits = self.actor_head(enhanced_features)  # [batch, 16]
        state_value = self.critic_head(enhanced_features)   # [batch, 1]
            
        return action_logits, state_value


class RNNActorCritic(nn.Module):
    """基于单向LSTM的Actor-Critic网络
    
    架构:
    - Input Projection: 416 -> 128
    - LSTM: 128 -> 128 (2层,单向)
    - Attention Pooling
    - Actor Head: 128 -> 64 -> 32 -> 16
    - Critic Head: 128 -> 64 -> 16 -> 1
    
    特点: 通过LSTM hidden state捕捉token概率变化的时序模式
          使用单向LSTM确保训练-推理一致性
    """
    
    def __init__(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        device = device or DEFAULT_DEVICE
        dtype = DEFAULT_DTYPE
        
        # 1. 输入投影: 416 -> 128
        self.input_projection = nn.Sequential(
            nn.Linear(416, 128, device=device, dtype=dtype),
            nn.LayerNorm(128, device=device, dtype=dtype),
            nn.SiLU()
        )
        
        # 2. 单向LSTM: 捕捉时序依赖
        #    input=128, hidden=128, 2层, 单向 → output_dim=128
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=False,  # 单向,确保训练-推理一致性
            device=device,
            dtype=dtype
        )
        
        # 3. Attention机制: 从LSTM输出中提取关键信息
        self.attention = nn.Sequential(
            nn.Linear(128, 64, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(64, 1, device=device, dtype=dtype)
        )
        
        # 4. Actor头: 128 -> 64 -> 32 -> 16
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(64, 32, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(32, 16, device=device, dtype=dtype)
        )
        
        # 5. Critic头: 128 -> 64 -> 16 -> 1
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(64, 16, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(16, 1, device=device, dtype=dtype)
        )
        
        # LSTM隐藏状态 (用于跨时间步传递)
        self.hidden_state = None
    
    def forward(self, prob_vector: torch.Tensor, 
                layer_index: torch.Tensor,
                context_length: torch.Tensor,
                step_idx: torch.Tensor,
                is_reset: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            prob_vector: [batch, 399] token概率向量
            layer_index: [batch] 当前层索引
            context_length: [batch] 上下文长度
            step_idx: [batch] 步数索引
            is_reset: 是否重置LSTM隐藏状态(新样本开始时为True)
        
        Returns:
            action_logits: [batch, 16]
            state_value: [batch, 1]
        """
        batch_size = prob_vector.shape[0]
        
        # 重置隐藏状态(新样本)
        if is_reset or self.hidden_state is None:
            h0 = torch.zeros(2, batch_size, 128,  # num_layers=2, hidden=128
                           device=prob_vector.device, 
                           dtype=prob_vector.dtype)
            c0 = torch.zeros(2, batch_size, 128,
                           device=prob_vector.device,
                           dtype=prob_vector.dtype)
            self.hidden_state = (h0, c0)
        
        # 1. 构建layer one-hot编码
        layer_onehot = self._build_layer_onehot(layer_index, batch_size, prob_vector.device)
        
        # 2. 拼接完整状态: 399 + 15 + 1 + 1 = 416
        full_state = torch.cat([
            prob_vector,              # [batch, 399]
            layer_onehot,             # [batch, 15]
            step_idx.unsqueeze(-1),   # [batch, 1]
            context_length.unsqueeze(-1)  # [batch, 1]
        ], dim=-1)  # [batch, 416]
        
        # 3. 输入投影: 416 -> 128
        projected = self.input_projection(full_state)  # [batch, 128]
        
        # 4. LSTM处理 (需要序列维度)
        projected_seq = projected.unsqueeze(1)  # [batch, seq_len=1, 128]
        lstm_out, self.hidden_state = self.lstm(projected_seq, self.hidden_state)
        # lstm_out: [batch, 1, 128] (单向: hidden=128)
        
        # 5. Attention pooling
        attention_weights = self.attention(lstm_out)  # [batch, 1, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = (lstm_out * attention_weights).sum(dim=1)  # [batch, 128]
        
        # 6. Actor和Critic输出
        action_logits = self.actor_head(context_vector)  # [batch, 16]
        state_value = self.critic_head(context_vector)   # [batch, 1]
        
        return action_logits, state_value
    
    def _build_layer_onehot(self, layer_index, batch_size, device):
        """构建layer one-hot编码"""
        onehot = torch.zeros(batch_size, 15, device=device, dtype=DEFAULT_DTYPE)
        mask = layer_index >= 9  # fixed_layers=9
        if mask.any():
            local_indices = (layer_index[mask] - 9).long()
            onehot[mask].scatter_(1, local_indices.unsqueeze(1), 1)
        return onehot
    
    def reset_hidden_state(self):
        """外部调用的重置方法"""
        self.hidden_state = None

