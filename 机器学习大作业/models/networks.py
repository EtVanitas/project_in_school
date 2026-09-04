"""
模型组件模块 - LoRA 适配器、RNN Actor-Critic网络
全部使用 bfloat16 精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math
from config import LoRAConfig, RLConfig, DEFAULT_DTYPE, DEFAULT_DEVICE


class LoRALinear(nn.Module):
    """带 LoRA 的线性层 - 原始权重冻结，LoRA 矩阵可训练"""
    
    def __init__(self, in_features: int, out_features: int, frozen_weight: torch.Tensor):
        super().__init__()
        
        lora_config = LoRAConfig()
        device = frozen_weight.device
        base_dtype = frozen_weight.dtype
        
        # 原始权重（冻结）
        self.register_buffer("weight", frozen_weight, persistent=False)
        self.register_buffer("bias", torch.zeros(out_features, device=device, dtype=base_dtype), persistent=False)
        
        # LoRA 矩阵（可训练）
        self.lora_A = nn.Linear(in_features, lora_config.r, bias=False, device=device, dtype=DEFAULT_DTYPE)
        self.lora_B = nn.Linear(lora_config.r, out_features, bias=False, device=device, dtype=DEFAULT_DTYPE)
        
        # 缩放因子
        self.scaling = lora_config.alpha / lora_config.r
        
        # Dropout
        self.lora_dropout = nn.Dropout(lora_config.dropout)
        
        # 初始化 LoRA 权重
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
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
        self.layer = transformer_layer
        self._replace_linear_layers(transformer_layer)
    
    def _replace_linear_layers(self, module: nn.Module):
        """递归替换所有目标线性层为 LoRALinear"""
        lora_config = LoRAConfig()
        
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(target in name for target in lora_config.target_modules):
                # 创建 LoRA 层
                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    frozen_weight=child.weight.data.clone()
                )
                
                # 复制 bias
                if child.bias is not None:
                    lora_layer.bias.data.copy_(child.bias.data)
                
                setattr(module, name, lora_layer)
            else:
                self._replace_linear_layers(child)
    
    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class RNNActorCritic(nn.Module):
    """基于单向LSTM的Actor-Critic网络（残差连接架构）"""
    
    def __init__(self):
        super().__init__()
        rl_config = RLConfig()
        device = DEFAULT_DEVICE
        dtype = DEFAULT_DTYPE
        
        # 1. Prob向量投影: vocab_size -> 256
        self.prob_projection = nn.Linear(
            rl_config.vocab_size, rl_config.prob_proj_dim, device=device, dtype=dtype
        )
        
        # 2. Prob向量投影 + layer拼接后降维: 256 + 20 -> 128
        input_to_lstm_dim = rl_config.prob_proj_dim + rl_config.layer_encoding_dim  # 256 + 20 = 276
        self.lstm_input_proj = nn.Sequential(
            nn.Linear(input_to_lstm_dim, rl_config.lstm_hidden_dim, device=device, dtype=dtype),
            nn.SiLU()
        )
        
        # 3. LSTM: 128 -> 128 (单层)
        self.lstm = nn.LSTM(
            input_size=rl_config.lstm_hidden_dim,
            hidden_size=rl_config.lstm_hidden_dim,
            num_layers=rl_config.num_layers,  # 1
            batch_first=True,
            bidirectional=rl_config.bidirectional,  # False
            device=device,
            dtype=dtype
        )
        
        # 4. 状态融合: 128(LSTM输出) + 128(后半残差) + 20(layer) = 276
        fusion_dim = rl_config.fusion_dim  # 276
        
        # 5. Actor头: 276 -> 64 -> 16
        self.actor_head = nn.Sequential(
            nn.Linear(fusion_dim, rl_config.actor_hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(rl_config.actor_hidden_dim, rl_config.action_dim, device=device, dtype=dtype)
        )
        
        # 6. Critic头: 276 -> 64 -> 16 -> 1 -> Sigmoid
        self.critic_head = nn.Sequential(
            nn.Linear(fusion_dim, rl_config.critic_hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(rl_config.critic_hidden_dim, rl_config.critic_hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(rl_config.critic_hidden_dim, 1, device=device, dtype=dtype),
            nn.Sigmoid()  # 将价值限制在[0, 1]范围
        )
        
        # LSTM隐藏状态
        self.hidden_state = None
    
    def forward(self, prob_vector: torch.Tensor, 
                layer_index: torch.Tensor,
                is_reset: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 重置LSTM hidden state (新token开始时)
        if is_reset:
            self.reset_hidden_state()
        
        # 1. 构建layer one-hot编码 (20维)
        layer_onehot = self._build_layer_onehot(layer_index, prob_vector.device)  # [20]
        
        # 2. Prob向量投影: [151936] -> [256]
        prob_features = self.prob_projection(prob_vector)  # [256]
        
        # 3. 分割为前后两半用于残差连接: [128] + [128]
        half_dim = prob_features.shape[-1] // 2  # 128
        prob_front = prob_features[:half_dim]  # [128]
        prob_back = prob_features[half_dim:]   # [128] (残差连接，后面使用)
        
        # 4. 完整prob特征与layer拼接: [256] + [20] = [276]
        combined_with_layer = torch.cat([prob_features, layer_onehot], dim=-1)  # [276]
        
        # 5. 降维到LSTM输入: [276] -> [128]
        lstm_input = self.lstm_input_proj(combined_with_layer)  # [128]
        
        # 6. LSTM处理: [128] -> [128]
        lstm_seq = lstm_input.unsqueeze(0).unsqueeze(0)  # [1, 1, 128] (batch=1, seq_len=1)
        lstm_out, self.hidden_state = self.lstm(lstm_seq, self.hidden_state)  # [1, 1, 128]
        lstm_context = lstm_out.squeeze(0).squeeze(0)  # [128]
        
        # 7. 融合: [128(LSTM)] + [128(后半残差)] + [20(layer)] = [276]
        fused_features = torch.cat([lstm_context, prob_back, layer_onehot], dim=-1)  # [276]
        
        # 8. Actor和Critic输出
        action_logits = self.actor_head(fused_features)  # [20]
        state_value = self.critic_head(fused_features)  # [1]
        
        return action_logits, state_value
    
    def _build_layer_onehot(self, layer_index, device):
        """构建layer one-hot编码 (20维: 覆盖层8-27 + 输出动作)，输入为标量"""
        # layer_index已经是标量 (Python int)
        local_idx = int(layer_index) - 8
        local_idx = max(0, min(19, local_idx))  # clamp to [0, 19]
        
        # 创建one-hot向量
        onehot = torch.zeros(20, device=device, dtype=DEFAULT_DTYPE)
        onehot[local_idx] = 1.0
        
        return onehot
    
    def reset_hidden_state(self):
        """重置LSTM hidden state为零"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # 从LSTM层获取num_layers和hidden_size
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size

        h0 = torch.zeros(num_layers, 1, hidden_dim, device=device, dtype=dtype)
        c0 = torch.zeros(num_layers, 1, hidden_dim, device=device, dtype=dtype)
        self.hidden_state = (h0, c0)
