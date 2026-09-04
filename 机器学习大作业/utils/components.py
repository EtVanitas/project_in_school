"""模型更新工具 - RL训练和梯度前向传播
包含RoPE位置编码计算功能
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

from config import DEFAULT_DTYPE, ACTION_OUTPUT, RLConfig, ModelConfig


def train_actor_critic(actor_critic, rl_optimizer, 
                         trajectory: Dict, device: torch.device) -> Tuple[float, float, float]:
    """训练一条完整轨迹的 Actor-Critic (保持RNN时序)"""

    rl_config = RLConfig()
    grad_clip = rl_config.grad_clip
    step_penalty = rl_config.step_penalty
    
    states = trajectory['states']
    actions = trajectory['actions']
    final_reward = trajectory['final_reward']

    seq_len = len(states)
    if seq_len == 0:
        return 0.0, 0.0, 0.0
    
    # 清零梯度
    rl_optimizer.zero_grad()
    
    # 重置RNN隐藏状态 (每条轨迹从头开始)
    actor_critic.reset_hidden_state()
    
    # 按时间步展开轨迹
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    
    # Actor和Critic在所有步骤都更新
    for t in range(seq_len):
        state = states[t]
        action = actions[t]
        
        # 提取状态特征
        prob_vector = state['prob_vector']  # [151936]
        layer_idx = state['layer_idx']  # 标量
        action_tensor = torch.tensor(action, dtype=torch.long)  # 标量
        
        # 前向传播 (RNN会自动使用上一步的hidden_state)
        action_logits, predicted_value = actor_critic(
            prob_vector=prob_vector,
            layer_index=layer_idx,
            is_reset=False  # 不reset,保持连续性
        )
        
        # 使用步数惩罚倒推每个状态的目标价值
        remaining_steps = seq_len - 1 - t
        target_value = final_reward - step_penalty * remaining_steps
        
        # 裁剪目标价值到 [0, 1] 范围，避免数值异常
        target_value = max(0.0, min(1.0, target_value))
        target_value_tensor = torch.tensor(target_value, device=device, dtype=DEFAULT_DTYPE)  # 标量
        
        # Advantage: 使用倒推的目标价值
        advantage = target_value_tensor - predicted_value.detach()  # 标量
        
        # 计算Actor Loss (所有步骤)
        log_prob = F.log_softmax(action_logits, dim=-1)  # [20]
        selected_log_prob = log_prob[action_tensor]  # 标量
        actor_loss_t = -(selected_log_prob * advantage)  # 标量
        total_actor_loss += actor_loss_t
        
        # Critic Loss: 所有步骤都计算
        critic_loss_t = F.mse_loss(predicted_value, target_value_tensor)  # 标量
        total_critic_loss += critic_loss_t
    
    # 平均每个时间步的loss
    avg_actor_loss = total_actor_loss / seq_len
    avg_critic_loss = total_critic_loss / seq_len
    
    # 简化损失组合: 直接相加,因为价值已在[0,1]范围,量级接近
    total_loss = avg_actor_loss + avg_critic_loss
    
    # 反向传播
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), grad_clip)
    rl_optimizer.step()
    
    return avg_actor_loss.item(), avg_critic_loss.item(), total_loss.item()


def forward_with_grad(layered_loader, input_ids: torch.Tensor, path: List[int],
                        target_token: torch.Tensor, seq_len: int,
                        device: torch.device) -> torch.Tensor:
    """沿给定路径有梯度地前向传播(使用梯度检查点)"""

    model_config = ModelConfig()
    
    # 预计算 RoPE
    cos, sin = compute_rope(seq_len, device, model_config.hidden_size)
    
    # 前9层固定执行
    hidden = layered_loader.model.model.embed_tokens(input_ids.unsqueeze(0))
    for layer_idx in range(model_config.fixed_layers):
        layer = layered_loader.layers[layer_idx]
        hidden = layer(hidden, position_embeddings=(cos, sin))
    
    # 定义单层的 forward 函数（用于梯度检查点）
    def layer_forward(h, target_layer_idx, cos, sin):
        """单层的前向传播（需要保存输入以支持 checkpoint）"""
        return layered_loader.layers[target_layer_idx](
            h,
            position_embeddings=(cos, sin)
        )
    
    # 沿路径执行所有层（使用梯度检查点）
    for action in path:
        if action == ACTION_OUTPUT:
            # 输出动作：Final Norm + LM Head
            hidden = layered_loader.model.model.norm(hidden)
            last_hidden = hidden[:, -1, :]
            logits = layered_loader.model.lm_head(last_hidden)
            ce_loss = F.cross_entropy(logits, target_token.unsqueeze(0))
            return ce_loss
        else:
            # 跳转动作：直接执行目标层，action 0-18 → layer 9-27
            next_layer = action + model_config.fixed_layers  # 9 + [0-18] = [9-27]
            
            # 只执行下一层（不是所有中间层）
            if next_layer < len(layered_loader.layers):
                # 使用梯度检查点，不保存中间激活值
                hidden = torch.utils.checkpoint.checkpoint(
                    layer_forward,
                    hidden,
                    next_layer,
                    cos,
                    sin,
                    use_reentrant=False  # 推荐使用非重入模式
                )
    
    # [兜底] 如果路径结束仍未输出，强制计算loss
    hidden = layered_loader.model.model.norm(hidden)
    last_hidden = hidden[:, -1, :]
    logits = layered_loader.model.lm_head(last_hidden)
    ce_loss = F.cross_entropy(logits, target_token.unsqueeze(0))
    return ce_loss


# RoPE 管理
_rotary_emb = None

def init_rope(layered_loader, device: torch.device):
    """初始化 RoPE 编码器"""
    global _rotary_emb
    if _rotary_emb is not None:
        return
    
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    config = layered_loader.model.config
    
    # Qwen3使用rope_theta配置
    if not hasattr(config, 'rope_theta'):
        config.rope_theta = 1000000  # 默认值
    
    _rotary_emb = Qwen3RotaryEmbedding(config=config, device=device)
    print("RoPE 编码器已初始化")

def compute_rope(seq_len: int, device: torch.device, 
                hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 RoPE 位置编码
    
    Qwen3使用2D position_ids: [batch, seq_len]
    返回的cos/sin shape: [batch, seq_len, head_dim]
    """
    global _rotary_emb
    if _rotary_emb is None:
        raise RuntimeError("RoPE 未初始化，请先调用 init_rope()")
    
    # Qwen3需要2D position_ids: [batch, seq_len]
    batch_size = 1
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [1, seq_len]
    dummy_hidden = torch.randn(batch_size, seq_len, hidden_size, 
                              device=device, dtype=DEFAULT_DTYPE)
    cos, sin = _rotary_emb(dummy_hidden, position_ids)
    
    return cos, sin
