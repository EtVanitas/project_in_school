"""奖励计算模块

提供状态构建和价值计算的共用工具函数
用于 Stage1A、Stage1B 和 Stage2
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from config import ACTION_OUTPUT, RLConfig, DEFAULT_DTYPE


class TokenProbabilityTracker:
    """跟踪token概率变化的管理器"""
    
    def __init__(self, max_tokens: int = 399):
        self.max_tokens = max_tokens
        self.token_positions = {}  # token_id -> position
        self.prob_vector = torch.zeros(max_tokens, dtype=DEFAULT_DTYPE)
    
    def update(self, current_probs: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """更新概率向量"""
        # 找出>threshold的token
        mask = current_probs > threshold
        selected_token_ids = torch.where(mask)[0]  # [num_selected]
        selected_probs = current_probs[selected_token_ids]
        
        # 1. 更新已存在的token概率 (即使<5%也保留原值)
        for token_id in list(self.token_positions.keys()):
            if token_id in selected_token_ids:
                pos = self.token_positions[token_id]
                idx = (selected_token_ids == token_id).nonzero(as_tuple=True)[0].item()
                self.prob_vector[pos] = selected_probs[idx]
            # 如果不在current中,保持原值不变
        
        # 2. 添加新token到空位
        for token_id, prob in zip(selected_token_ids.tolist(), selected_probs.tolist()):
            if token_id not in self.token_positions:
                if len(self.token_positions) < self.max_tokens:
                    pos = len(self.token_positions)
                    self.token_positions[token_id] = pos
                    self.prob_vector[pos] = prob
                # 如果满了,忽略新token (理论上不会发生)
        
        return self.prob_vector.clone()
    
    def reset(self):
        """重置追踪器 (新样本开始时调用)"""
        self.token_positions.clear()
        self.prob_vector.zero_()


def select_best_paths_and_collect_triples(paths: List[Tuple[int, ...]], 
                                          all_losses: List[List[float]],
                                          all_trajectories: List[List[List[Tuple[Dict, int]]]]):
    """为每个样本选择最优路径并收集所有路径的三元组（Stage1A 专用）"""
    batch_size = len(all_losses[0]) if all_losses else 0
    num_paths = len(paths)
    
    results = []
    all_triples = []
    
    for sample_idx in range(batch_size):
        # 提取该样本在所有path上的loss
        sample_losses = [all_losses[path_idx][sample_idx] for path_idx in range(num_paths)]
        
        # 找到最小loss的path索引
        best_path_idx = min(range(num_paths), 
                           key=lambda i: sample_losses[i] if sample_losses[i] is not None else float('inf'))
        best_path = paths[best_path_idx]
        best_ce = sample_losses[best_path_idx]
        
        # 为该样本的所有路径计算三元组（不只是最优路径）
        for path_idx in range(num_paths):
            path_trajectories = all_trajectories[path_idx][sample_idx]
            path_ce = sample_losses[path_idx]
            
            if path_ce is not None:
                triples = compute_path_values(path_trajectories, path_ce)
                all_triples.extend(triples)
        
        results.append({
            'best_path': best_path,
            'best_loss': best_ce,
        })
    
    return results, all_triples

def train_actor_critic_batch(actor_critic, rl_optimizer,
    batch: List[Dict], lm_head, device: torch.device, grad_clip: float = 1.0) -> Tuple[float, float, float]:
    """训练一个 batch 的 Actor-Critic（通用函数）"""
    if not batch:
        return 0.0, 0.0, 0.0
    
    # 提取批量数据
    hidden_states = torch.stack([t['state']['hidden'] for t in batch]).to(device, dtype=DEFAULT_DTYPE)
    layer_indices = torch.stack([t['state']['layer_idx'] for t in batch]).to(device, dtype=DEFAULT_DTYPE)
    step_idxs = torch.stack([t['state']['step_idx'] for t in batch]).to(device, dtype=DEFAULT_DTYPE)
    context_lengths = torch.stack([t['state']['seq_len'] for t in batch]).to(device, dtype=DEFAULT_DTYPE)
    
    actions = torch.tensor([t['action'] for t in batch], device=device, dtype=torch.long)
    target_values = torch.tensor([t['value'] for t in batch], device=device, dtype=DEFAULT_DTYPE).unsqueeze(1)
    
    # 将隐藏状态投影到词表并softmax
    with torch.no_grad():
        logits = lm_head(hidden_states)
        vocab_probs = F.softmax(logits, dim=-1)  # [batch, vocab_size]
    
    # 清零梯度
    rl_optimizer.zero_grad()
    
    # 前向传播：输入改为词表概率
    action_logits, predicted_values = actor_critic(
        vocab_probs=vocab_probs.to(DEFAULT_DTYPE),  # [batch, vocab_size]
        layer_index=layer_indices.to(DEFAULT_DTYPE),
        context_length=context_lengths.to(DEFAULT_DTYPE),
        step_idx=step_idxs.to(DEFAULT_DTYPE)
    )
    
    # 计算 Actor Loss
    log_probs = F.log_softmax(action_logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
    
    advantage = target_values - predicted_values.detach()
    
    if advantage.numel() > 1:
        adv_mean = advantage.mean()
        adv_std = advantage.std() + 1e-8
        normalized_advantage = (advantage - adv_mean) / adv_std
    else:
        normalized_advantage = advantage
    
    actor_loss = -(selected_log_probs * normalized_advantage).mean()
    
    # 计算 Critic Loss
    critic_loss = F.mse_loss(predicted_values, target_values)
    
    # 根据两个 loss 的比值自动调整权重，避免 Critic 主导梯度
    with torch.no_grad():
        actor_abs = actor_loss.abs() + 1e-8
        critic_abs = critic_loss.abs() + 1e-8
        ratio = critic_abs / actor_abs
        # 限制权重范围 [0.0001, 1.0]
        critic_weight = torch.clamp(1.0 / ratio, min=0.0001, max=1.0).item()
    
    # 联合损失（动态权重）
    total_loss = actor_loss + critic_weight * critic_loss
    
    # 调试信息：每 100 次调用输出一次权重
    if not hasattr(train_actor_critic_batch, '_call_count'):
        train_actor_critic_batch._call_count = 0
    train_actor_critic_batch._call_count += 1
    if train_actor_critic_batch._call_count % 100 == 0:
        print(f"    [Loss平衡] Actor={actor_loss.item():.4f}, Critic={critic_loss.item():.4f}, "
              f"CriticWeight={critic_weight:.4f}, Total={total_loss.item():.4f}")
    
    # 反向传播
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), grad_clip)
    
    # 更新优化器
    rl_optimizer.step()
    
    return actor_loss.item(), critic_loss.item(), total_loss.item()

def rebuild_base_probs_batch(batch_base_data: List[Dict], vocab_size: int, 
                             device: torch.device) -> List[torch.Tensor]:
    """批量重建基准概率分布（Stage1A 和 Stage1B 共用）"""
    batch_base_probs = []
    for base_data in batch_base_data:
        base_logits_full = torch.zeros(vocab_size, device=device, dtype=DEFAULT_DTYPE)
        base_logits_full.scatter_(0, base_data['top_indices'].to(device), 
                                 base_data['top_logits'].to(device))
        base_probs = F.softmax(base_logits_full, dim=-1)
        batch_base_probs.append(base_probs)
    
    return batch_base_probs

def compute_path_values(trajectories: List[Tuple[Dict, int]], total_ce: float) -> List[Dict]:
    """为单条路径计算价值（线性分配：终点奖励 - 步数惩罚）"""
    # 从配置获取步数惩罚
    rl_config = RLConfig()
    step_penalty = rl_config.step_penalty
    
    n_steps = len(trajectories)
    triples = []
    
    # 依次计算每个步骤的价值
    for i in range(n_steps):
        state, action = trajectories[i]
        
        if action == ACTION_OUTPUT:
            # 输出动作：价值 = -交叉熵
            value = -total_ce
        else:
            # 继续动作：价值 = 终点奖励 - (到输出的步数 × 步数惩罚)
            steps_to_output = n_steps - 1 - i  # 距离输出还有多少步
            value = -total_ce - step_penalty * steps_to_output
        
        triples.append({
            'state': state,
            'action': action,
            'value': value
        })
        
    return triples

def build_state_dict(hidden_state: torch.Tensor, layer_idx: int, step_idx: int, seq_len: int) -> Dict:
    """构建状态字典（不包含跳转标志）"""
    return {
        'hidden': hidden_state.cpu(),
        'layer_idx': torch.tensor(layer_idx),
        'step_idx': torch.tensor(step_idx),
        'seq_len': torch.tensor(seq_len)
    }


# RoPE 管理（类级别缓存）
_rope_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
_rotary_emb = None  # 延迟初始化

def init_rope(layered_loader, device: torch.device):
    """初始化 RoPE 编码器（延迟初始化，模块级别共享）"""
    global _rotary_emb

    if _rotary_emb is not None:
        return  # 已初始化
    
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    config = layered_loader.model.config
    if not hasattr(config, 'rope_parameters'):
        config.rope_parameters = {
            'rope_type': 'default',
            'rope_theta': 10000000,
            'partial_rotary_factor': 0.25
        }
    
    _rotary_emb = Qwen3_5TextRotaryEmbedding(config=config, device=device)
    print(f"✓ RoPE 编码器已初始化")

def precompute_rope(seq_len: int, device: torch.device, 
                   hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 位置编码（带缓存，模块级别共享）"""
    global _rotary_emb
    
    # 检查缓存
    if seq_len in _rope_cache:
        return _rope_cache[seq_len]
    
    # 确保 RoPE 已初始化
    if _rotary_emb is None:
        raise RuntimeError("RoPE 未初始化，请先调用 init_rope()")
    
    # 计算 RoPE
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_hidden = torch.randn(1, seq_len, hidden_size, 
                              device=device, dtype=DEFAULT_DTYPE)
    cos, sin = _rotary_emb(dummy_hidden, position_ids)
    
    # 缓存结果
    _rope_cache[seq_len] = (cos, sin)
    
    return cos, sin

def clear_rope_cache():
    """清空 RoPE 缓存（释放显存，模块级别）"""
    global _rope_cache
    _rope_cache.clear()
    torch.cuda.empty_cache()
