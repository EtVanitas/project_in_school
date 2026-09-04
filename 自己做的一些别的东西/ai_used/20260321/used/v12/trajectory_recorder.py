"""
轨迹记录器（长期记忆）

存储 (x2_feature, remain) 轨迹，基于 L2 距离和使用频率检索长期记忆
"""

import torch
import torch.nn.functional as F
from typing import List
from config import Config


class TrajectoryRecorder:
    """轨迹记录器（CPU 存储，按需传输到 GPU）"""
    
    def __init__(self, m=1024, n=1024):
        """初始化"""
        self.m, self.n = m, n
        
        # 长期记忆库（CPU 存储）
        self.x2_memory_bank: List[torch.Tensor] = []
        self.remain_bank: List[torch.Tensor] = []
        self.usage_count: List[int] = [] # 使用次数
        
        # 容量控制
        self.max_memory_size = Config.flow.LONG_TERM_MEMORY_MAX_SIZE
        self.min_usage_threshold = Config.flow.LONG_TERM_MEMORY_MIN_USAGE
        self.usage_bonus_coefficient = Config.flow.LONG_TERM_MEMORY_USAGE_BONUS
    
    def record(self, x2_features, remains):
        """记录轨迹（x4 对应的 x2 和 remain）"""
        for i in range(x2_features.size(0)):
            x2_feat = x2_features[i].cpu().clone()
            remain = remains[i].cpu().clone()
            
            self.x2_memory_bank.append(x2_feat)
            self.remain_bank.append(remain)
            self.usage_count.append(0)  # 初始使用次数为 0
        
        # 内存管理：如果超出容量，删除使用次数少的
        if len(self.x2_memory_bank) > self.max_memory_size:
            self._prune_low_usage_memories()
    
    def query_long_term_memory(self, x2_query, topk=None):
        """长期记忆查询：基于 x2 特征相似度检索 remain"""
        if topk is None:
            topk = Config.flow.LONG_TERM_MEMORY_TOPK
        
        if len(self.x2_memory_bank) == 0:
            return torch.zeros_like(x2_query)
        
        # CPU -> GPU
        x2_bank = torch.stack(self.x2_memory_bank).to(x2_query.device)
        remain_bank = torch.stack(self.remain_bank).to(x2_query.device)
        usage_weights = torch.tensor(self.usage_count, device=x2_query.device).float()
        
        # Step 1: 计算 L2 距离
        distances = self._batch_l2_distance(x2_query, x2_bank)
        
        # Step 2: 结合使用次数加权
        # 使用次数越多，权重越大（即使距离稍远也优先考虑）
        usage_bonus = usage_weights.unsqueeze(0) * self.usage_bonus_coefficient
        weighted_distances = distances - usage_bonus  # 距离越小越好，所以减去 bonus
        
        # Step 3: TopK 选择
        topk_distances, topk_indices = torch.topk(
            -weighted_distances,  # 负距离（因为要最小距离）
            k=min(topk, len(self.x2_memory_bank)),
            dim=1
        )
        
        # Step 4: 加权平均 remain
        weights = 1.0 / (topk_distances.abs() + 1e-6)
        weights = F.softmax(weights, dim=1)
        
        batch_topk_remains = []
        for i in range(x2_query.size(0)):
            indices = topk_indices[i]
            topk_remains = remain_bank[indices]
            weighted_remain = torch.einsum(
                'ij,jkl->ikl',
                weights[i:i+1].transpose(0, 1),
                topk_remains
            ).squeeze(1)
            batch_topk_remains.append(weighted_remain)
        
        result = torch.stack(batch_topk_remains, dim=0)
        
        # Step 5: 更新使用次数
        self._update_usage_count(topk_indices)
        
        return result
    
    def _batch_l2_distance(self, x1, x2):
        """批量 L2 距离计算"""
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        
        # 使用 torch.cdist 高效计算
        dist = torch.cdist(x1_flat, x2_flat, p=2)
        return dist
    
    def _update_usage_count(self, topk_indices):
        """更新被选中轨迹的使用次数"""
        unique_indices, counts = torch.unique(
            topk_indices.flatten(),
            return_counts=True
        )
        
        for idx, count in zip(unique_indices.tolist(), counts.tolist()):
            if idx < len(self.usage_count):
                self.usage_count[idx] += count
    
    def _prune_low_usage_memories(self):
        """
        剪枝：保留使用频繁的轨迹（类似人类大脑的"重复强化"机制）
        """
        # 按使用次数排序（降序）
        sorted_indices = sorted(
            range(len(self.usage_count)),
            key=lambda i: self.usage_count[i],
            reverse=True
        )
        
        # 保留前 max_memory_size 个
        keep_indices = set(sorted_indices[:self.max_memory_size])
        
        self.x2_memory_bank = [self.x2_memory_bank[i] for i in keep_indices]
        self.remain_bank = [self.remain_bank[i] for i in keep_indices]
        self.usage_count = [self.usage_count[i] for i in keep_indices]
    
    def clear(self):
        """清空所有状态"""
        self.x2_memory_bank.clear()
        self.remain_bank.clear()
        self.usage_count.clear()
