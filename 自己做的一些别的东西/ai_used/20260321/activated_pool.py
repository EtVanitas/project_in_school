"""激活池管理（单列表存储）"""

import torch
from typing import List, Tuple
from config import Config


class ActivatedPool:
    """激活池（单列表架构）"""
    
    def __init__(self, max_size=None):
        if max_size is None:
            max_size = Config.flow.ACTIVATED_POOL_MAX_SIZE
        self.items: List[Tuple[torch.Tensor, int, float]] = []
        self.max_size = max_size
        self.min_score: float = float('inf')  # 预存最低分
    
    def can_add(self, act_idx, score):
        """预占位检查（基于预存的最低分）"""
        if len(self.items) < self.max_size:
            return True
        return score > self.min_score
    
    def add(self, x, act_idx, score):
        """添加一项（超限淘汰最低分，自动更新 min_score）"""
        # 先检查是否需要淘汰（避免无效添加）
        if len(self.items) >= self.max_size and score <= self.min_score:
            return  # 分数不够高且池已满，直接拒绝
        
        # 添加新项
        self.items.append((x, act_idx, score))
        
        # 如果超限，淘汰最低分
        if len(self.items) > self.max_size:
            # 找到最低分的项并删除
            min_idx = min(range(len(self.items)), key=lambda i: self.items[i][2])
            del self.items[min_idx]
        
        # 更新最低分（只扫描一次）
        if self.items:
            self.min_score = min(item[2] for item in self.items)
        else:
            self.min_score = float('inf')
    
    def clear(self):
        """清空并返回所有项"""
        items = self.items.copy()
        self.items.clear()
        self.min_score = float('inf')  # 重置最低分
        return items
    
    def __len__(self):
        return len(self.items)
