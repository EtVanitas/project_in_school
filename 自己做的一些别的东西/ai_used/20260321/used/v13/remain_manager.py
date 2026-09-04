"""Remain 管理器（全局短期记忆管理）"""

import torch
import math
from collections import defaultdict
from config import Config


class RemainManager:
    """短期记忆管理器（CPU 存储）"""
    
    def __init__(self, num_activates=None, m: int = 1024, n: int = 1024):
        if num_activates is None:
            num_activates = Config.model.NUM_ACTIVATE_CLASSES
        self.num_activates = num_activates
        self.m = m
        self.n = n
        self.decay_factor = math.exp(-Config.remain.DECAY_RATE)
        # CPU 存储
        self.remains = [torch.zeros(m, n) for _ in range(num_activates)]
    
    def get_remain(self, act_idx, device):
        """获取指定 act_idx 的 remain（CPU→GPU 传输）"""
        remain_cpu = self.remains[act_idx]
        return remain_cpu.to(device)
    
    def update_batch(self, updates):
        """批量更新 remain（GPU→CPU→GPU 转换）"""
        groups = defaultdict(list)
        
        for act_idx, x_cpu, score_cpu in updates:
            groups[act_idx].append((x_cpu, score_cpu.item()))
        
        # 对每个 act_idx 做加权平均（CPU 操作）
        for act_idx, items in groups.items():
            xs_cpu = torch.stack([item[0] for item in items])
            scores = torch.tensor([item[1] for item in items])
            
            # softmax 权重
            weights = torch.softmax(scores, dim=0)
            
            # 加权平均
            weighted_x = torch.sum(weights[:, None, None] * xs_cpu, dim=0)
            
            # 直接相加（CPU 操作）
            self.remains[act_idx] += weighted_x
    
    def decay_all(self):
        """指数衰减所有 remain（CPU 操作，向量化）"""
        # 使用列表推导式 + 批量乘法（避免显式 for 循环）
        self.remains = [r * self.decay_factor for r in self.remains]
    
    def clear(self):
        """清空所有 remain（CPU 操作）"""
        for i in range(len(self.remains)):
            self.remains[i].zero_()
