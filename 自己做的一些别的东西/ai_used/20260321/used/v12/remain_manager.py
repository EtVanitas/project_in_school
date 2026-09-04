"""
Remain 管理器（短期记忆）

管理每个 label 的 72 个 Remain，支持批量查询、更新和指数衰减
"""

import torch
import math
from typing import Dict, List, Tuple
from collections import defaultdict
from config import Config


class RemainManager:
    """Remain 管理器（CPU 存储，按需传输到 GPU）"""
    
    def __init__(self, m: int = 1024, n: int = 1024):
        """初始化"""
        self.m = m
        self.n = n
        self.remain_groups_cpu: Dict[int, List[torch.Tensor]] = {}
    
    def initialize_remains_for_label(self, label: int):
        """为指定 label 初始化 Remain"""
        num_act = Config.model.NUM_ACTIVATE_CLASSES
        if label not in self.remain_groups_cpu:
            self.remain_groups_cpu[label] = [
                torch.zeros(self.m, self.n) for _ in range(num_act)
            ]
    
    def get_remains_batched(self, labels, act_idxs):
        """批量获取 Remain（短期记忆）"""
        num_act = Config.model.NUM_ACTIVATE_CLASSES
        N = len(labels)
        device = labels.device
        
        # GPU -> CPU
        labels_cpu = labels.cpu()
        act_idxs_cpu = act_idxs.cpu()
        
        # 按 label 分组
        unique_labels = labels_cpu.unique().tolist()
        remains = torch.zeros(N, self.m, self.n, device='cpu')
        
        # 分批处理
        for label in unique_labels:
            mask = labels_cpu == label
            if label in self.remain_groups_cpu:
                label_remains = torch.stack(self.remain_groups_cpu[label])  # (num_act, m, n)
                label_act_idxs = act_idxs_cpu[mask]
                remains[mask] = label_remains[label_act_idxs]
        
        # CPU -> GPU
        return remains.to(device)
    
    def update_remains_batched(self, activated_data):
        """批量更新 Remain（直接相加 + 加权平均）"""
        num_act = Config.model.NUM_ACTIVATE_CLASSES
        
        xs = activated_data['xs']
        labels = activated_data['labels']
        act_idxs = activated_data['act_idxs']
        scores = activated_data['scores']
        
        # 按 (label, act_idx) 分组
        group_ids = labels * num_act + act_idxs
        unique_group_ids = group_ids.unique().tolist()
        
        for group_id in unique_group_ids:
            label = group_id // num_act
            act_idx = group_id % num_act
            
            self.initialize_remains_for_label(label)
            
            # 找到属于这个组的所有 x
            mask = group_ids == group_id
            group_xs = xs[mask]
            group_scores = scores[mask]
            
            # 计算权重（softmax）
            weights = torch.softmax(group_scores, dim=0)
            
            # 加权平均
            weighted_x = torch.sum(weights[:, None, None] * group_xs, dim=0)
            
            # 直接相加
            self.remain_groups_cpu[label][act_idx] += weighted_x
    
    def decay_all_remains(self):
        """指数衰减所有 Remain"""
        decay_factor = math.exp(-Config.remain.DECAY_RATE)
        for label, remains in self.remain_groups_cpu.items():
            for i in range(len(remains)):
                remains[i] *= decay_factor
    
    def clear(self):
        """清空所有状态"""
        self.remain_groups_cpu.clear()
