"""批量状态管理 - 向量化版本

替代 XNode 列表的高效数据结构
用于支持完全向量化的前向传播
"""

import torch
from typing import List, Tuple


class BatchedState:
    """
    批处理的 X 状态（完全向量化）
    
    核心思想：Struct-of-Arrays 而不是 Array-of-Structs
    
    优势:
    1. 内存连续 (cache friendly)
    2. 批量操作 (SIMD)
    3. 无 Python 对象开销
    4. GPU 直接访问
    """
    
    def __init__(self, max_xs: int = 50, m: int = 1024, n: int = 1024):
        """
        Args:
            max_xs: 最大 X 数量（预分配容量）
            m: 序列长度
            n: 特征维度
        """
        self.max_xs = max_xs
        self.m = m
        self.n = n
        
        # ========== 核心数据（连续存储）==========
        self.xs_data = torch.zeros(max_xs, m, n)  # (K, m, n)
        
        # ========== 状态信息（整数 Tensor）==========
        self.ages = torch.zeros(max_xs, dtype=torch.int)           # (K,)
        self.status = torch.zeros(max_xs, dtype=torch.int8)        # (K,) 0=unactivated, 1=activated
        
        # ========== Manager 权限（布尔矩阵）==========
        self.allowed_managers = torch.zeros(max_xs, 8, dtype=torch.bool)  # (K, 8)
        self.manager_mask = torch.zeros(max_xs, 8, dtype=torch.float)     # (K, 8) 激活的 Manager
        
        # ========== x7 专用信息 ==========
        self.prev_act_idx = torch.full((max_xs,), -1, dtype=torch.int)    # (K,)
        self.prev_s_value = torch.zeros(max_xs)                           # (K,)
        
        # ========== 有效掩码（标记哪些 X 是有效的）==========
        self.valid_mask = torch.zeros(max_xs, dtype=torch.bool)  # (K,)
        
        # ========== 激活结果（临时存储）==========
        # 用于保存激活检测结果，避免重复计算
        self.direct_indices = torch.zeros(max_xs, dtype=torch.int)   # 直接激活的索引
        self.assist_indices = torch.zeros(max_xs, dtype=torch.int)   # 辅助激活的索引
        self.fail_indices = torch.zeros(max_xs, dtype=torch.int)     # 失败的索引
        self.direct_count = 0
        self.assist_count = 0
        self.fail_count = 0
    
    def reset(self):
        """重置状态（清空所有 X）"""
        self.valid_mask.zero_()
        self.direct_count = 0
        self.assist_count = 0
        self.fail_count = 0
    
    def count_valid(self) -> int:
        """获取有效 X 数量"""
        return self.valid_mask.sum().item()
    
    def initialize_inputs(self, xs: torch.Tensor):
        """
        初始化输入 X
        
        Args:
            xs: (batch_size, m, n) - 初始输入
        """
        batch_size = xs.size(0)
        assert batch_size <= self.max_xs, f"batch_size {batch_size} > max_xs {self.max_xs}"
        
        # 复制数据
        self.xs_data[:batch_size].copy_(xs)
        
        # 设置状态
        self.ages[:batch_size] = 0
        self.status[:batch_size] = 0  # unactivated
        self.prev_act_idx[:batch_size] = -1
        self.prev_s_value[:batch_size] = 0.0
        
        # 默认允许所有 Manager
        self.allowed_managers[:batch_size] = True
        
        # 设置有效掩码
        self.valid_mask[:batch_size] = True
    
    def mark_as_deleted(self, indices: torch.Tensor):
        """
        标记一批 X 为删除状态
        
        Args:
            indices: (N,) - 要删除的 X 索引
        """
        self.valid_mask[indices] = False
    
    def increment_ages(self):
        """对所有未激活的 X 增加年龄"""
        mask = self.valid_mask & (self.status == 0)  # unactivated
        self.ages[mask] += 1
    
    def get_timeout_mask(self, max_age: int = 3) -> torch.Tensor:
        """
        获取超时的 X 掩码
        
        Returns:
            timeout_mask: (K,) - 超时（age > max_age）的 X
        """
        return self.valid_mask & (self.ages > max_age)
    
    def set_manager_permissions(self, node_indices: torch.Tensor, manager_indices: List[int]):
        """
        设置指定 X 的 Manager 权限
        
        Args:
            node_indices: (N,) - X 索引
            manager_indices: 允许的 Manager 列表
        """
        self.allowed_managers[node_indices] = False
        if len(manager_indices) > 0:
            self.allowed_managers[node_indices, manager_indices] = True
    
    def get_active_x_indices(self) -> torch.Tensor:
        """获取所有有效 X 的索引"""
        return torch.nonzero(self.valid_mask, as_tuple=False).squeeze(-1)
    
    def get_active_xs(self) -> torch.Tensor:
        """获取所有有效 X 的数据"""
        indices = self.get_active_x_indices()
        return self.xs_data[indices]
    
    def pack_for_processing(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        打包有效 X 用于处理
        
        Returns:
            xs_packed: (K_valid, m, n) - 压缩后的有效 X
            original_indices: (K_valid,) - 对应的原始索引
        """
        indices = self.get_active_x_indices()
        xs_packed = self.xs_data[indices]
        return xs_packed, indices
    
    def __repr__(self):
        K = self.count_valid()
        return (f"BatchedState(valid={K}/{self.max_xs}, "
                f"direct={self.direct_count}, assist={self.assist_count}, "
                f"fail={self.fail_count})")
