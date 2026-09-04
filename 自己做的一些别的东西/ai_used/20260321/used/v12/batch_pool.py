"""
动态批处理池

使用三个并行列表管理 x 数据，支持：
1. 动态容量控制（防止显存爆炸）
2. 批量添加/提取（向量化操作）
3. 按 s 值排序（溢出时保留高质量）
"""

import torch
from typing import List, Tuple

class BatchPool:
    """动态批处理池（GPU 存储）"""
    
    def __init__(self, max_size: int = 500, m: int = 1024, n: int = 1024, device: str = 'cuda'):
        """初始化"""
        self.max_size = max_size  # 最大容量
        self.m, self.n = m, n
        self.device = torch.device(device)
        
        self.xs_list: List[torch.Tensor] = []
        self.labels_list: List[int] = []
        self.activate_indices_list: List[int] = []
    
    def __len__(self) -> int:
        return len(self.xs_list)
    
    def extend_with_scores(self, xs, labels, act_idxs, scores):
        """批量添加张量，按 s 值排序保留高质量的"""
        k = len(xs)
        remaining = self.max_size - len(self.xs_list)
        
        # 如果没有溢出，直接添加（不排序）
        if k <= remaining:
            self.extend(xs, labels, act_idxs)
            return k, 0
        
        # 溢出时：按 s 值排序，保留最高的
        sorted_indices = torch.argsort(scores, descending=True)[:remaining]
                
        xs_sorted = xs[sorted_indices]
        labels_sorted = labels[sorted_indices]
        act_idxs_sorted = act_idxs[sorted_indices]
        
        self.extend(xs_sorted, labels_sorted, act_idxs_sorted)
        
        discarded_count = k - remaining
        if remaining > 0:
            print(f"[BatchPool] 按 s 值筛选：保留 {remaining}/{k}个 "
                  f"(丢弃{discarded_count}个，最低 s={scores[sorted_indices[-1]].item():.3f})")
        else:
            print(f"[BatchPool] 容量已满，全部丢弃 {k}/{k}个")
        
        return remaining, discarded_count
    
    def extend(self, xs, labels, act_idxs):
        """批量添加张量（溢出时丢弃超出的部分）"""
        k = len(xs)
        
        # 检查是否会溢出
        if len(self.xs_list) + k > self.max_size:
            overflow_count = len(self.xs_list) + k - self.max_size
            keep_count = k - overflow_count
            
            if keep_count <= 0:
                print(f"[BatchPool] 溢出警告：尝试添加{k}个 x，但容量只剩{self.max_size - len(self.xs_list)}，全部丢弃")
                return
            
            # 截取不溢出的部分
            xs = xs[:keep_count]
            labels = labels[:keep_count]
            act_idxs = act_idxs[:keep_count]
            print(f"[BatchPool] 溢出处理：丢弃 {overflow_count}/{k} 个新来的 x")
        
        # 确保 xs 在正确的设备上
        if xs.device != self.device:
            xs = xs.to(self.device)
        
        # labels 和 act_idxs 转为 Python list
        self.xs_list.extend(xs.unbind(0))
        self.labels_list.extend(labels.tolist())
        self.activate_indices_list.extend(act_idxs.tolist())
    
    def clear(self, device=None):
        """清空池子并返回所有数据"""
        if len(self.xs_list) == 0: # 空池子返回
            target_device = device if device else self.device
            return (
                torch.empty(0, self.m, self.n, device=target_device),
                torch.empty(0, dtype=torch.long, device=target_device),
                torch.empty(0, dtype=torch.long, device=target_device)
            )
        
        # 移到指定设备
        target_device = device if device else self.device
        xs = torch.stack([x.to(target_device) for x in self.xs_list])
        labels = torch.tensor(self.labels_list, dtype=torch.long, device=target_device)
        act_idxs = torch.tensor(self.activate_indices_list, dtype=torch.long, device=target_device)
        
        # 清空
        self.xs_list.clear()
        self.labels_list.clear()
        self.activate_indices_list.clear()
        
        return xs, labels, act_idxs
