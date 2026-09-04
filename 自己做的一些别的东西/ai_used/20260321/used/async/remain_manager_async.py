"""Remain 管理器（异步流水线版本）

特性:
- CPU 存储 + 后台线程池更新
- 双缓冲机制
- 非阻塞入队
- 自动合并待处理更新

使用方式:
    # 同步版本（默认）
    from remain_manager import RemainManager
    
    # 异步版本（性能优化）
    from remain_manager_async import RemainManagerAsync
"""

import torch
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple, Dict, Optional
from config import Config


class RemainManagerAsync:
    """短期记忆管理器（CPU 存储 + 异步流水线更新）"""
    
    def __init__(self, num_activates=None, m: int = 1024, n: int = 1024):
        if num_activates is None:
            num_activates = Config.model.NUM_ACTIVATE_CLASSES
        self.num_activates = num_activates
        self.m = m
        self.n = n
        
        # ========== CPU 存储（双缓冲机制）==========
        self.remains = [torch.zeros(m, n) for _ in range(num_activates)]
        
        # 待处理的更新（pending pool）
        # pending_updates[act_idx] = List[(x_cpu, score, timestamp)]
        self.pending_updates: Dict[int, List[Tuple[torch.Tensor, float, int]]] = \
            defaultdict(list)
        
        # ========== 异步组件 ==========
        # 后台线程池（2 个工作线程）
        self.executor = ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix="remain_worker"
        )
        
        # 正在进行的异步任务
        self.pending_futures: List[Future] = []
        
        # 锁保护
        self.lock = threading.Lock()
        
        # 全局时间戳（用于合并策略）
        self._timestamp_counter = 0
        
        print(f"[RemainManagerAsync] 已初始化 (num_activates={num_activates})")
    
    def get_remain(self, act_idx: int, device: torch.device) -> torch.Tensor:
        """
        获取 remain（立即返回，自动应用 pending 更新）
        
        Args:
            act_idx: activate 类别索引
            device: 目标设备（GPU/CPU）
        
        Returns:
            remain tensor（已包含所有已完成的更新）
        """
        with self.lock:
            # 先应用所有待处理的更新（同步点）
            self._apply_pending_updates(act_idx)
            
            # 返回 remain（可能需要 GPU→CPU 传输）
            remain_cpu = self.remains[act_idx]
            return remain_cpu.to(device)
    
    def _apply_pending_updates(self, act_idx: int):
        """
        应用指定 act_idx 的所有待处理更新
        
        策略：加权平均（新更新的权重更高）
        """
        pending = self.pending_updates.get(act_idx, [])
        if not pending:
            return
        
        # 提取所有更新
        xs_cpu = torch.stack([item[0] for item in pending])  # (k, m, n)
        scores = torch.tensor([item[1] for item in pending])  # (k,)
        timestamps = torch.tensor([item[2] for item in pending])  # (k,)
        
        # 时间衰减权重：越新的权重越高
        time_weights = torch.exp(-(timestamps.max() - timestamps) / 10.0)
        
        # 综合权重 = score 权重 × 时间权重
        combined_weights = torch.softmax(scores, dim=0) * time_weights
        combined_weights /= combined_weights.sum()  # 归一化
        
        # 加权平均
        weighted_x = torch.sum(
            combined_weights[:, None, None] * xs_cpu, 
            dim=0
        )  # (m, n)
        
        # 应用到 remain
        self.remains[act_idx] += weighted_x
        
        # 清空待处理队列
        self.pending_updates[act_idx] = []
    
    def enqueue_update(self, act_idx: int, x_cpu: torch.Tensor, score_cpu: float):
        """
        异步入队更新（非阻塞，立即返回）
        
        Args:
            act_idx: activate 类别索引
            x_cpu: CPU 上的 x 张量（已 detach）
            score_cpu: CPU 上的激活分数
        """
        with self.lock:
            self._timestamp_counter += 1
            timestamp = self._timestamp_counter
            
            # 添加到待处理队列
            self.pending_updates[act_idx].append((x_cpu, score_cpu, timestamp))
        
        # 可选：如果队列太长，触发后台清理
        if len(self.pending_updates[act_idx]) > 10:
            self._trigger_async_merge(act_idx)
    
    def _trigger_async_merge(self, act_idx: int):
        """触发后台合并任务（防止队列过长）"""
        future = self.executor.submit(self._merge_pending_updates, act_idx)
        self.pending_futures.append(future)
        
        # 清理已完成的任务
        self.pending_futures = [f for f in self.pending_futures if not f.done()]
    
    def _merge_pending_updates(self, act_idx: int):
        """
        后台线程：合并待处理的更新
        
        策略：只保留最新的 K 个（防止内存爆炸）
        """
        with self.lock:
            pending = self.pending_updates.get(act_idx, [])
            if len(pending) <= 10:
                return  # 不需要合并
            
            # 按时间戳排序，保留最新的 10 个
            pending.sort(key=lambda x: x[2], reverse=True)
            self.pending_updates[act_idx] = pending[:10]
    
    def enqueue_batch_updates(self, updates: List[Tuple[int, torch.Tensor, float]]):
        """
        批量异步入队（推荐用法）
        
        Args:
            updates: [(act_idx, x_cpu, score), ...]
        """
        # 按 act_idx 分组
        groups = defaultdict(list)
        for act_idx, x_cpu, score in updates:
            groups[act_idx].append((x_cpu, score.item()))
        
        # 批量提交到线程池
        for act_idx, items in groups.items():
            # 合并同组更新
            xs_cpu = torch.stack([item[0] for item in items])
            scores = torch.tensor([item[1] for item in items])
            weights = torch.softmax(scores, dim=0)
            merged_x = torch.sum(weights[:, None, None] * xs_cpu, dim=0)
            merged_score = scores.max().item()
            
            # 入队
            self.enqueue_update(act_idx, merged_x, merged_score)
    
    def wait_all(self):
        """
        等待所有异步更新完成（用于 epoch 结束或关键同步点）
        """
        for future in self.pending_futures:
            future.result()
        self.pending_futures.clear()
    
    def decay_all(self):
        """指数衰减（同步操作，确保安全）"""
        # 先等待所有异步更新
        self.wait_all()
        
        with self.lock:
            decay_factor = math.exp(-Config.remain.DECAY_RATE)
            for i in range(len(self.remains)):
                self.remains[i] *= decay_factor
    
    def clear(self):
        """清空所有 remain（同步操作）"""
        self.wait_all()
        
        with self.lock:
            for i in range(len(self.remains)):
                self.remains[i].zero_()
            self.pending_updates.clear()
    
    def shutdown(self):
        """关闭线程池（程序退出时调用）"""
        self.wait_all()
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass
