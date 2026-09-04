"""RemainManager - Remain 生命周期管理（精简版）

功能：
1. 初始化（从长期记忆）
2. 更新（根据激活动态调整）
3. 衰减（随时间指数衰减）
4. 查询（获取用于激活判断的 remain）

存储策略：CPU 内存（节省显存），使用时临时复制到 GPU
"""

import torch
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class RemainManager:
    """Remain 生命周期管理器（含长期记忆）"""
    
    def __init__(self, num_classes: int = 72, m: int = 1024, n: int = 1024):
        self.num_classes = num_classes
        self.m, self.n = m, n
        self.remain_list: List[Tuple[int, torch.Tensor]] = []  # (age, cpu_tensor)
        self.long_term_memory: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.decay_rate_a = 0.3
    
    def initialize_from_long_term(self, long_term_memory: Dict[int, torch.Tensor]):
        """从长期记忆初始化所有 remain"""
        self.remain_list = []
        
        for act_idx in range(self.num_classes):
            if act_idx in long_term_memory:
                remain = long_term_memory[act_idx].clone().cpu()
            else:
                remain = torch.zeros(self.m, self.n)
            self.remain_list.append((0, remain))
        
        print(f"[RemainManager] Initialized {self.num_classes} remains from long-term memory")
    
    def update_delayed(self, activated_data: List[Tuple[int, torch.Tensor, float]]):
        """
        延迟更新 remain（在所有 x7 测试完成后调用）
        
        Args:
            activated_data: [(activate_idx, x_gpu, s), ...]
        """
        if len(activated_data) == 0:
            return
        
        # 按 activate_idx 分组
        grouped = defaultdict(list)
        for act_idx, x_gpu, s in activated_data:
            grouped[act_idx].append((x_gpu, s))
        
        # 批量更新每个 activate 的 remain
        for act_idx, xs_with_s in grouped.items():
            if act_idx >= len(self.remain_list):
                continue
            
            age, current_remain_cpu = self.remain_list[act_idx]
            total_s = sum(s for _, s in xs_with_s)
            
            if total_s > 0:
                device = xs_with_s[0][0].device
                weighted_sum_gpu = torch.zeros(self.m, self.n, device=device)
                
                for x_gpu, s in xs_with_s:
                    weight = s / total_s
                    weighted_sum_gpu += weight * x_gpu
                
                current_remain_gpu = current_remain_cpu.to(device)
                new_remain_cpu = (current_remain_gpu + weighted_sum_gpu).cpu()
                
                # 重置年龄为 0
                self.remain_list[act_idx] = (0, new_remain_cpu)
    
    def decay(self, iteration: int):
        """对所有 remain 应用指数衰减 e^{-a*(age+1)}"""
        new_remain_list = []
            
        for age, x_remain_cpu in self.remain_list:
            decay_factor = torch.exp(-torch.tensor(self.decay_rate_a * (age + 1)))
            aged_x_cpu = x_remain_cpu * decay_factor
            new_remain_list.append((age + 1, aged_x_cpu))
            
        self.remain_list = new_remain_list
    
    def get_remain_for_activation(self, act_idx: int) -> Optional[torch.Tensor]:
        """获取用于激活判断的 remain（复制到 GPU，且 detach）
        
        Returns:
            remain tensor on GPU (detached, no grad) or None if empty
        """
        if act_idx >= len(self.remain_list):
            return None
        
        _, x_remain_cpu = self.remain_list[act_idx]
        
        # 检查是否为空（全零）
        if x_remain_cpu is None or torch.all(x_remain_cpu == 0):
            return None
        
        # 移动到 GPU 并 detach（关键：阻断梯度流向 remain）
        # 注意：不指定具体设备，由调用方决定（通过 .to(device)）
        with torch.no_grad():
            remain_gpu = x_remain_cpu.detach()
        return remain_gpu
    
    def get_remain_value(self, act_idx: int) -> torch.Tensor:
        """
        获取 remain 值（用于轨迹记录）
        
        Returns:
            remain_value: 标量值（无梯度）
        """
        if act_idx >= len(self.remain_list):
            return torch.tensor(0.0)
        
        _, remain_cpu = self.remain_list[act_idx]
        
        # 返回平均值作为代表
        with torch.no_grad():
            return remain_cpu.mean()
    
    # ========== 长期记忆功能 ==========
    
    def store_trajectory(self, act_idx: int, x_cpu: torch.Tensor):
        """存储成功轨迹到长期记忆（限制最多 1000 条）"""
        self.long_term_memory[act_idx].append(x_cpu)
        
        max_store = 1000
        if len(self.long_term_memory[act_idx]) > max_store:
            self.long_term_memory[act_idx] = self.long_term_memory[act_idx][-max_store:]
    
    def retrieve_long_term(self, act_idx: int) -> Optional[torch.Tensor]:
        """检索长期记忆的平均模式"""
        if act_idx not in self.long_term_memory or len(self.long_term_memory[act_idx]) == 0:
            return None
        return torch.stack(self.long_term_memory[act_idx]).mean(dim=0)
    
    def update_epoch_end(self):
        """Epoch 结束后：从长期记忆中提取平均模式更新 remain"""
        print(f"[RemainManager] Updating remains from long-term memory...")
        
        updated_count = 0
        for act_idx in range(self.num_classes):
            long_term = self.retrieve_long_term(act_idx)
            if long_term is not None:
                self.remain_list[act_idx] = (0, long_term.clone().cpu())
                updated_count += 1
        
        print(f"[RemainManager] Updated {updated_count}/{self.num_classes} remains")
    
    def get_statistics(self) -> Dict:
        """获取 remain 使用统计"""
        ages = self.get_all_remain_ages()
        
        return {
            'total_remains': len(self.remain_list),
            'avg_age': sum(ages) / len(ages) if len(ages) > 0 else 0,
            'young_remains': sum(1 for age in ages if age < 5),
            'old_remains': sum(1 for age in ages if age > 20),
        }