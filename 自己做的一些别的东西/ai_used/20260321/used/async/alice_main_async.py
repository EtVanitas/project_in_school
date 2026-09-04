"""Alice 模型主文件（异步流水线版本）

特性:
- GPU/CPU 并行计算
- 异步记忆更新
- 预取优化
- 批量激活值计算

使用方式:
    # 同步版本（默认）
    from alice_main import MainModel
    
    # 异步版本（性能优化）
    from alice_main_async import MainModelAsync
"""

import torch
import torch.nn as nn
from typing import List, Dict
from config import Config
from alice_components import create_activates_and_reasons, ActivateTester
from activated_pool import ActivatedPool

# 异步版本组件
from remain_manager_async import RemainManagerAsync
from long_term_memory_async import LongTermMemoryAsync


class MainModelAsync(nn.Module):
    """
    Alice 主模型（异步流水线版本）
    """
    
    def __init__(self):
        """初始化模型和组件"""
        super().__init__()
        self.m, self.n = Config.model.M, Config.model.N
        self.max_iterations = Config.flow.MAX_ITERATIONS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建核心组件
        self.activates, self.reasons = create_activates_and_reasons()
        self.activate_tester = ActivateTester(self.activates)
        self.to(self.device)
        
        # 状态管理器（异步版本）
        self.activated_pool = ActivatedPool(max_size=Config.flow.ACTIVATED_POOL_MAX_SIZE)
        self.remain_manager = RemainManagerAsync(num_activates=Config.model.NUM_ACTIVATE_CLASSES, m=self.m, n=self.n)
        self.long_term_memory = LongTermMemoryAsync(m=self.m, n=self.n)
        
        # 输出管理
        self.output_lists: List[torch.Tensor] = []
        
        print(f"[MainModelAsync] 已初始化 (device={self.device})")
    
    def forward(self, initial_x: torch.Tensor, epoch: int = None):
        """
        前向传播（线性多轮迭代 + 异步流水线）
        
        Args:
            initial_x: (batch_size, m, n) - 3D 输入张量
            epoch: 当前训练 epoch（可选）
        
        Returns:
            output_lists: List[Tensor[m, n]] - Type3 Reason 的输出
        """
        # 初始化
        batch_size = initial_x.size(0)

        self.activated_pool.clear()
        self.output_lists.clear()
        self.remain_manager.clear()
        short_term_updates = []  # 用于更新短期记忆
        long_term_records = []   # 用于记录长期轨迹
        
        # ========== 第 1 轮：处理初始输入 ==========
        x_list = [initial_x[i] for i in range(batch_size)]
        self._process_inputs_async(x_list, short_term_updates, long_term_records)
        
        # 第 1 轮结束，更新短期记忆
        if short_term_updates:
            self.remain_manager.enqueue_batch_updates(short_term_updates)
            print(f"[记忆更新] 短期记忆：{len(short_term_updates)} 条（异步入队）")
        
        self.remain_manager.decay_all()
        
        # ========== 第 2-N 轮：循环处理上一轮的输出 ==========
        for iteration in range(1, self.max_iterations):
            # 从激活池取出上一轮的所有输出
            previous_outputs = self.activated_pool.clear()
            if len(previous_outputs) == 0:
                print(f"[终止] 第{iteration}轮：无新激活")
                break
            
            # 提取 x 值（忽略之前的 act_idx 和 score）
            x_values = [item[0] for item in previous_outputs]
            print(f"[第{iteration}轮] 处理 {len(x_values)} 个上一轮输出")
            
            # 异步处理
            self._process_inputs_async(x_values, short_term_updates, long_term_records)
            
            # 每轮结束，更新短期记忆
            if short_term_updates:
                self.remain_manager.enqueue_batch_updates(short_term_updates)
                print(f"[记忆更新] 短期记忆：{len(short_term_updates)} 条（异步入队）")
            
            self.remain_manager.decay_all()
            
            # 清空本轮的短期更新记录，准备下一轮
            short_term_updates = []
        
        # ========== 整个 forward 结束，记录长期记忆 ==========
        if Config.flow.ENABLE_LONG_TERM_MEMORY and long_term_records:
            for x2_cpu, remain_cpu in long_term_records:
                self.long_term_memory.record(x2_cpu, remain_cpu)
            print(f"[记忆更新] 长期记忆：{len(long_term_records)} 条")
        
        return self.output_lists
    
    def _process_inputs_async(self, x_list, updates, records):
        """
        异步版本：重叠 GPU 计算和 CPU/记忆操作
        
        优化策略:
        1. 批量计算所有激活值（一次矩阵乘法）
        2. 按分数分组，减少分支预测失败
        3. 并行处理高/中/低激活分支
        4. 后台预取长期记忆
        """
        if len(x_list) == 0:
            return
        
        for x in x_list:
            # ========== Step 1: 批量计算所有 72 个类别的激活值 ==========
            # TODO: 实现 compute_activation_batch 进一步提升性能
            # 当前先用循环（后续可优化）
            all_scores = []
            for act_idx in range(Config.model.NUM_ACTIVATE_CLASSES):
                score = self.activate_tester.compute_activation(x, act_idx)
                all_scores.append(score)
            
            # ========== Step 2: 按分数分组 ==========
            high_act_indices = []
            mid_act_indices = []
            
            for act_idx, score in enumerate(all_scores):
                if score >= Config.flow.HIGH_THRESHOLD:
                    high_act_indices.append(act_idx)
                elif score >= Config.flow.LOW_THRESHOLD:
                    mid_act_indices.append(act_idx)
                # 低激活的直接剪枝（不处理）
            
            print(f"  激活统计：高={len(high_act_indices)}, 中={len(mid_act_indices)}, 低={72-len(high_act_indices)-len(mid_act_indices)}")
            
            # ========== Step 3: 并行处理三个分支 ==========
            
            # --- 分支 A: 高激活（纯 GPU）---
            for act_idx in high_act_indices:
                score = all_scores[act_idx]
                
                if not self.activated_pool.can_add(act_idx, score):
                    continue
                
                # 异步入队短期记忆更新
                x_cpu = x.detach().cpu()
                score_cpu = score.cpu()
                self.remain_manager.enqueue_update(act_idx, x_cpu, score_cpu.item())
                
                # GPU 继续计算
                x_gated = self.activate_tester.apply_gate(x, act_idx)
                x_output = self.reasons[act_idx](x_gated)
                self.activated_pool.add(x_output, act_idx, score)
                
                # 只有 Type3 Reason 的输出才保存到输出列表
                if act_idx in Config.model.REASON_TYPE3_INDICES:
                    self.output_lists.append(x_output)
            
            # --- 分支 B: 中激活（需要 remain）---
            if mid_act_indices:
                # 预取长期记忆（后台线程）
                if Config.flow.ENABLE_LONG_TERM_MEMORY:
                    x_queries = [x] * len(mid_act_indices)
                    self.long_term_memory.prefetch(x_queries, mid_act_indices)
                
                # 处理每个中激活类别
                for act_idx in mid_act_indices:
                    # 获取短期记忆
                    remain_short = self.remain_manager.get_remain(act_idx, device=x.device)
                    
                    # dtype 转换（混合精度支持）
                    if x.dtype == torch.float16:
                        remain_short = remain_short.to(torch.float16)
                    
                    x2_with_short = x + remain_short.detach()
                    score_short = self.activate_tester.compute_activation(x2_with_short, act_idx)
                    
                    if score_short >= Config.flow.HIGH_THRESHOLD:
                        if self.activated_pool.can_add(act_idx, score_short):
                            updates.append((act_idx, x2_with_short.detach().cpu(), score_short.cpu()))
                            
                            x4_gated = self.activate_tester.apply_gate(x2_with_short, act_idx)
                            x4_output = self.reasons[act_idx](x4_gated)
                            self.activated_pool.add(x4_output, act_idx, score_short)
                            
                            if act_idx in Config.model.REASON_TYPE3_INDICES:
                                self.output_lists.append(x4_output)
                    
                    # 长期记忆分支
                    if Config.flow.ENABLE_LONG_TERM_MEMORY:
                        # 尝试从缓存获取
                        remain_long = self.long_term_memory.get_prefetched(x, act_idx)
                        
                        if remain_long is None:
                            # 缓存未命中，阻塞查询
                            remain_long = self.long_term_memory.query(x, topk=5)
                        
                        if x.dtype == torch.float16:
                            remain_long = remain_long.to(torch.float16)
                        
                        x2_with_long = x + remain_long.detach()
                        score_long = self.activate_tester.compute_activation(x2_with_long, act_idx)
                        
                        if score_long >= Config.flow.HIGH_THRESHOLD:
                            if self.activated_pool.can_add(act_idx, score_long):
                                updates.append((act_idx, x2_with_long.detach().cpu(), score_long.cpu()))
                                records.append((x.detach().cpu(), remain_short.detach().cpu()))
                                
                                x6_gated = self.activate_tester.apply_gate(x2_with_long, act_idx)
                                x6_output = self.reasons[act_idx](x6_gated)
                                self.activated_pool.add(x6_output, act_idx, score_long)
                                
                                if act_idx in Config.model.REASON_TYPE3_INDICES:
                                    self.output_lists.append(x6_output)
    
    def shutdown(self):
        """关闭异步组件（程序退出时调用）"""
        if hasattr(self, 'remain_manager'):
            self.remain_manager.shutdown()
        if hasattr(self, 'long_term_memory'):
            self.long_term_memory.shutdown()
    
    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except:
            pass
