"""
Alice 模型主文件

核心组件:
- BatchPool: 动态批处理池
- RemainManager: 短期记忆管理
- TrajectoryRecorder: 轨迹记录 + 长期记忆查询
- Activate-Reason 分离架构
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from config import Config
from alice_components import create_activates_and_reasons, BatchActivateTester
from batch_pool import BatchPool
from remain_manager import RemainManager
from trajectory_recorder import TrajectoryRecorder


class MainModel(nn.Module):
    """Alice 主模型（动态批处理 + 双记忆系统）"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化参数
        self.m, self.n = Config.model.M, Config.model.N
        self.max_iterations = Config.flow.MAX_ITERATIONS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建组件
        self.activates, self.reasons = create_activates_and_reasons()
        self.batch_activate_tester = BatchActivateTester(self.activates)
        self.to(self.device)
        
        # 两个批处理池
        self.activated_pool = BatchPool(max_size=Config.flow.MAX_ACTIVATED_X, m=self.m, n=self.n, device=str(self.device))
        self.pending_pool = BatchPool(max_size=Config.flow.MAX_UNACTIVATED_X, m=self.m, n=self.n, device=str(self.device))
        
        # Remain 管理器（短期记忆）
        self.remain_manager = RemainManager(m=self.m, n=self.n)
        
        # 轨迹记录器（长期记忆）
        self.trajectory_recorder = TrajectoryRecorder(m=self.m, n=self.n)
        
        # 统计
        self.activate_usage_count = [0] * Config.model.NUM_ACTIVATE_CLASSES
        
        # 输出管理
        self.output_lists: Dict[int, List[torch.Tensor]] = {}
        
        # 缓存阈值（注册为 buffer，会自动跟随模型设备）
        r_high, r_low = Config.model.get_threshold_for_stage(Config.model.CURRENT_STAGE)
        self.register_buffer('_cached_r_high', r_high)
        self.register_buffer('_cached_r_low', r_low)
    
    def forward(self, initial_xs: torch.Tensor, epoch: int = None):
        """
        前向传播
        
        Args:
            initial_xs: (batch_size, m, n)
            epoch: 当前训练 epoch
        
        Returns:
            output_lists: Dict[int, List[Tensor]] 按 label 分组的输出
            stats: Dict 统计信息
        """
        try:
            # 维度检查
            assert initial_xs.dim() == 3, f"Expected 3D input, got {initial_xs.dim()}D"
            assert initial_xs.size(1) == self.m, f"Expected dim1={self.m}, got {initial_xs.size(1)}"
            assert initial_xs.size(2) == self.n, f"Expected dim2={self.n}, got {initial_xs.size(2)}"
            
            batch_size = initial_xs.size(0)
            device = initial_xs.device
            
            # 初始化
            self.activated_pool.clear()
            self.pending_pool.clear()
            self.output_lists.clear()
            self.remain_manager.clear()
            
            # 初始化输入（activate_idx=-1 表示新输入）
            self.activated_pool.extend(
                initial_xs,                              # (batch, m, n)
                torch.arange(batch_size, device=device), # labels: [0, 1, ..., batch-1]
                torch.full((batch_size,), -1, device=device)  # act_idxs: [-1, -1, ..., -1]
            )
            
            # 初始化每个 label 的 remain 组
            for label in range(batch_size):
                self.remain_manager.initialize_remains_for_label(label)
            
            # 主循环
            for iteration in range(self.max_iterations):

                # Step 1: 清空激活池并处理（全是 act_idx=-1 的新输入）
                new_xs, new_labels, new_act_idxs = self.activated_pool.clear(device=device)
                
                # 初始化 Remain（如果是第一次遇到这个 label）
                for label in torch.unique(new_labels):
                    self.remain_manager.initialize_remains_for_label(label.item())
                
                # Step 2: Activate 检测（重复 num_act 次）
                num_act = Config.model.NUM_ACTIVATE_CLASSES
                n_new = len(new_xs)
                xs_test = new_xs.repeat_interleave(num_act, dim=0)      # (n_new*num_act, m, n)
                act_idx_test = torch.arange(num_act, device=device).repeat(n_new)
                labels_test = new_labels.repeat_interleave(num_act)
                
                # Step 3: BatchActivateTester 处理
                (x1_processed, x1_labels, x1_act_idxs, s_x1_norm,
                 x2_xs, x2_labels, x2_act_idxs, s_x2_norm) = self.batch_activate_tester(
                    xs_test, labels_test, act_idx_test,
                    r_high=self._cached_r_high,
                    r_low=self._cached_r_low
                )
                
                # Step 4: x1 进入激活池
                self.activated_pool.extend_with_scores(x1_processed, x1_labels, x1_act_idxs, s_x1_norm)

                # Step 5: x2 + pending → Remain 辅助
                pending_xs, pending_labels, pending_act_idxs = self.pending_pool.clear(device=device)

                (x4_processed, x4_labels, x4_act_idxs, s_x4_norm,
                #  x6_xs, x6_labels, x6_act_idxs, s_x6_norm,
                 x5_xs, x5_labels, x5_act_idxs, s_x5_norm) = self._process_with_remain_vectorized(
                    x2_xs, x2_labels, x2_act_idxs,
                    pending_xs, pending_labels, pending_act_idxs,
                    device
                )
                
                # Step 6: x4 进入激活池
                self.activated_pool.extend_with_scores(x4_processed, x4_labels, x4_act_idxs, s_x4_norm)

                # Step 6.5: x6 进入激活池
                self.activated_pool.extend_with_scores(x6_xs, x6_labels, x6_act_idxs, s_x6_norm)

                # Step 7: x5 进入待激活池
                self.pending_pool.extend_with_scores(x5_xs, x5_labels, x5_act_idxs, s_x5_norm)
                
                # Step 8: 收集激活数据用于 Remain 更新
                # 合并 x1 和 x4
                all_activated_xs = torch.cat([x1_processed, x4_processed], dim=0).detach().cpu()
                all_activated_labels = torch.cat([x1_labels, x4_labels], dim=0).cpu()
                all_activated_act_idxs = torch.cat([x1_act_idxs, x4_act_idxs], dim=0).cpu()
                all_activated_scores = torch.cat([s_x1_norm.detach().cpu(), s_x4_norm.detach().cpu()], dim=0)
                
                activated_data = {
                    'xs': all_activated_xs,
                    'labels': all_activated_labels,
                    'act_idxs': all_activated_act_idxs,
                    'scores': all_activated_scores
                }
                self.remain_manager.update_remains_batched(activated_data)
                
                # Step 10: Reason 处理
                reason_xs, reason_labels, reason_act_idxs = self.activated_pool.clear(device=device)
                
                # 情况 1：激活池为空 → 终止
                if len(reason_xs) == 0:
                    print(f"[终止] 激活池为空，无 x 可处理")
                    break
                
                # 情况 2：Reason 处理
                next_xs = self._process_reasons_batched(reason_xs, reason_labels, reason_act_idxs, device)
                
                # 情况 2a：Reason 后没有产生新 x → 终止
                if len(next_xs) == 0:
                    break
                
                # 情况 2b：Reason 产生了新 x → 加入激活池继续
                next_xs_tensor, next_labels = next_xs
                next_act_idxs = torch.full((len(next_xs_tensor),), -1, device=device)
                self.activated_pool.extend(next_xs_tensor, next_labels, next_act_idxs)
                
                # Step 11: 指数衰减 Remain
                # 每轮调用一次，相当于乘以 e^(-0.3) ≈ 0.7408
                self.remain_manager.decay_all_remains()
            
            # 返回输出和统计
            stats = {
                'iterations': iteration + 1,
                'activate_usage': self.activate_usage_count.copy(),
                'final_x_count': len(self.activated_pool) + len(self.pending_pool),
                'output_counts': {label: len(outputs) for label, outputs in self.output_lists.items()},
                'total_outputs': sum(len(lst) for lst in self.output_lists.values()),
                'pool_stats': {
                    'activated': len(self.activated_pool),
                    'pending': len(self.pending_pool)
                }
            }
            
            return self.output_lists, stats
            
        except AssertionError as e:
            print(f"[Error] Input validation failed: {e}")
            raise
        except RuntimeError as e:
            print(f"[Error] Forward failed: {e}")
            raise
    
    def _process_with_remain_vectorized(self, x2_xs, x2_labels, x2_act_idxs,
                                         pending_xs, pending_labels, pending_act_idxs,
                                         device):
        """
        x2 + pending → Remain 辅助（向量化）, 得到 x4, x5, x6
        """
        # Step 1: 合并 x2 和 pending
        if len(pending_xs) > 0:
            all_to_test_xs = torch.cat([x2_xs, pending_xs], dim=0)
            all_to_test_labels = torch.cat([x2_labels, pending_labels.to(device)], dim=0)
            all_to_test_act_idxs = torch.cat([x2_act_idxs, pending_act_idxs.to(device)], dim=0)
        else:
            all_to_test_xs = x2_xs
            all_to_test_labels = x2_labels
            all_to_test_act_idxs = x2_act_idxs
        
        if len(all_to_test_xs) == 0:
            return (torch.empty(0, self.m, self.n), torch.empty(0), torch.empty(0),
                    torch.empty(0, self.m, self.n), torch.empty(0), torch.empty(0),
                    torch.empty(0, self.m, self.n), torch.empty(0), torch.empty(0))
        
        # Step 2: 获取短期记忆
        short_remains = self.remain_manager.get_remains_batched(all_to_test_labels, all_to_test_act_idxs)
        short_remains = short_remains.detach() # 统计规律，不需要梯度
        
        # Step 3: 获取长期记忆
        long_remains = self.trajectory_recorder.query_long_term_memory(
            x2_query=all_to_test_xs,
            topk=Config.flow.LONG_TERM_MEMORY_TOPK
        )
        long_remains = long_remains.detach() # 历史模式，不需要梯度
        
        # Step 4: 双记忆处理
        xs_with_short = all_to_test_xs + short_remains
        (x4_processed, x4_labels, x4_act_idxs, s_x4_norm,
         x5_xs, x5_labels, x5_act_idxs, s_x5_norm,
         mask_x4) = self.batch_activate_tester(
            xs_with_short, all_to_test_labels, all_to_test_act_idxs,
            r_high=self._cached_r_high,
            r_low=self._cached_r_low,
            return_mask=True  # mask 用于轨迹记录
        )
        
        # 长期记忆分支（不需要梯度）
        xs_with_long = all_to_test_xs + long_remains
        with torch.no_grad():
            (x6_xs, x6_labels, x6_act_idxs, s_x6_norm,
            x7_xs, x7_labels, x7_act_idxs, s_x7_norm, # x7 丢弃
            mask_x6) = self.batch_activate_tester(
                xs_with_long, all_to_test_labels, all_to_test_act_idxs,
                r_high=self._cached_r_high,
                r_low=self._cached_r_low,
                return_mask=True
            )
                
        # Step 5: 记录轨迹（基于短期记忆的高激活）
        if mask_x4.sum() > 0:
            mask_x4_cpu = mask_x4.cpu()
            original_features = xs_with_short[mask_x4].cpu()
            used_remains = short_remains[mask_x4_cpu].cpu()

            # 记录轨迹
            self.trajectory_recorder.record(x2_features=original_features, remains=used_remains)
        
        return (x4_processed, x4_labels, x4_act_idxs, s_x4_norm,
                # x6_xs, x6_labels, x6_act_idxs, s_x6_norm,
                x5_xs, x5_labels, x5_act_idxs, s_x5_norm)
    
    def _process_reasons_batched(self, reason_xs, reason_labels, reason_act_idxs, device):
        """
        按 activate_idx 分组批量处理 Reason
        """
        next_xs_list = []
        next_labels_list = []
        
        # 按 activate_idx 分组处理
        for act_idx in range(Config.model.NUM_ACTIVATE_CLASSES):
            # 获取当前 batch
            mask = reason_act_idxs == act_idx
            if mask.sum() == 0:
                continue
            
            xs_batch = reason_xs[mask]
            labels_batch = reason_labels[mask]
            reason = self.reasons[act_idx]
            if len(xs_batch) == 0:
                continue

            # 容量检查：跳过已满的 label
            if any(len(self.output_lists.get(label, [])) >= Config.flow.MAX_OUTPUT_PER_LABEL 
                   for label in torch.unique(labels_batch).tolist()):
                continue
            
            # Reason 前向传播
            if reason.reason_type == 'forget':  # Type3
                out, next_input = reason(xs_batch)
                
                # 1. 主输出 → 保存到 output_lists（不 detach，用于计算损失）
                for i, label in enumerate(labels_batch.tolist()):
                    if len(self.output_lists.get(label, [])) < Config.flow.MAX_OUTPUT_PER_LABEL:
                        output = out[i] if out.dim() == 3 else out
                        self.output_lists.setdefault(label, []).append(output)
                
                # 2. 下一轮输入 → 加入 next_xs_list
                for i, n_input in enumerate(next_input.unbind(0)):
                    next_xs_list.append(n_input)
                    next_labels_list.append(labels_batch[i].item())
            else:  # Type1 & Type2
                # 使用对应的 reason 处理，结果作为下一轮输入
                out = reason(xs_batch)
                for i, o in enumerate(out.unbind(0)):
                    next_xs_list.append(o)
                    next_labels_list.append(labels_batch[i].item())
            
            # 更新激活次数统计
            if act_idx >= 0:
                self.activate_usage_count[act_idx] += 1
        
        # 返回结果
        if len(next_xs_list) == 0:
            return torch.empty(0, self.m, self.n), torch.empty(0)
        
        next_xs = torch.stack(next_xs_list)
        next_labels = torch.tensor(next_labels_list, dtype=torch.long, device=device)
        
        return next_xs, next_labels
