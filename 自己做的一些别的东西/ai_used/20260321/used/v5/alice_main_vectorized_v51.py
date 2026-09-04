"""
Alice 模型主体逻辑 - 向量化版本 v5.1（计算图完整版）
关键修复：
1. 保留完整计算图用于反向传播
2. 实现 Reason 处理和输出
3. Forget 矩阵正确剪枝
4. Remain 去梯度化但 x 保留梯度
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from collections import defaultdict
from config import Config
from alice_enhanced_v4 import Activate, Manager, ReasonType1, ReasonType2, ReasonType3
from remain_manager import RemainManager
from trajectory_recorder import BatchTrajectoryRecorder
from batched_state import BatchedState


class MainModelVectorized(nn.Module):
    """向量化版本的大模型主体（v5.1 - 计算图完整版）"""
    
    def __init__(self):
        super().__init__()
        
        # ========== 初始化参数 ==========
        m = Config.model.M
        n = Config.model.N
        p = Config.model.P
        q = Config.model.Q
        
        self.m, self.n, self.p, self.q = m, n, p, q
        self.max_iterations = Config.flow.MAX_ITERATIONS
        self.decay_rate_a = Config.flow.DECAY_RATE_A
        self.max_xs = Config.flow.MAX_ACTIVATED_X + Config.flow.MAX_UNACTIVATED_X
        
        # ========== 创建 72 个 Activate ==========
        self.activates = nn.ModuleList([
            Activate(m, n, Config.model.ACTIVATE_THRESHOLDS[i])
            for i in range(Config.model.NUM_ACTIVATE_CLASSES)
        ])
        
        # ========== 创建 8 个 Manager ==========
        self.managers = nn.ModuleList([
            Manager(
                m, n, 
                Config.model.MANAGER_THRESHOLDS[i],
                managed_reason_indices=Config.model.MANAGER_RANGES[i][1:]
            )
            for i in range(Config.model.NUM_MANAGER_CLASSES)
        ])
        
        # ========== 创建 72 个 Reason（三种类型） ==========
        self.reasons = nn.ModuleList()
        for i in range(Config.model.NUM_ACTIVATE_CLASSES):
            if i in Config.model.REASON_TYPE1_INDICES:
                self.reasons.append(ReasonType1())
            elif i in Config.model.REASON_TYPE3_INDICES:
                self.reasons.append(ReasonType3(m, n, p, q))
            else:
                self.reasons.append(ReasonType2(m, n, p, q))
        
        # ========== 组件初始化 ==========
        self.remain_manager = RemainManager(num_classes=72, m=1024, n=1024)
        self.trajectory_recorder = BatchTrajectoryRecorder(max_trajectories=100)
        
        # ========== 状态管理 ==========
        self.state = BatchedState(max_xs=self.max_xs, m=m, n=n)
        
        # ========== 统计信息 ==========
        self.activate_usage_count = [0] * Config.model.NUM_ACTIVATE_CLASSES
        
        # ========== 预堆叠参数（性能优化）==========
        self._prestack_params()
    
    def _prestack_params(self):
        """预堆叠 Manager 和 Activate 参数用于批处理"""
        # Manager 参数
        self.manager_A_stack = torch.stack([mgr.A for mgr in self.managers]).cpu()  # (8, 1, m)
        self.manager_B_stack = torch.stack([mgr.B for mgr in self.managers]).cpu()  # (8, n, 1)
        self.manager_r_stack = torch.tensor([mgr.r for mgr in self.managers]).cpu()  # (8,)
        
        # Activate 参数
        self.activate_A_stack = torch.stack([act.A for act in self.activates]).cpu()  # (72, 1, m)
        self.activate_B_stack = torch.stack([act.B for act in self.activates]).cpu()  # (72, n, 1)
    
    def _test_managers_vectorized(self):
        """完全向量化的 Manager 预筛选（保留计算图）"""
        K = self.state.count_valid()
        if K == 0:
            return
        
        xs = self.state.xs_data[:K]  # (K, m, n) - 保留计算图
        device = xs.device
        
        # 参数送 GPU（不 detach，保留梯度）
        A = self.manager_A_stack.to(device)  # (8, 1, m)
        B = self.manager_B_stack.to(device)  # (8, n, 1)
        r = self.manager_r_stack.to(device)  # (8,)
        
        # 扩展维度用于批处理
        xs_exp = xs.unsqueeze(0).expand(8, -1, -1, -1).reshape(-1, self.m, self.n)
        A_rep = A.unsqueeze(1).expand(-1, K, -1, -1).reshape(-1, 1, self.m)
        B_rep = B.unsqueeze(1).expand(-1, K, -1, -1).reshape(-1, self.n, 1)
        
        # 批量矩阵乘法（保留计算图）
        temp = torch.bmm(A_rep, xs_exp)  # (8K, 1, n)
        s_all = torch.bmm(temp, B_rep)   # (8K, 1, 1)
        s_all = s_all.reshape(8, K)      # (8, K)
        s_all = torch.sigmoid(s_all)
        
        # 阈值判断
        manager_mask = (s_all >= r.unsqueeze(1)).float()  # (8, K)
        
        # 应用 allowed_managers 权限
        allowed = self.state.allowed_managers[:K].T.float()  # (8, K)
        manager_mask = manager_mask * allowed
        
        # 检查是否激活任何 Manager
        any_activated = manager_mask.any(dim=0)  # (K,)
        
        # 标记 x2: 激活不了任何 Manager
        deleted_indices = torch.arange(K, device=device)[~any_activated.bool()]
        if len(deleted_indices) > 0:
            self.state.mark_as_deleted(deleted_indices)
        
        # 保存结果供下一步使用
        self.state.manager_mask[:K] = manager_mask.T
    
    def _activate_and_reason_vectorized(self, epoch: int = None):
        """
        向量化版本的 Activate + Reason 处理（保留完整计算图）
        
        关键：
        1. 使用高级索引保留计算图
        2. 批量计算 modified_x
        3. 分组进行 Reason 处理
        4. Forget 矩阵正确剪枝
        """
        K = self.state.count_valid()
        if K == 0:
            return [], []
        
        xs = self.state.xs_data[:K]  # (K, m, n) - 保留计算图
        device = xs.device
        
        # ========== Step 1: 构建测试掩码 ==========
        test_mask = torch.zeros(K, 72, dtype=torch.bool, device=device)
        
        # 情况 A: x0/x1 - 测试所有允许的 Activate
        initial_mask = (self.state.prev_act_idx[:K] == -1) & (self.state.status[:K] == 0)
        initial_indices = torch.arange(K, device=device)[initial_mask]
        
        if len(initial_indices) > 0:
            for k in initial_indices:
                for mgr_idx in range(8):
                    if self.state.manager_mask[k, mgr_idx] > 0:
                        act_indices = Config.model.MANAGER_RANGES[mgr_idx][1:]
                        test_mask[k, act_indices] = True
        
        # 情况 B: x7 - 只测试上次的 activate
        retry_mask = self.state.prev_act_idx[:K] != -1
        retry_indices = torch.arange(K, device=device)[retry_mask]
        
        if len(retry_indices) > 0:
            for k in retry_indices:
                act_idx = self.state.prev_act_idx[k]
                test_mask[k, act_idx] = True
        
        # ========== Step 2: 提取需要测试的 (x_idx, act_idx) 对 ==========
        x_indices, act_indices = torch.where(test_mask)  # (N_pairs,)
        N_pairs = len(x_indices)
        
        if N_pairs == 0:
            return [], []
        
        # 提取对应的 x（关键：使用高级索引保留计算图）
        xs_to_test = xs[x_indices]  # (N_pairs, m, n) - 保留计算图！
        
        # ========== Step 3: 批量计算 s 值 ==========
        A_test = self.activate_A_stack[act_indices].to(device)  # (N_pairs, 1, m)
        B_test = self.activate_B_stack[act_indices].to(device)  # (N_pairs, n, 1)
        
        temp = torch.bmm(A_test, xs_to_test)  # (N_pairs, 1, n)
        s_all = torch.bmm(temp, B_test)       # (N_pairs, 1, 1)
        s_all = torch.sigmoid(s_all).squeeze()  # (N_pairs,)
        
        # ========== Step 4: 三重阈值筛选 ==========
        r_high = Config.model.ACTIVATE_THRESHOLD_HIGH
        r_low = Config.model.ACTIVATE_THRESHOLD_LOW
        
        direct_mask = s_all >= r_high                      # x3
        assist_mask = (s_all >= r_low) & (s_all < r_high)  # x4/x6/x7
        
        # ========== Step 5: Remain 辅助（向量化）==========
        assist_x_idx = x_indices[assist_mask]
        assist_act_idx = act_indices[assist_mask]
        N_assist = len(assist_x_idx)
        
        s_final = s_all.clone()  # 最终的 s 值
        
        if N_assist > 0:
            # 批量获取 remain（detach 但不影响 x 的梯度）
            remains_dict = {}
            for act_idx in assist_act_idx.unique():
                remain_cpu = self.remain_manager.get_remain_for_activation(act_idx)
                if remain_cpu is not None:
                    # 关键：remain 要 detach，但 x+remain 后仍然有梯度
                    remains_dict[act_idx.item()] = remain_cpu.detach().to(device)
            
            # 为每个 assist 的 x 添加对应的 remain
            x_with_remain = xs_to_test[assist_mask].clone()
            for i, act_idx in enumerate(assist_act_idx):
                act_idx_item = act_idx.item()
                if act_idx_item in remains_dict:
                    x_with_remain[i] = x_with_remain[i] + remains_dict[act_idx_item]
            
            # 重新计算 s 值
            A_assist = self.activate_A_stack[assist_act_idx].to(device)
            B_assist = self.activate_B_stack[assist_act_idx].to(device)
            
            temp = torch.bmm(A_assist, x_with_remain)
            s_assist = torch.bmm(temp, B_assist)
            s_assist = torch.sigmoid(s_assist).squeeze()
            
            # 判断是否成功
            success_mask = s_assist >= r_high
            
            # 更新成功的 x6
            assist_success_mask = assist_mask.clone()
            assist_success_mask[assist_mask] = success_mask
            
            # 更新 s_final
            s_final[assist_success_mask] = s_assist[success_mask]
            
            # 更新 xs_to_test（x6 使用 x_with_remain）
            xs_to_test[assist_mask] = x_with_remain
            xs_to_test[assist_fail_mask] = xs_to_test[assist_mask][~success_mask]
        
        # ========== Step 6: 分类收集 ==========
        all_success_mask = direct_mask | (assist_mask & (s_final >= r_high))
        
        success_x_idx = x_indices[all_success_mask]
        success_act_idx = act_indices[all_success_mask]
        success_s = s_final[all_success_mask]
        
        fail_x_idx = x_indices[~all_success_mask]
        fail_act_idx = act_indices[~all_success_mask]
        fail_s = s_final[~all_success_mask]
        
        # ========== Step 7: 批量计算 modified_x（保留计算图）==========
        # 对成功的 x 计算 Activate 的输出
        if len(success_x_idx) > 0:
            xs_success = xs_to_test[all_success_mask]  # (N_success, m, n)
            
            # 调用 Activate 的 forward（会计算 modified_x）
            # 注意：这里需要逐个 activate 处理，因为每个 x 对应不同的 activate
            modified_xs = []
            for i in range(len(success_x_idx)):
                act_idx = success_act_idx[i].item()
                x_single = xs_success[i:i+1]  # (1, m, n)
                
                # 调用 Activate（保留计算图）
                activate = self.activates[act_idx]
                modified_x, _ = activate(x_single, epoch=epoch)
                modified_xs.append(modified_x.squeeze(0))
            
            modified_xs_batch = torch.stack(modified_xs)  # (N_success, m, n)
        else:
            modified_xs_batch = torch.zeros(0, self.m, self.n, device=device)
        
        # ========== Step 8: 按 activate_idx 分组进行 Reason 处理 ==========
        output_list = []
        next_xs = []
        
        # 按 activate_idx 分组
        grouped = defaultdict(list)
        for i in range(len(success_x_idx)):
            act_idx = success_act_idx[i].item()
            grouped[act_idx].append((modified_xs_batch[i], success_x_idx[i], success_s[i].item()))
        
        # 对每组进行 Reason 处理
        for act_idx, items in grouped.items():
            if self.activate_usage_count[act_idx] >= Config.flow.MAX_ACTIVATE_USAGE:
                continue
            
            # 打包这一组的 modified_x
            group_xs = torch.stack([item[0] for item in items])
            
            # Reason 处理
            reason = self.reasons[act_idx]
            
            if isinstance(reason, ReasonType3):
                out, forgotten_out = reason(group_xs)
            else:
                out = reason(group_xs)
                forgotten_out = out
            
            # 特殊输出
            if act_idx in Config.special_output.INDICES:
                output_list.append(out)
            
            # Forget 剪枝（关键：detach 但 requires_grad_(True)）
            if isinstance(reason, ReasonType3):
                forgotten_out = forgotten_out.detach().requires_grad_(True)
            
            # 添加到下一轮
            for f_out in forgotten_out.unbind(0):
                next_xs.append(f_out)
            
            # 收集 remain 更新数据
            for _, x_idx, s_val in items:
                original_x_idx = success_x_idx[torch.where(success_x_idx == x_idx)[0][0]]
                activated_data_for_remain.append((
                    act_idx,
                    modified_xs_batch[torch.where(success_x_idx == original_x_idx)[0][0]].detach().cpu(),
                    s_val
                ))
            
            self.activate_usage_count[act_idx] += 1
        
        # ========== Step 9: 更新状态 ==========
        # 标记失败的 x 为 x7
        for i in range(len(fail_x_idx)):
            x_idx = fail_x_idx[i].item()
            act_idx = fail_act_idx[i].item()
            s_val = fail_s[i].item()
            
            self.state.prev_act_idx[x_idx] = act_idx
            self.state.prev_s_value[x_idx] = s_val
        
        return output_list, next_xs
    
    def forward(self, initial_xs: torch.Tensor, epoch: int = None):
        """
        向量化版本的前向传播（保留完整计算图）
        
        Args:
            initial_xs: (batch_size, m, n)
            epoch: 当前训练 epoch（用于 STE）
        
        Returns:
            output_list: 特殊输出列表
            stats: 统计信息
        """
        batch_size = initial_xs.size(0)
        device = initial_xs.device
        
        # ========== 初始化 ==========
        self.state.reset()
        self.state.initialize_inputs(initial_xs)
        
        # 初始化 remain
        if len(self.remain_manager.remain_list) == 0:
            self.remain_manager.initialize_from_long_term({})
        
        output_list = []
        activated_data_for_remain = []
        current_xs = []
        
        # ========== 主循环 ==========
        for iteration in range(self.max_iterations):
            
            # ========== Step 1: 超时剪枝 ==========
            timeout_mask = self.state.get_timeout_mask(Config.flow.MAX_AGE_WITHOUT_ACTIVATION)
            timeout_indices = torch.nonzero(timeout_mask, as_tuple=False).squeeze(-1)
            if len(timeout_indices) > 0:
                self.state.mark_as_deleted(timeout_indices)
            
            # ========== Step 2: Manager 预筛选（向量化）==========
            self._test_managers_vectorized()
            
            # ========== Step 3: Activate + Reason 处理（向量化）==========
            outputs, next_xs = self._activate_and_reason_vectorized(epoch=epoch)
            
            output_list.extend(outputs)
            current_xs = next_xs
            
            # ========== Step 4: 年龄递增 ==========
            self.state.increment_ages()
            
            # ========== Step 5: 延迟更新 Remain ==========
            if len(activated_data_for_remain) > 0:
                self.remain_manager.update_delayed(activated_data_for_remain)
                activated_data_for_remain.clear()
            
            # ========== Step 6: 衰减 Remain ==========
            self.remain_manager.decay(iteration)
            
            # ========== Step 7: 轨迹记录 ==========
            with torch.no_grad():
                for act_idx in range(72):
                    remain_val = self.remain_manager.get_remain_value(act_idx)
                    self.trajectory_recorder.record(
                        iteration=iteration,
                        activate_idx=act_idx,
                        remain_value=remain_val.item()
                    )
            
            # ========== 提前终止条件 ==========
            if self.state.count_valid() == 0 and len(current_xs) == 0:
                break
            
            if len(output_list) >= 50:
                break
            
            if len(current_xs) == 0:
                break
        
        # ========== 返回统计信息 ==========
        stats = {
            'iterations': iteration + 1,
            'activate_usage': self.activate_usage_count.copy(),
            'final_x_count': self.state.count_valid() + len(current_xs),
        }
        
        return output_list, stats
