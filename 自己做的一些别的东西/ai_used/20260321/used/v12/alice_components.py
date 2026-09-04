"""
Alice 核心组件

- Activate: 激活矩阵参数存储
- Reason: 三种类型的推理模块（simple/transform/forget）
- BatchActivateTester: 批量激活测试（线性+RMSNorm+ 门控）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from config import Config


class Activate(nn.Module):
    """激活矩阵（仅用于参数存储）"""
    def __init__(self, m: int, n: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(1, m))
        nn.init.kaiming_uniform_(self.A)
        
        self.B = nn.Parameter(torch.randn(n, 1))
        nn.init.kaiming_uniform_(self.B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """占位符，实际不会调用"""
        raise NotImplementedError("Use BatchActivateTester instead")


class Reason(nn.Module):
    """
    统一的 Reason
    
    三种类型：
    - Type1 (simple): 直接返回
    - Type2 (transform): 三路径融合（复杂变换）
    - Type3 (forget): 三路径 + 遗忘门（已删除）
    """
    
    def __init__(self, m: int, n: int, p: int, q: int, reason_type: str = 'simple'):
        super().__init__()
        self.m, self.n, self.p, self.q = m, n, p, q
        self.reason_type = reason_type
        
        if reason_type == 'transform':
            self._init_type2_params()
        elif reason_type == 'forget':
            self._init_type3_params()
    
    def _init_type2_params(self):
        """初始化 Type2 的参数（三路径融合）"""
        # 路径 1
        self.A1 = nn.Parameter(torch.randn(self.p, self.m))
        self.B1 = nn.Parameter(torch.randn(self.n, self.q))
        
        # 路径 2
        self.A2 = nn.Parameter(torch.randn(self.p, self.m))
        self.B2 = nn.Parameter(torch.randn(self.n, self.q))
        
        # 路径 3
        self.A3 = nn.Parameter(torch.randn(self.p, self.m))
        self.B3 = nn.Parameter(torch.randn(self.n, self.q))
        
        # 恢复维度
        self.C = nn.Parameter(torch.randn(self.m, self.p))
        self.D = nn.Parameter(torch.randn(self.q, self.n))
        
        # 初始化
        for param in [self.A1, self.B1, self.A2, self.B2, self.A3, self.B3, self.C, self.D]:
            nn.init.kaiming_uniform_(param)
    
    def _init_type3_params(self):
        """初始化 Type3 的参数（三路径 + 遗忘门）"""
        self._init_type2_params()
        
        # 遗忘门 [n×n]
        self.forget = nn.Parameter(torch.randn(self.n, self.n))
        nn.init.kaiming_uniform_(self.forget)
    
    def forward(self, x: torch.Tensor):
        """前向传播"""
        if self.reason_type == 'simple':
            return x
        elif self.reason_type == 'transform':
            return self._forward_type2(x)
        elif self.reason_type == 'forget':
            return self._forward_type3(x)
        else:
            raise ValueError(f"Unknown reason_type: {self.reason_type}")
    
    def _forward_type2(self, x: torch.Tensor) -> torch.Tensor:
        """Type2: 三路径融合"""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        batch_size = x.size(0)
        device = x.device
        
        # 扩展参数
        A1_exp = self.A1.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        B1_exp = self.B1.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        A2_exp = self.A2.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        B2_exp = self.B2.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        A3_exp = self.A3.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        B3_exp = self.B3.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        C_exp = self.C.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        D_exp = self.D.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        
        # 三条路径
        x1 = F.silu(torch.bmm(A1_exp, x))
        x1 = F.silu(torch.bmm(x1, B1_exp))
        
        x2 = F.silu(torch.bmm(x, B2_exp))
        x2 = F.silu(torch.bmm(A2_exp, x2))
        
        x3 = F.silu(torch.bmm(A3_exp, x))
        x3 = F.silu(torch.bmm(x3, B3_exp))
        
        # 融合
        x_sum = x1 + x2
        x_final = x3 * x_sum
        
        # 恢复维度（带残差连接）
        c_out = torch.bmm(C_exp, x_final)
        silu_c = F.silu(c_out)
        cd_out = torch.bmm(silu_c, D_exp)
        out = x + F.silu(cd_out)
        
        return out.squeeze(0) if squeeze else out
    
    def _forward_type3(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Type3: 三路径融合（移除遗忘门）"""
        out = self._forward_type2(x)
        
        # 返回两份相同的输出
        # - 一份用于保存到 output_lists
        # - 一份用于作为下一轮输入
        return out, out


def create_activates_and_reasons():
    """创建 Activate 和 Reason 模块"""
    m, n, p, q = Config.model.M, Config.model.N, Config.model.P, Config.model.Q
    num_activate = Config.model.NUM_ACTIVATE_CLASSES
    
    # 创建 num_activate 个 Activate
    activates = nn.ModuleList([
        Activate(m, n)
        for _ in range(num_activate)
    ])
    
    # 创建 num_activate 个 Reason
    reasons = nn.ModuleList()
    for i in range(num_activate):
        if i in Config.model.REASON_TYPE1_INDICES:
            reasons.append(Reason(m, n, p, q, reason_type='simple'))
        elif i in Config.model.REASON_TYPE3_INDICES:
            reasons.append(Reason(m, n, p, q, reason_type='forget'))
        else:
            reasons.append(Reason(m, n, p, q, reason_type='transform'))
    
    return activates, reasons


class BatchActivateTester(nn.Module):
    """
    批量激活测试器 - 门控机制版本
    """
    
    def __init__(self, activates: nn.ModuleList):
        super().__init__()
        self._prestack_params(activates)
        self.scale = nn.Parameter(torch.ones(1))
    
    def _prestack_params(self, activates: nn.ModuleList):
        """预堆叠 Activate 参数用于批处理"""
        A_stack = torch.stack([act.A for act in activates])
        B_stack = torch.stack([act.B for act in activates])
        self.register_buffer('A_stack', A_stack)
        self.register_buffer('B_stack', B_stack)
    
    def forward(self, xs, labels, act_indices, r_high, r_low, return_mask=False):
        """前向传播"""
        device = xs.device
        
        # 确保阈值在正确设备上（统一处理，支持 tensor 和标量）
        r_high = r_high.to(device) if isinstance(r_high, torch.Tensor) else torch.tensor(r_high, device=device)
        r_low = r_low.to(device) if isinstance(r_low, torch.Tensor) else torch.tensor(r_low, device=device)
        
        # Step 1: 计算原始激活值
        s_raw = self._compute_raw_activation(xs, act_indices)
        
        # Step 2: 按 label 分组后 RMSNorm 归一化
        s_norm = self._group_rmsnorm(s_raw, labels)
        
        # Step 3: 双阈值判断（简化版，r_high 可能是标量或向量）
        r_high_expanded = r_high[act_indices] if r_high.dim() > 0 else r_high
        
        mask_high = s_norm >= r_high_expanded
        mask_low = (s_norm >= r_low) & (s_norm < r_high_expanded)
        
        # Step 4: 分离高/低激活值
        high_xs = xs[mask_high]
        high_labels = labels[mask_high]
        high_idxs = act_indices[mask_high]
        high_scores = s_norm[mask_high]
        
        low_xs = xs[mask_low]
        low_labels = labels[mask_low]
        low_idxs = act_indices[mask_low]
        low_scores = s_norm[mask_low]
        
        # Step 5: 对高激活值应用门控处理
        if len(high_xs) > 0:
            high_xs_processed = self._apply_gate_mechanism(high_xs, high_idxs)
        else:
            high_xs_processed = high_xs
        
        # Step 6: 返回结果
        if return_mask:
            return (high_xs_processed, high_labels, high_idxs, high_scores,
                    low_xs, low_labels, low_idxs, low_scores, mask_high)
        else:
            return (high_xs_processed, high_labels, high_idxs, high_scores,
                    low_xs, low_labels, low_idxs, low_scores)
    
    def _compute_raw_activation(self, xs, act_indices):
        """计算原始激活值（线性）"""
        A_test = self.A_stack[act_indices]  # (N, 1, m)
        B_test = self.B_stack[act_indices]  # (N, n, 1)
        
        # 线性计算
        temp = torch.bmm(xs, B_test)        # (N, m, 1)
        s_raw = torch.bmm(A_test, temp)     # (N, 1, 1)
        s_raw = s_raw.squeeze(-1).squeeze(-1)  # (N,)
        
        return s_raw
    
    def _group_rmsnorm(self, values, labels):
        """按 label 分组 RMSNorm（支持不连续 label）"""
        # 获取唯一 label 并创建映射
        unique_labels = torch.unique(labels)
        num_groups = len(unique_labels)
        
        # 创建 label 映射：将不连续的 label 映射到连续的索引
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        mapped_labels = torch.tensor([label_to_idx[label.item()] for label in labels], 
                                     dtype=torch.long, device=labels.device)
        
        # 计算每组 RMS
        squared = values ** 2
        sum_squared = torch.zeros(num_groups, device=values.device)
        sum_squared.scatter_add_(0, mapped_labels, squared)
        
        count = torch.bincount(mapped_labels, minlength=num_groups)
        rms_per_group = torch.sqrt(sum_squared / count.clamp(min=1) + 1e-6)
        
        # 应用归一化
        rms_for_each_value = rms_per_group[mapped_labels]
        normalized = values / rms_for_each_value
        
        # 应用可学习缩放
        values_norm = self.scale * normalized
        
        return values_norm
    
    def _apply_gate_mechanism(self, xs, act_indices):
        """门控机制：G = (x@B)@(A@x)，RMSNorm 后 Hadamard 积"""
        K = len(xs)
        
        # 获取对应参数
        A_test = self.A_stack[act_indices]  # (K, 1, m)
        B_test = self.B_stack[act_indices]  # (K, n, 1)
        
        # Step 1: 计算门控矩阵 G = (x @ B) @ (A @ x)
        col = torch.bmm(xs, B_test)         # (K, m, 1)
        row = torch.bmm(A_test, xs)         # (K, 1, n)
        G = torch.bmm(col, row)             # (K, m, n) 外积
        
        # Step 2: 对 G 做全元素 RMSNorm（每个 x 独立）
        G_flat = G.view(K, -1)              # (K, m*n)
        rms = torch.sqrt(torch.mean(G_flat ** 2, dim=1, keepdim=True) + 1e-6)  # (K, 1)
        G_normalized = G / rms.unsqueeze(-1).unsqueeze(-1)  # (K, m, n)
        
        # Step 3: 应用门控（Hadamard 积）
        xs_gated = G_normalized * xs
        
        return xs_gated
