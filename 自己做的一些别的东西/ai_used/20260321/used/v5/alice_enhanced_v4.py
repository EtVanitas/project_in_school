"""
Alice 模型核心组件 - 增强版 v4.0
完整重构版：支持 Manager 预筛选、双重阈值、STE 训练、批处理集成

更新内容:
1. Manager 预筛选 + 权限控制
2. 双重阈值激活检测（0.7/0.5）
3. x7 优化重试策略
4. STE (Straight-Through Estimator) 训练
5. Softmax+TopK 松弛化
6. 批处理集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from config import Config


class Activate(nn.Module):
    """激活类模块 - 支持 STE 训练和 TopK 松弛化"""
    
    def __init__(self, m: int, n: int, r: float):
        super().__init__()
        self.m, self.n, self.r = m, n, r
        # 可学习参数 A(1×m) 和 B(n×1)
        self.A = nn.Parameter(torch.randn(1, m))
        self.B = nn.Parameter(torch.randn(n, 1))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        
        # STE 参数
        self.ste_k = Config.model.STE_K
        self.epoch_threshold = Config.model.EPOCH_THRESHOLD
    
    def forward(self, x: torch.Tensor, epoch: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入矩阵 (batch×m×n) 或 (m×n)
            epoch: 当前训练 epoch（用于切换策略）
        
        Returns:
            output: 激活后的输出 s * (X ⊙ W)，未激活返回 0
            s: 激活值
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        batch_size = x.size(0)
        # 确保参数在正确的设备上
        A_exp = self.A.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        B_exp = self.B.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        
        # 计算原始激活值
        s_raw = torch.sigmoid(torch.bmm(torch.bmm(A_exp, x), B_exp))
        
        # ========== 决定使用哪种策略 ==========
        if epoch is not None and epoch < self.epoch_threshold:
            # 早期：Softmax + TopK 探索
            s = self._softmax_topk_relaxation(s_raw, k=Config.model.ACTIVATE_TOP_K)
        else:
            # 后期：STE 或纯 Sigmoid
            if Config.model.USE_STE:
                # STE: 前向 hard，反向 soft
                s = self._ste_forward(s_raw)
            else:
                # 默认：纯 Sigmoid
                s = s_raw
        
        # 计算门控矩阵 W（特征变换 - 全部使用 SiLU）
        U = F.silu(torch.bmm(x, B_exp))
        V = F.silu(torch.bmm(A_exp, x))
        W = torch.bmm(U, V)
        
        # 输出 s * (X ⊙ W)
        output = s * (x * W)
        
        return (output.squeeze(0) if squeeze else output,
                s.squeeze(0) if squeeze else s)
    
    def _softmax_topk_relaxation(self, s_raw: torch.Tensor, k: int) -> torch.Tensor:
        """
        Softmax + TopK 松弛化（早期探索阶段）
        
        Args:
            s_raw: 原始 s 值
            k: top-k 数量
        
        Returns:
            relaxed_s: 松弛化的 s 值（可导）
        """
        # Softmax 松弛化
        temperature = 2.0
        weights = torch.softmax(s_raw / temperature, dim=-1)
        
        # TopK 选择
        topk_values, topk_indices = torch.topk(weights, k=k, dim=-1)
        
        # 构建松弛化掩码（可导）
        relaxed_s = torch.zeros_like(s_raw)
        relaxed_s.scatter_(-1, topk_indices, topk_values)
        
        return relaxed_s
    
    def _ste_forward(self, s_raw: torch.Tensor) -> torch.Tensor:
        """
        STE (Straight-Through Estimator) 前向传播
        
        前向：硬阈值
        反向：soft 近似
        
        Returns:
            s_hard: 硬阈值输出（但梯度可反向传播）
        """
        class StraightThrough(torch.autograd.Function):
            @staticmethod
            def forward(ctx, s_raw, threshold, k):
                ctx.save_for_backward(s_raw, threshold)
                return (s_raw >= threshold).float()
            
            @staticmethod
            def backward(ctx, grad_output):
                s_raw, threshold = ctx.saved_tensors
                s_soft = torch.sigmoid(k * (s_raw - threshold))
                return grad_output * s_soft, None, None
        
        s_hard = StraightThrough.apply(s_raw, self.r, self.ste_k)
        return s_hard


class Manager(nn.Module):
    """辅助管理类 - 支持 STE 训练和 TopK 松弛化"""
    
    def __init__(self, m: int, n: int, r: float, managed_reason_indices: List[int]):
        """
        Args:
            managed_reason_indices: 管理的 Reason 索引列表（不包括第一类）
        """
        super().__init__()
        self.m, self.n, self.r = m, n, r
        self.managed_reason_indices = managed_reason_indices
        
        # 可学习参数 A(1×m) 和 B(n×1)
        self.A = nn.Parameter(torch.randn(1, m))
        self.B = nn.Parameter(torch.randn(n, 1))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        
        # STE 参数
        self.ste_k = Config.model.STE_K
        self.epoch_threshold = Config.model.EPOCH_THRESHOLD
    
    def forward(self, x: torch.Tensor, epoch: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入矩阵 (batch×m×n) 或 (m×n)
            epoch: 当前训练 epoch（用于切换策略）
        
        Returns:
            activated_mask: 激活掩码 (batch_size,) 或 (8, batch_size)
            s_values: 激活值 (batch_size, 1)
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        batch_size = x.size(0)
        # 确保参数在正确的设备上
        A_exp = self.A.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        B_exp = self.B.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        
        # 计算原始激活值 s = A @ X @ B
        s = torch.bmm(torch.bmm(A_exp, x), B_exp)
        
        # ========== 决定使用哪种策略 ==========
        if epoch is not None and epoch < self.epoch_threshold:
            # 早期：Softmax + TopK 探索
            activated_mask = self._softmax_topk_relaxation(s, k=Config.model.MANAGER_TOP_K)
        else:
            # 后期：STE 或硬阈值
            if Config.model.USE_STE:
                # STE: 前向 hard，反向 soft
                activated_mask = self._ste_forward(s)
            else:
                # 默认：硬阈值
                activated_mask = (s >= self.r).float()
        
        if squeeze:
            activated_mask = activated_mask.squeeze(-1).squeeze(-1)
            s = s.squeeze(0)
        
        return activated_mask, s
    
    def _softmax_topk_relaxation(self, s: torch.Tensor, k: int) -> torch.Tensor:
        """Softmax + TopK 松弛化"""
        # 按 batch 维度做 softmax
        if s.dim() == 4:
            s_flat = s.view(s.size(0), -1)  # (batch, num_managers)
            weights = torch.softmax(s_flat / 2.0, dim=-1)
            
            # TopK
            topk_values, topk_indices = torch.topk(weights, k=k, dim=-1)
            
            # 构建掩码
            mask = torch.zeros_like(s_flat)
            mask.scatter_(-1, topk_indices, topk_values)
            mask = mask.view(*s.shape)
        else:
            weights = torch.softmax(s / 2.0, dim=-1)
            topk_values, topk_indices = torch.topk(weights, k=k, dim=-1)
            mask = torch.zeros_like(s)
            mask.scatter_(-1, topk_indices, topk_values)
        
        return mask
    
    def _ste_forward(self, s: torch.Tensor) -> torch.Tensor:
        """STE 前向传播"""
        class StraightThrough(torch.autograd.Function):
            @staticmethod
            def forward(ctx, s, threshold, k):
                ctx.save_for_backward(s, threshold)
                return (s >= threshold).float()
            
            @staticmethod
            def backward(ctx, grad_output):
                s, threshold = ctx.saved_tensors
                s_soft = torch.sigmoid(k * (s - threshold))
                return grad_output * s_soft, None, None
        
        return StraightThrough.apply(s, self.r, self.ste_k)


class ReasonType1(nn.Module):
    """第一类 Reason - Pass-through（无参数，直接输出）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """直接返回输入，不做任何处理"""
        return x


class ReasonType2(nn.Module):
    """第二类 Reason - Standard Reason（标准三路径推理）"""
    
    def __init__(self, m: int, n: int, p: int, q: int):
        super().__init__()
        self.m, self.n, self.p, self.q = m, n, p, q
        
        # 路径 1: X₁ = SiLU(A₁@X) @ B₁
        self.A1 = nn.Parameter(torch.randn(p, m))
        self.B1 = nn.Parameter(torch.randn(n, q))
        
        # 路径 2: X₂ = SiLU(A₂ @ SiLU(X@B₂))
        self.A2 = nn.Parameter(torch.randn(p, m))
        self.B2 = nn.Parameter(torch.randn(n, q))
        
        # 路径 3: X₃ = SiLU(A₃@X) @ B₃
        self.A3 = nn.Parameter(torch.randn(p, m))
        self.B3 = nn.Parameter(torch.randn(n, q))
        
        # 恢复维度：C(m×p), D(q×n)
        self.C = nn.Parameter(torch.randn(m, p))
        self.D = nn.Parameter(torch.randn(q, n))
        
        # 初始化
        for param in [self.A1, self.B1, self.A2, self.B2, self.A3, self.B3, self.C, self.D]:
            nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        三路径融合（全部使用 SiLU）
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        batch_size = x.size(0)
        
        # 扩展参数并确保在正确的设备上
        A1 = self.A1.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        B1 = self.B1.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        A2 = self.A2.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        B2 = self.B2.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        A3 = self.A3.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        B3 = self.B3.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        C = self.C.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        D = self.D.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        
        # 路径 1（全部使用 SiLU）
        x1 = F.silu(torch.bmm(A1, x))
        x1 = F.silu(torch.bmm(x1, B1))
        
        # 路径 2（全部使用 SiLU）
        x2 = F.silu(torch.bmm(x, B2))
        x2 = F.silu(torch.bmm(A2, x2))
        
        # 路径 3（全部使用 SiLU）
        x3 = F.silu(torch.bmm(A3, x))
        x3 = F.silu(torch.bmm(x3, B3))
        
        # 融合
        x_sum = x1 + x2
        x_final = x3 * x_sum
        
        # 恢复维度
        out = torch.bmm(torch.bmm(C, x_final), D)
        
        return out.squeeze(0) if squeeze else out


class ReasonType3(nn.Module):
    """第三类 Reason - With Forget Gate（带遗忘门）"""
    
    def __init__(self, m: int, n: int, p: int, q: int):
        super().__init__()
        self.m, self.n, self.p, self.q = m, n, p, q
        
        # 路径 1-3 参数（同 Type2）
        self.A1 = nn.Parameter(torch.randn(p, m))
        self.B1 = nn.Parameter(torch.randn(n, q))
        self.A2 = nn.Parameter(torch.randn(p, m))
        self.B2 = nn.Parameter(torch.randn(n, q))
        self.A3 = nn.Parameter(torch.randn(p, m))
        self.B3 = nn.Parameter(torch.randn(n, q))
        self.C = nn.Parameter(torch.randn(m, p))
        self.D = nn.Parameter(torch.randn(q, n))
        
        # 遗忘门参数 [n×n]
        self.forget = nn.Parameter(torch.randn(n, n))
        
        # 初始化
        for param in [self.A1, self.B1, self.A2, self.B2, self.A3, self.B3, self.C, self.D, self.forget]:
            nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        三路径融合 + 遗忘门
        
        Returns:
            out: 原始输出（去输出列表）
            forgotten_out: 遗忘后的输出（参与下一轮循环）
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        batch_size = x.size(0)
        
        # 扩展参数并确保在正确的设备上
        A1 = self.A1.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        B1 = self.B1.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        A2 = self.A2.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        B2 = self.B2.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        A3 = self.A3.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        B3 = self.B3.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        C = self.C.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        D = self.D.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        forget = self.forget.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        
        # 标准三路径推理（全部使用 SiLU）
        x1 = F.silu(torch.bmm(A1, x))
        x1 = F.silu(torch.bmm(x1, B1))
        
        x2 = F.silu(torch.bmm(x, B2))
        x2 = F.silu(torch.bmm(A2, x2))
        
        x3 = F.silu(torch.bmm(A3, x))
        x3 = F.silu(torch.bmm(x3, B3))
        
        x_sum = x1 + x2
        x_final = x3 * x_sum
        out = torch.bmm(torch.bmm(C, x_final), D)
        
        # 遗忘操作：sigmoid(out @ forget) ⊙ out
        forget_gate = torch.sigmoid(torch.bmm(out, forget))
        forgotten_out = forget_gate * out
        
        return (out.squeeze(0) if squeeze else out, 
                forgotten_out.squeeze(0) if squeeze else forgotten_out)
