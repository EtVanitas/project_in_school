"""Alice 核心组件"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class Activate(nn.Module):
    """激活矩阵（参数存储）"""
    def __init__(self, m: int, n: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(1, m))
        nn.init.kaiming_uniform_(self.A)
        
        self.B = nn.Parameter(torch.randn(n, 1))
        nn.init.kaiming_uniform_(self.B)


class Reason(nn.Module):
    """统一的 Reason（simple/transform/forget）"""
    
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
        """初始化 Type3 的参数（三路径 + 遗忘门（已丢弃））"""
        self._init_type2_params()
    
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
        # 三条路径
        x1 = F.silu(self.A1 @ x)
        x1 = F.silu(x1 @ self.B1)
        
        x2 = F.silu(x @ self.B2)
        x2 = F.silu(self.A2 @ x2)
        
        x3 = F.silu(self.A3 @ x)
        x3 = F.silu(x3 @ self.B3)
        
        # 融合
        x_sum = x1 + x2
        x_final = x3 * x_sum
        
        # 恢复维度（带残差连接）
        c_out = self.C @ x_final
        silu_c = F.silu(c_out)
        cd_out = silu_c @ self.D
        out = x + F.silu(cd_out)
        
        return out
    
    def _forward_type3(self, x: torch.Tensor) -> torch.Tensor:
        """Type3: 三路径融合，返回与 type2 相同的输出"""
        out = self._forward_type2(x)
        return out


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


class ActivateTester(nn.Module):
    """单个样本的激活测试器"""
    
    def __init__(self, activates: nn.ModuleList):
        super().__init__()
        self.activates = activates
        self.scale = nn.Parameter(torch.ones(1))
    
    def compute_activation(self, x, act_idx):
        """计算激活值（带 sigmoid）"""
        activate = self.activates[act_idx]
        A = activate.A
        B = activate.B
        score = (A @ x @ B).squeeze()
        return torch.sigmoid(score)
