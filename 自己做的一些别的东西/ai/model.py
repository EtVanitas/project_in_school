import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 高斯变换 + SiLU ====================
class GaussSiLU(nn.Module):
    """对输入向量的每个标量计算一组可学习高斯分布下的概率密度，再线性变换并 SiLU 激活"""
    def __init__(self, num, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(num))
        self.log_var = nn.Parameter(torch.zeros(num))
        self.w = nn.Parameter(torch.empty(num, dim))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        # x: (B, L)
        # log 空间计算高斯 pdf：exp 后不再除法，数值稳定且省一次 sqrt/除法
        inv_var = torch.exp(-self.log_var)                      # (num,)
        diff = x.unsqueeze(-1) - self.mean.unsqueeze(0)         # (B, L, num)
        log_pdf = -0.5 * diff ** 2 * inv_var \
                  - 0.5 * (self.log_var + math.log(2 * math.pi))
        pdf = torch.exp(log_pdf)

        out = pdf @ self.w
        return F.silu(out)


# ==================== 傅里叶 + Morlet 小波变换 ====================
class WaveTransform(nn.Module):
    """对输入的时间参数 t 计算傅里叶基和 Morlet 小波基的响应，再线性变换（无激活）"""
    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.num_fourier = num // 2
        self.num_morlet = num - self.num_fourier
        self.offset = nn.Parameter(torch.zeros(num))
        self.log_scale = nn.Parameter(torch.zeros(num))
        self.w = nn.Parameter(torch.empty(num, dim))
        nn.init.xavier_uniform_(self.w)

    def forward(self, t):
        # t: (B,) 批量时间参数
        scale = torch.exp(self.log_scale) + 1e-8         # (num,)
        sqrt_scale = torch.sqrt(scale)
        diff = (t.unsqueeze(-1) - self.offset.unsqueeze(0)) / scale.unsqueeze(0)  # (B, num)

        fourier = torch.cos(diff[:, :self.num_fourier]) / sqrt_scale[:self.num_fourier].unsqueeze(0)
        morlet_cos = torch.cos(diff[:, self.num_fourier:])
        morlet_win = torch.exp(-0.5 * diff[:, self.num_fourier:] ** 2)
        morlet = (morlet_cos * morlet_win) / sqrt_scale[self.num_fourier:].unsqueeze(0)

        vals = torch.cat([fourier, morlet], dim=1)        # (B, num)
        return vals @ self.w                              # (B, dim)


# ==================== 融合模块 ====================
class Fusion(nn.Module):
    """融合向量和时间参数：向量经高斯变换得到矩阵，时间经小波变换得到查询，两者相乘输出"""
    def __init__(self, gauss_num, wave_num, dim):
        super().__init__()
        self.gauss = GaussSiLU(gauss_num, dim)
        self.wave = WaveTransform(wave_num, dim)

    def forward(self, vec, t):
        gauss_out = self.gauss(vec)                         # (B, L, dim)
        wave_out = self.wave(t).unsqueeze(1)                # (B, 1, dim)
        out = torch.bmm(wave_out, gauss_out.transpose(1, 2)).squeeze(1)  # (B, L)
        return out


# ==================== RMSNorm ====================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


# ==================== 中间层块 ====================
class LayerBlock(nn.Module):
    def __init__(self, d_model, ff_dim, gauss_num, wave_num):
        super().__init__()
        self.w_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.value = nn.Parameter(torch.empty(ff_dim, d_model))
        nn.init.xavier_uniform_(self.value)
        self.fusion = Fusion(gauss_num, wave_num, gauss_num * 2)
        self.norm = RMSNorm(d_model)

    def forward(self, h, hist_weights):
        # h: (1, d_model)
        # hist_weights: (K, ff_dim) 该层历史权重矩阵，按时间从新到旧，K 可为 0
        w_cur = F.softmax(self.w_proj(h), dim=-1)          # (1, ff_dim)

        # 合并当前与历史权重（空记忆时 cat 自动退化为仅 w_cur），形状 (K+1, ff_dim)
        all_w = torch.cat([w_cur, hist_weights], dim=0)

        # 计算所有值向量
        v = all_w @ self.value                             # (K+1, d_model)

        # 时间参数从 0 开始
        t = torch.arange(all_w.shape[0], dtype=torch.float32, device=h.device)

        # 批量融合并求和
        fused = self.fusion(v, t)                          # (K+1, d_model)
        total = fused.sum(dim=0, keepdim=True)             # (1, d_model)

        out = h + self.norm(total)
        return out, w_cur.detach()


# ==================== 主体语言模型 ====================
class WaveGaussianLM(nn.Module):
    def __init__(self, vocab_size, d_model=1024, num_layers=32,
                 first_gauss=1024, first_wave=1024,
                 layer_ff_dim=4096, layer_gauss=128, layer_wave=128):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layer_ff_dim = layer_ff_dim

        self.embed = nn.Embedding(vocab_size, d_model)
        self.first_fusion = Fusion(first_gauss, first_wave, first_gauss * 2)
        self.first_norm = RMSNorm(d_model)
        self.layers = nn.ModuleList([
            LayerBlock(d_model, layer_ff_dim, layer_gauss, layer_wave)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids, memory):
        # ids: (L,) 当前窗口 token id（仅支持 1D，由 ModelWrapper 保证）
        # memory: (K, num_layers, layer_ff_dim) 历史权重记忆，K 为记忆长度
        L = ids.shape[0]
        K = memory.shape[0]
        device = ids.device

        # ---------- 聚合窗口 token ----------
        embeds = self.embed(ids)                            # (L, d_model)
        distances = torch.arange(L-1, -1, -1, dtype=torch.float32, device=device)  # 当前为0
        fused_tokens = self.first_fusion(embeds, distances) # (L, d_model)
        agg = fused_tokens.sum(dim=0, keepdim=True)         # (1, d_model)
        h = self.first_norm(agg)                            # (1, d_model)

        # ---------- 逐层处理 ----------
        # 预提取每层历史权重矩阵（时间反转为新->旧）：(num_layers, K, ff_dim)
        hist_per_layer = memory.flip(0).transpose(0, 1)

        cur_weights = []
        for li, layer in enumerate(self.layers):
            h, w_detached = layer(h, hist_per_layer[li])
            cur_weights.append(w_detached)                  # (1, layer_ff_dim)

        new_mem = torch.cat(cur_weights, dim=0).unsqueeze(0)  # (1, num_layers, layer_ff_dim)
        logits = self.head(h)                               # (1, vocab_size)
        return logits, new_mem


# ==================== 外部管理器 ====================
class ModelWrapper:
    """维护 token 窗口和记忆队列，负责数据整理，将整理好的 ids 和 memory 传给主体模型"""
    def __init__(self, model, max_input_len=1024, max_mem_len=32):
        self.model = model
        self.max_input_len = max_input_len
        self.max_mem_len = max_mem_len
        self.device = next(model.parameters()).device
        self.window = []
        self.memory = torch.empty(0, model.num_layers, model.layer_ff_dim, device=self.device)

    def reset(self):
        """重置外部状态（每个新序列从头开始）"""
        self.window = []
        self.memory = torch.empty(0, self.model.num_layers, self.model.layer_ff_dim,
                                  device=self.device)

    def step(self, token_id):
        """接收一个新 token，返回预测下一个 token 的 logits (1, vocab_size)"""
        self.window.append(token_id)
        if len(self.window) > self.max_input_len:
            self.window.pop(0)

        # 整理输入
        ids = torch.tensor(self.window, dtype=torch.long, device=self.device)
        # 调用主体
        logits, new_mem = self.model(ids, self.memory)

        # 更新记忆队列（裁剪到最大长度）
        self.memory = torch.cat([self.memory, new_mem], dim=0)
        if self.memory.shape[0] > self.max_mem_len:
            self.memory = self.memory[-self.max_mem_len:]

        return logits