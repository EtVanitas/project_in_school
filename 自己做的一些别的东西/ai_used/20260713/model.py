# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from config import ModelArgs


# -------------------- 基础归一化 --------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# -------------------- RoPE 相关 --------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    shape = [1] * xq_.ndim
    shape[1] = xq_.shape[1]      # seq_len
    shape[-1] = xq_.shape[-1]    # head_dim // 2
    freqs_cis = freqs_cis.view(*shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# -------------------- 双向 Transformer 层（无掩码） --------------------
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        # 使用 PyTorch 的 Flash Attention（通过 SDPA）
        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,          # 不使用 mask
            is_causal=False          # 双向注意力
        )
        # out 的形状为 [bsz, n_heads, seqlen, head_dim]
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.dim
        hidden_dim = args.ffn_hidden_dim * 2 if args.ffn_hidden_dim is not None else 4 * dim
        self.use_swiglu = args.use_swiglu
        if self.use_swiglu:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        else:
            self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
            self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        if self.use_swiglu:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        else:
            return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = Attention(args)
        self.ff = FeedForward(args)
        self.norm1 = RMSNorm(args.dim, eps=args.norm_eps)
        self.norm2 = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        h = x + self.attn(self.norm1(x), freqs_cis)
        return h + self.ff(self.norm2(h))


# -------------------- 自定义中间层（新架构） --------------------
class CustomBlock(nn.Module):
    """
    输入:  [batch, 1024, 1]
    输出:  [batch, 1024, 1]
    内部流程：
      1. 对 1024 维度做 softmax，使其变为概率分布
      2. 广播乘法: 用 [1024,1024] 可学习矩阵将 1024 个标量映射为 1024 个 1024 维向量。
      3. SwiGLU: up/gate 投影到 3072，down 投影到 2048。
      4. 压缩 1024 维度 → [batch, 1, 2048]
      5. 线性投影 [2048, 1024] → [batch, 1, 1024]
      6. 交换后两维 → [batch, 1024, 1]
      加入残差连接。
    """
    def __init__(self, dim=1024, swiglu_expand=3072, swiglu_out=2048):
        super().__init__()
        # 第一步：类似词表嵌入的广播矩阵
        self.scale_embed = nn.Parameter(torch.randn(dim, dim))
        # SwiGLU 组件
        self.gate = nn.Linear(dim, swiglu_expand, bias=False)
        self.up = nn.Linear(dim, swiglu_expand, bias=False)
        self.down = nn.Linear(swiglu_expand, swiglu_out, bias=False)
        # 压缩后的投影
        self.fc_out = nn.Linear(swiglu_out, dim, bias=False)

    def forward(self, x):
        residual = x                            # [bs, 1024, 1]

        # 1. 对 1024 维度做 softmax，使其类似概率分布
        x_sm = F.softmax(x, dim=1)              # [bs, 1024, 1]

        # 2. 广播乘法 → [bs, 1024, 1024]
        out = x_sm * self.scale_embed           # 广播: [bs,1024,1] * [1024,1024] → [bs,1024,1024]

        # 3. SwiGLU
        gate = self.gate(out)                   # [bs, 1024, 3072]
        up = F.silu(self.up(out))               # [bs, 1024, 3072]
        out = self.down(gate * up)              # [bs, 1024, 2048]

        # 4. 压缩 1024 维度（平均池化）
        out = out.mean(dim=1, keepdim=True)     # [bs, 1, 2048]

        # 5. 投影回 1024 维
        out = self.fc_out(out)                  # [bs, 1, 1024]

        # 6. 交换后两维
        out = out.transpose(1, 2)               # [bs, 1024, 1]

        # 残差连接
        return residual + out


# -------------------- 完整混合模型 --------------------
class MixedArchTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        dim = args.dim                       # 1024
        self.max_seq_len = args.max_seq_len  # 最长上下文长度
        self.n_heads = args.n_heads
        head_dim = dim // args.n_heads

        # 词嵌入
        self.tok_embeddings = nn.Embedding(args.vocab_size, dim)

        # 前 6 层 Transformer
        self.front_blocks = nn.ModuleList([
            TransformerBlock(args) for _ in range(6)
        ])

        # 中间 12 层自定义块
        self.middle_blocks = nn.ModuleList([
            CustomBlock(dim=dim, swiglu_expand=3072, swiglu_out=2048)
            for _ in range(12)
        ])

        # 从自定义块转换回 Transformer 的“连接矩阵”
        # 将 [bs, 1024, 1] 映射为 [bs, 1024, 1024]（与 CustomBlock 第一步形式一致）
        self.to_transformer = nn.Parameter(torch.randn(dim, dim))

        # 后 6 层 Transformer
        self.back_blocks = nn.ModuleList([
            TransformerBlock(args) for _ in range(6)
        ])

        # 最终输出
        self.final_norm = RMSNorm(dim, eps=args.norm_eps)
        self.lm_head = nn.Linear(dim, args.vocab_size, bias=False)
        # 共享权重
        self.lm_head.weight = self.tok_embeddings.weight

        # 预计算 RoPE（前后 Transformer 层需要不同长度）
        # 前块使用 max_seq_len，后块需要处理 dim 维度的位置（中间块输出为 [bs, dim, dim]）
        freqs_cis_front = precompute_freqs_cis(head_dim, self.max_seq_len, args.rope_theta)
        freqs_cis_back  = precompute_freqs_cis(head_dim, dim, args.rope_theta)
        self.register_buffer("freqs_cis_front", freqs_cis_front)
        self.register_buffer("freqs_cis_back", freqs_cis_back)

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]   seq_len 必须 <= max_seq_len
        返回 logits: [batch_size, 1024, vocab_size]
        """
        bsz, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"输入序列长度必须小于 {self.max_seq_len}"

        # 1. 词嵌入 + 前 6 层 Transformer（启用检查点）
        h = self.tok_embeddings(input_ids)              # [bs, 1024, 1024]
        freqs_cis_front = self.freqs_cis_front[:seq_len].to(h.device)
        freqs_cis_back  = self.freqs_cis_back.to(h.device)

        for blk in self.front_blocks:
            if self.args.use_gradient_checkpointing and self.training:
                h = checkpoint.checkpoint(blk, h, freqs_cis_front, use_reentrant=False)
            else:
                h = blk(h, freqs_cis_front)                   # [bs, 1024, 1024]

        # 2. 压缩 seq 维度 → [bs, 1, 1024] → [bs, 1024, 1]
        h = h.mean(dim=1, keepdim=True)                 # [bs, 1, 1024]
        h = h.transpose(1, 2)                           # [bs, 1024, 1]

        # 保存第 6 层的输出残差（压缩后），用于循环跳回时相加
        residual = h

        # 3. 中间 12 层自定义块，支持循环
        for cycle in range(self.args.num_cycles):
            # 每轮循环都完整走一遍 12 层中间块
            for blk in self.middle_blocks:
                h = blk(h)                              # [bs, 1024, 1]
            # 如果还有下一轮循环，则加上第 6 层的残差（相当于回到第 7 层）
            if cycle < self.args.num_cycles - 1:
                h = h + residual

        # 4. 转换回 Transformer 之前，对 h 做 softmax（沿 1024 维）
        h_sm = F.softmax(h, dim=1)                      # [bs, 1024, 1]
        h = h_sm * self.to_transformer                  # [bs, 1024, 1024]

        # 5. 后 6 层 Transformer
        for blk in self.back_blocks:
            h = blk(h, freqs_cis_back)                       # [bs, 1024, 1024]

        # 6. 输出头
        h = self.final_norm(h)
        logits = self.lm_head(h)                        # [bs, 1024, vocab_size]
        return logits


# -------------------- 训练封装（MLM） --------------------
class TransformerForMLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = MixedArchTransformer(args)
        self.args = args

    def forward(self, input_ids, labels=None):
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.args.vocab_size), labels.view(-1))
        return logits, loss