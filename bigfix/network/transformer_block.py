import math

import torch
from torch import nn
import torch.nn.functional as F


def param_count(archi, model):
    print(f"Size of model {archi}: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, dim, h_dim, multiple_of=256, bias=False, dropout=0.):
        super().__init__()
        self.dropout = dropout
        # swinGLU
        h_dim = int(2 * h_dim / 3)
        # make sure it is a power of 256
        h_dim = multiple_of * ((h_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, h_dim, bias=bias)
        self.w2 = nn.Linear(h_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, h_dim, bias=bias)

    def forward(self, x):
        # SwiGLU activation
        x = F.silu(self.w1(x)) * self.w3(x)
        if self.dropout > 0. and self.training:
            x = F.dropout(x, self.dropout)

        return self.w2(x)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim, linear=False, bias=False)
        self.key_norm = RMSNorm(dim, linear=False, bias=False)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., use_flash=True, bias=False):
        super().__init__()
        self.flash = use_flash # use flash attention?
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.qk_norm = QKNorm(num_heads * self.head_dim)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x, mask=None):
        b, h_w, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # normalize queries and keys
        xq, xk = self.qk_norm(xq, xk, xv)
        xq = xq.view(b, h_w, self.n_local_heads, self.head_dim)
        xk = xk.view(b, h_w, self.n_local_heads, self.head_dim)
        xv = xv.view(b, h_w, self.n_local_heads, self.head_dim)

        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        # attention
        if self.flash:
            if mask is not None:
                mask = mask.view(b, 1, 1, h_w)
            output = F.scaled_dot_product_attention(xq, xk, xv, mask, dropout_p=self.dropout if self.training else 0.)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(b, h_w, -1)
        # output projection
        proj = self.wo(output)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj

class GroupAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, group_size=64, dropout=0., use_flash=True, bias=False):
        super().__init__()
        self.flash = use_flash # use flash attention?
        self.n_local_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.group_size = group_size

        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.qk_norm = QKNorm(num_heads * self.head_dim)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x, mask=None):
        b, L, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.qk_norm(xq, xk, xv)

        xq = xq.view(b, L, self.n_local_heads, self.head_dim)
        xk = xk.view(b, L, self.n_local_heads, self.head_dim)
        xv = xv.view(b, L, self.n_local_heads, self.head_dim)

        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))
        # (b, heads, L, d)

        # ---- GROUPING ----
        g = self.group_size
        assert L % g == 0, "Sequence length must be divisible by group size"
        num_groups = L // g

        xq = xq.view(b, self.n_local_heads, num_groups, g, self.head_dim)
        xk = xk.view(b, self.n_local_heads, num_groups, g, self.head_dim)
        xv = xv.view(b, self.n_local_heads, num_groups, g, self.head_dim)

        # merge batch and groups
        xq = xq.reshape(b * num_groups, self.n_local_heads, g, self.head_dim)
        xk = xk.reshape(b * num_groups, self.n_local_heads, g, self.head_dim)
        xv = xv.reshape(b * num_groups, self.n_local_heads, g, self.head_dim)

        if mask is not None:
            mask = mask.view(b, 1, 1, L)
            mask = mask.view(b, 1, num_groups, g)
            mask = mask.reshape(b * num_groups, 1, 1, g)

        # ---- ATTENTION PER GROUP ----
        output = F.scaled_dot_product_attention(
            xq, xk, xv, mask,
            dropout_p=self.dropout if self.training else 0.
        )

        # ---- UNGROUP ----
        output = output.view(b, num_groups, self.n_local_heads, g, self.head_dim)
        output = output.permute(0, 1, 3, 2, 4).contiguous()
        output = output.view(b, L, -1)

        proj = self.wo(output)
        if self.dropout > 0 and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj
    

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output


class AdaNorm(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.norm_final = RMSNorm(x_dim, linear=True, bias=True, eps=1e-5)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, x_dim * 2)) # 1024*6 * y_dim

    def forward(self, x, y):
        shift, scale = self.mlp(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=False, use_flash=True):
        super().__init__()
        self.flash = use_flash
        self.n_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.wq = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)  # used on text
        self.wv = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)  # used on text
        self.wo = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        self.dropout = dropout

    def forward(self, q_x, kv_text, text_mask=None):
        """
        q_x: (B, Nq, D)  -- image tokens (queries)
        kv_text: (B, Nt, D) -- text embeddings (keys/values)
        text_mask: optional boolean mask (B, Nt) where True = keep, False = mask out
        """
        b, nq, _ = q_x.shape
        _, nt, _ = kv_text.shape

        q = self.wq(q_x).view(b, nq, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, Hd)
        k = self.wk(kv_text).view(b, nt, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nt, Hd)
        v = self.wv(kv_text).view(b, nt, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nt, Hd)

        # If flash scaled_dot_product_attention available, use it
        if self.flash:
            # flash expects (B, H, Lq, D) etc. but PyTorch's scaled_dot_product_attention uses (B, H, Lq, D)
            attn_mask = None
            if text_mask is not None:
                # convert (B, Nt) -> (B, 1, 1, Nt) with True=keep -> we need False where masked -> use boolean mask
                # scaled_dot_product_attention expects Bool mask with True indicating keep (or is it the opposite)? using this shape is safe for torch>=2.1
                attn_mask = text_mask.view(b, 1, 1, nt)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout if self.training else 0.)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,Nq,Nt)
            if text_mask is not None:
                # text_mask: (B, Nt) -> expand to (B,1,1,Nt); masked positions set to -inf
                mask = (~text_mask).view(b, 1, 1, nt)  # True where masked
                scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores.float(), dim=-1).type_as(q)
            out = torch.matmul(attn, v)  # (B,H,Nq,Hd)
        out = out.transpose(1, 2).contiguous().view(b, nq, -1)  # (B, Nq, H*Hd)
        proj = self.wo(out)
        if self.dropout > 0. and self.training:
            proj = F.dropout(proj, self.dropout)
        return proj


class BlockDown2d(nn.Module):
    def __init__(self, in_ch, hidden_dim, down_scale):
        super().__init__()
        mid = hidden_dim // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=down_scale, stride=down_scale),
            nn.GroupNorm(1, mid),
            nn.GELU(),
            nn.Conv2d(mid, hidden_dim, kernel_size=1),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, stride=down_scale),
        )

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class BlockUp1d(nn.Module):
    def __init__(self, hidden_dim, out_ch, proj):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, out_ch * proj * proj)
        self.norm = RMSNorm(out_ch * proj * proj, linear=True, bias=False, eps=1e-5)

    def forward(self, x):
        x, gate = self.fc(x).chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        x = self.out(x)
        return self.norm(x)


class FourierEmbed(torch.nn.Module):
    def __init__(self, num_freqs=8):
        """
        num_freqs: number of frequency bands
        max_freq: maximum frequency
        """
        super().__init__()
        self.num_freqs = num_freqs

        # Precompute frequencies
        self.register_buffer(
            "freq_bands",
            2 ** torch.arange(num_freqs, dtype=torch.float32) * math.pi
        )

    def forward(self, x):
        """
        x: tensor of shape [B] or [B, 1], scalar values like H or W
        returns: [B, num_freqs*2] Fourier embedding
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # [B, 1]

        # Compute sinusoidal embeddings
        x_proj = x * self.freq_bands  # [B, num_freqs]
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [B, num_freqs*2]
        return emb