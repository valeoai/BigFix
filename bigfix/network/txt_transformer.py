# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn
from einops import rearrange
from bigfix.network.transformer_block import RMSNorm, Attention, FeedForward, CrossAttention, BlockUp1d, BlockDown2d


def modulate(x, gamma, beta):
    return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


def init_identity(in_proj, out_proj, proj):
    c_in = in_proj.in_channels
    c_hidden = in_proj.out_channels

    # --- in_proj: Conv2d weight shape [out_ch, in_ch, kH, kW] ---
    with torch.no_grad():
        # Zero everything
        in_proj.weight.zero_()
        in_proj.bias.zero_()

        # For identity: copy each input channel to first output channel
        for i in range(min(c_in, c_hidden)):
            in_proj.weight[i, i, :, :] = 1.0 / (proj * proj)

    # --- out_proj: Linear weight shape [c_hidden, c_in * proj^2] ---
    with torch.no_grad():
        out_proj.weight.zero_()
        out_proj.bias.zero_()
        for i in range(c_in):
            for j in range(proj*proj):
                idx = i * proj*proj + j
                if i < c_hidden:
                    out_proj.weight[i, idx] = 1.0


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, ada_ln=False, dropout=0.):
        super().__init__()
        self.ada_ln = ada_ln

        self.attn = Attention(dim, heads, dropout=dropout)
        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

        self.cross_attn = CrossAttention(dim, heads, dropout=dropout)
        self.ln1_bis = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

    def forward(self, x, text_cond, style=None, text_mask=None, mask=None):
        """
        Args:
            x: main hidden states [B, N, D]
            text_cond: optional key/values for cross-attention
            style: optional conditioning tensor [B, D] if ada_ln=True
        """

        if style is not None:
            # Generate all modulation parameters from style
            gamma1, beta1, alpha1, gamma2, beta2, alpha2, delta = style # self.mlp(style).chunk(6, dim=1)

            # --- Self-attention with AdaLN modulation ---
            x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), mask=mask)

            # --- Cross-attention ---
            x = x + delta.unsqueeze(1) * self.cross_attn(self.ln1_bis(x), text_cond, text_mask)

            # --- Feed-forward with AdaLN modulation ---
            x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))

        else:
            # --- Standard transformer block (no AdaLN) ---
            # print("No condition!!!!!")
            x = x + self.attn(self.ln1(x), mask=mask)
            x = x + self.cross_attn(self.ln1_bis(x), text_cond, text_mask)
            x = x + self.ff(self.ln2(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, ada_ln=False, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, ada_ln=ada_ln, dropout=dropout))

    def forward(self, x, text_cond, style=None, text_mask=None, mask=None):
        for block in self.layers:
            x = block(x, text_cond=text_cond, style=style, text_mask=text_mask, mask=mask)
        return x


class TextProjector(nn.Module):
    def __init__(self, txt_dim, hidden_dim, heads=8, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(txt_dim, hidden_dim) if txt_dim != hidden_dim else nn.Identity()

        # Keep number of heads valid for hidden_dim.
        nhead = min(heads, hidden_dim)
        while hidden_dim % nhead != 0 and nhead > 1:
            nhead -= 1

        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_embs):
        text_embs = self.in_proj(text_embs)
        text_embs = self.block(text_embs)
        return self.out_norm(text_embs)


class Transformer(nn.Module):
    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024, txt_dim=2048,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., register=1, proj=1, extra_cond=False):
        super().__init__()

        self.h = self.w = input_size
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.proj = proj
        self.register = register
        self.txt_dim = txt_dim
        self.c = self.hidden_dim #  // (proj ** 2)
        self.extra_cond = extra_cond

        self.tok_emb = nn.Embedding(codebook_size + 1, self.c)
        self.pos_emb = nn.Embedding(self.h * self.w, self.c)
        self.txt_proj = nn.Linear(self.txt_dim, self.hidden_dim)
        # self.txt_proj = TextProjector(
        #     txt_dim=self.txt_dim,
        #     hidden_dim=self.hidden_dim,
        #     heads=heads,
        #     dropout=dropout,
        # )

        if self.proj > 1:
            self.in_proj = nn.Sequential(
                nn.Conv2d(hidden_dim * proj ** 2, hidden_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )

            self.out_proj = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim * proj ** 2, kernel_size=1),
            )

            self.last_norm_proj = RMSNorm(hidden_dim, linear=True, bias=False)

        if self.extra_cond:
            # map arbitrary conditioning vector (e.g. reward/style embedding) into transformer hidden space
            self.reward_proj = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 8)  # gamma1,beta1,alpha1, gamma2,beta2,alpha2, alpha_cross
            )

        self.transformer = TransformerEncoder(
            dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, ada_ln=self.extra_cond, dropout=dropout
        )
        self.last_norm = RMSNorm(hidden_dim, linear=True, bias=False, eps=1e-5)

        self.head = nn.Linear(self.c, codebook_size + 1)
        self.head.weight = self.tok_emb.weight

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Init embedding
        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

        # init shared_adaln
        if self.extra_cond:
            nn.init.normal_(self.reward_proj[0].weight, std=0.01)
            nn.init.constant_(self.reward_proj[0].bias, 0.0)
            nn.init.normal_(self.reward_proj[2].weight, std=0.01)
            nn.init.constant_(self.reward_proj[2].bias, 0.0)

    def forward(self, x, text_embs, cond=None, text_mask=None, mask=None):
        """
        x: (B, H, W) integer tokens
        y: class labels (B,)
        drop_label: boolean mask (B,) indicating label dropout
        text_embs: (B, T, D) textual token embeddings (already projected to hidden_dim)
        text_mask: (B, T) boolean mask where True indicates valid text tokens
        """
        b, h, w = x.size()
        x = x.reshape(b, h*w)

        # label embedding as before
        x = self.tok_emb(x)
        text_embs = self.txt_proj(text_embs)

        pos = torch.arange(0, h * w, dtype=torch.long, device=x.device)
        pos = self.pos_emb(pos)
        x = x + pos

        if self.proj > 1:
            r = self.proj

            # PixelUnshuffle
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = rearrange(x, 'b c (h r1) (w r2) -> b (c r1 r2) h w', r1=r, r2=r)

            # Conv projection
            x = self.in_proj(x)

            # grid -> tokens
            h, w = h // r, w // r
            x = rearrange(x, 'b c h w -> b (h w) c')

        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([x, self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)], dim=1)

        if self.extra_cond:
            style = self.reward_proj(cond).chunk(8, dim=1)  # (B, (1+7)*hidden_dim)
            x = torch.cat([x, style[0].view(b, 1, self.hidden_dim)], dim=1)
            x = self.transformer(x, text_cond=text_embs, text_mask=text_mask, mask=mask, style=style[1:])
        else:
            x = self.transformer(x, text_cond=text_embs, text_mask=text_mask, mask=mask)

        # drop the register tokens if present
        x = x[:, :h*w].contiguous()
        x = self.last_norm(x)

        if self.proj > 1:
            r = self.proj
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.out_proj(x)
            x = rearrange(x, 'b (c r1 r2) h w -> b (h r1 w r2) c', r1=r, r2=r)

            x = self.last_norm_proj(x)

        logit = self.head(x)
        return logit


if __name__ == "__main__":

    def one_step_training_test(model, device="cuda"):
        model = model.to(device)
        model.train()

        # --- Fake inputs ---
        B = 2
        x = torch.randint(0, model.codebook_size, (B, model.h, model.w), device=device)
        text_embs = torch.randn(B, 77, model.txt_dim, device=device)
        cond = torch.randn(B, 3, device=device)  # conditioning vector

        # --- Optimizer (only for test) ---
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        # --- Forward ---
        out = model(x, text_embs, cond)
        loss = out.mean()

        # --- Backward ---
        optim.zero_grad()
        loss.backward()

        print("\n=== GRAD CHECK ===")

        # 1. Check reward_proj
        if model.extra_cond:
            print("\nreward_proj:")
            for name, p in model.reward_proj.named_parameters():
                if p.grad is None:
                    print(f"  {name:20s} → NO GRAD ❌")
                else:
                    print(f"  {name:20s} → grad_norm={p.grad.norm().item():.6f} ✅")

        # 2. Check transformer block weights
        print("\nTransformer block sample gradients:")
        for name, p in model.transformer.layers[0].named_parameters():
            if "attn" in name or "cross_attn" in name:
                if p.grad is None:
                    print(f"  {name:20s} → NO GRAD ❌")
                else:
                    print(f"  {name:20s} → grad_norm={p.grad.norm().item():.6f} ✅")
                # break  # show only first param

        # 3. Final head
        print("\nOutput head:")
        if model.head.weight.grad is None:
            print("  head.weight → NO GRAD ❌")
        else:
            print(f"  head.weight → grad_norm={model.head.weight.grad.norm().item():.6f} ✅")

        print("\nLoss:", loss.item())
        print("====================\n")


    # transformer = Transformer(
    #     input_size=,
    #     hidden_dim=384,
    #     codebook_size=16834,
    #     txt_dim=2048,
    #     depth=6,
    #     heads=6,
    #     mlp_dim=1536,
    #     extra_cond=True,
    #     proj=2
    # )

    # one_step_training_test(transformer)
