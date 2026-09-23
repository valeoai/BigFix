# Transformer Encoder architecture
# some part have been borrowed from:
#   - NanoGPT: https://github.com/karpathy/nanoGPT
#   - DiT: https://github.com/facebookresearch/DiT

import torch
from torch import nn

from einops import rearrange

from bigfix.network.transformer_block import RMSNorm, Attention, FeedForward, modulate, AdaNorm


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.ln1 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.attn = Attention(dim, heads, dropout=dropout)

        self.ln2 = RMSNorm(dim, linear=True, bias=False, eps=1e-5)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, cond, mask=None):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mlp(cond).chunk(6, dim=1)
        x = x + alpha1.unsqueeze(1) * self.attn(modulate(self.ln1(x), gamma1, beta1), mask=mask)
        x = x + alpha2.unsqueeze(1) * self.ff(modulate(self.ln2(x), gamma2, beta2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def forward(self, x, cond, mask=None):
        for block in self.layers:
            x = block(x, cond, mask=mask)
        return x


class Transformer(nn.Module):
    """ DiT-like transformer with adaLayerNorm with zero initializations """
    def __init__(self, input_size=16, hidden_dim=768, codebook_size=1024,
                 depth=12, heads=16, mlp_dim=3072, dropout=0., nclass=1000,
                 register=1, proj=1):
        super().__init__()

        self.nclass = nclass                                             # Number of classes
        self.h = self.w = input_size                                      # Number of tokens as input
        self.hidden_dim = hidden_dim                                     # Hidden dimension of the transformer
        self.codebook_size = codebook_size                               # Amount of code in the codebook
        self.proj = proj                                                 # Projection
        self.register = register                                         # add register token

        self.c = self.hidden_dim // (proj ** 2)
        self.cls_emb = nn.Embedding(nclass + 1, hidden_dim)              # Embedding layer for the class token
        self.tok_emb = nn.Embedding(codebook_size + 1, self.c)  # Embedding layer for the 'visual' token
        self.pos_emb = nn.Embedding((self.h * self.w), self.c) # Learnable Positional Embedding

        # The Transformer Encoder a la BERT :)
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        self.last_norm = AdaNorm(x_dim=self.c, y_dim=hidden_dim)   # Last Norm

        self.head = nn.Linear(self.c, codebook_size + 1)
        self.head.weight = self.tok_emb.weight  # weight tied with the tok_emb layer

        self.register = register
        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        if self.proj > 1:
            self.in_proj = nn.Sequential(
                nn.Conv2d(self.c, hidden_dim, kernel_size=self.proj, stride=self.proj),
                nn.SiLU()
            )
            self.pos_emb_coarse = nn.Embedding((self.h * self.w) // (proj ** 2), hidden_dim)
            self.out_proj = nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0
            ).to(memory_format=torch.channels_last)

        self.initialize_weights()  # Init weight

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.cls_emb.weight, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Zero-out adaNorm modulation layers in blocks:
        for block in self.transformer.layers:
            nn.init.constant_(block.mlp[1].weight, 0)
            nn.init.constant_(block.mlp[1].bias, 0)

        # Init embedding
        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

        # Init norm layer
        nn.init.constant_(self.last_norm.mlp[1].weight, 0)
        nn.init.constant_(self.last_norm.mlp[1].bias, 0)

    def forward(self, x, y, drop_label, mask=None):
        b, h, w = x.size()
        x = x.reshape(b, h*w)

        # Drop the label if drop_label
        y = torch.where(drop_label, torch.full_like(y, self.nclass), y)
        y = self.cls_emb(y)
        x = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(0, h * w, dtype=torch.long, device=x.device))
        x = x + pos

        # reshape, proj to smaller space (patchify!)
        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.c).contiguous()
            x = self.in_proj(x)
            _, _, h, w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w, b=b, c=self.hidden_dim).contiguous()

            # Add coarse pos emb
            coarse_pos = torch.arange(0, h * w, device=x.device)
            coarse_pos = self.pos_emb_coarse(coarse_pos)
            x = x + coarse_pos

        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([x, self.reg_tokens(reg).expand(b, self.register, self.hidden_dim)], dim=1)

        x = self.transformer(x, y, mask=mask)

        # drop the register
        x = x[:, :h*w].contiguous()

        if self.proj > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w, b=b, c=self.hidden_dim).contiguous()
            x = self.out_proj(x)
            x = rearrange(x, 'b (c s1 s2) h w -> b (h s1 w s2) c', s1=self.proj, s2=self.proj, b=b, h=h, w=w,
                          c=self.c).contiguous()

        x = self.last_norm(x, y)
        logit = self.head(x)

        return logit


if __name__ == "__main__":
    from thop import profile

    # for size in ["tiny", "small", "base", "large", "xlarge"]:
    size = "base"
    print(size)
    if size == "tiny":
        hidden_dim, depth, heads = 384, 6, 6
    elif size == "small":
        hidden_dim, depth, heads = 512, 8, 6
    elif size == "base":
        hidden_dim, depth, heads = 768, 12, 12
    elif size == "large":
        hidden_dim, depth, heads = 1024, 24, 16
    elif size == "xlarge":
        hidden_dim, depth, heads = 1152, 28, 16
    else:
        hidden_dim, depth, heads = 768, 12, 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    input_size = 128
    model = Transformer(input_size=input_size, nclass=1000, hidden_dim=hidden_dim, codebook_size=16834,
                        depth=depth, heads=heads, mlp_dim=hidden_dim * 4, dropout=0.1, proj=4).to(device)
    # model = torch.compile(model)
    code = torch.randint(0, 16384, size=(1, input_size, input_size)).to(device)
    cls = torch.randint(0, 1000, size=(1,)).to(device)
    d_label = (torch.rand(1) < 0.1).to(device)

    out = model(code, cls, d_label)
    print(out.size())
    # flops, params = profile(model, inputs=(code, cls, d_label))
    # print(f"FLOPs: {flops//1e9:.2f}G, Params: {params/1e6:.2f}M")
