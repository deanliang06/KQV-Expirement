import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    def __init__(self, dim=64, heads=4, dropout=0.05, mlp_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        # Pre-norm MLP
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, d = 64):
        super().__init__()
        self.d = d
        self.tokenizer = nn.Linear(312,64)
        self.summary_token = nn.Parameter(torch.randn(64))
        self.atten_down_blocks = nn.ModuleList([AttnBlock(64, 4, 0.05, 4) for _ in range(5)])

        
        self.proj = nn.Linear(64, 2*312*64+d*64)
        self.proj_layernorm = nn.LayerNorm(64)

        self.atten_up_blocks = nn.ModuleList([AttnBlock(64, 4, 0.05, 4) for _ in range(5)])   
        self.detokenizer_compress = nn.Linear(64, d)
        self.detokenizer_decompress = nn.Linear(64, d)
        self.feature_detoken = nn.Linear(64, d)

    def forward(self, x):
        #encoding
        x = self.tokenizer(x)
        summary = self.summary_token.view(1,64)
        attn_sum = torch.concatenate((summary,x))
        for block in self.atten_down_blocks:
            attn_sum = block(attn_sum)

        hyper_rep = attn_sum[0]

        #decoding
        proj = self.proj(self.proj_layernorm(hyper_rep)).reshape(2*312+self.d, 64)
        for block in self.atten_up_blocks:
            proj = block(proj)

        split = torch.split(proj, [312, 312, self.d])
        return self.detokenizer_compress(split[0]), self.detokenizer_decompress(split[1]), self.feature_detoken(split[2])
    
class FragmentedKQV(nn.Module):
    def __init__(self, autoencoder, params, d = 64):
        super().__init__()
        self.d = d
        self.autoencoder = autoencoder
        self.register_buffer("base_weights", params["weights"].detach().to(next(autoencoder.parameters()).device))
        self.register_buffer("base_bias", params["bias"].detach().to(next(autoencoder.parameters()).device))

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder
    def forward(self, x):
        comp, decomp, feature = self.autoencoder(self.base_weights)
        comp = comp.reshape(self.d, 312)
        decomp = decomp.reshape(312, self.d)
        feature = feature.reshape(self.d, self.d)
        x = F.linear(x, comp)
        x = F.linear(x, feature)
        return F.linear(x, decomp, self.base_bias)
