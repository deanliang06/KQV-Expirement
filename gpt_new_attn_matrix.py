from torch import nn

class GPTAttn(nn.Module):
    def __init__(self, autoencoder, regular_attn, d=64):
        super().__init__()

        self.autoencoder = autoencoder
        self.d = d

        self.down = nn.Linear(50257, d)
        self.down_ll = nn.LayerNorm(d)
        self.up = nn.Linear(50257, 312)

        self.register_buffer("base_weights", regular_attn["weights"].detach().to(next(autoencoder.parameters()).device))
        self.register_buffer("base_bias", regular_attn["bias"].detach().to(next(autoencoder.parameters()).device))

    def forward(self, x):
        return x