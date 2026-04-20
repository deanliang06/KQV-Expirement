import torch
import torch.nn.functional as F
from torch import nn


class GPTAttn(nn.Module):
    def __init__(self, autoencoder, regular_attn):
        super().__init__()
        device = next(autoencoder.parameters()).device

        self.register_buffer("base_weights", regular_attn["weights"].detach().to(device))
        self.register_buffer("base_bias", regular_attn["bias"].detach().to(device))
        self.q, self.k, self.v = torch.chunk(self.base_weights, 3, dim=1)
        self.q_bias, self.k_bias, self.v_bias = torch.chunk(self.base_bias, 3)

        self.autoencoder = autoencoder
        self._cached_q_weight = None

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder
        self._cached_q_weight = None

    def reconstruct_q(self):
        q_weight = self.autoencoder(self.q)
        self._cached_q_weight = q_weight
        return q_weight

    def clear_cache(self):
        self._cached_q_weight = None

    def forward(self, x):
        q_weight = self.reconstruct_q()
        qx = F.linear(x, q_weight, self.q_bias)
        kx = F.linear(x, self.k, self.k_bias)
        vx = F.linear(x, self.v, self.v_bias)
        return torch.cat((qx, kx, vx), dim=-1)
