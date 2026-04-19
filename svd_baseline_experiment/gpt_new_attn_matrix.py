import torch
import torch.nn.functional as F
from torch import nn

from svd_q_auto import TruncatedSVDQ


class GPTAttn(nn.Module):
    def __init__(self, regular_attn, rank=128):
        super().__init__()
        device = regular_attn["weights"].device

        self.register_buffer("base_weights", regular_attn["weights"].detach().to(device))
        self.register_buffer("base_bias", regular_attn["bias"].detach().to(device))
        self.q, self.k, self.v = torch.chunk(self.base_weights, 3, dim=1)
        self.q_bias, self.k_bias, self.v_bias = torch.chunk(self.base_bias, 3)

        self.svd_q = TruncatedSVDQ(self.q, rank=rank).to(device)

    def forward(self, x):
        qx = self.svd_q(x) + self.q_bias
        kx = F.linear(x, self.k, self.k_bias)
        vx = F.linear(x, self.v, self.v_bias)
        return torch.cat((qx, kx, vx), dim=-1)
