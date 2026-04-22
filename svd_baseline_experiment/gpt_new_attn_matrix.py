from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from svd_baseline_experiment.svd_q_auto import TruncatedSVDQ
else:
    from .svd_q_auto import TruncatedSVDQ


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
        # GPT-2 Conv1D stores weights as [in_features, out_features], but
        # F.linear expects [out_features, in_features].
        kx = F.linear(x, self.k.t(), self.k_bias)
        vx = F.linear(x, self.v.t(), self.v_bias)
        return torch.cat((qx, kx, vx), dim=-1)
