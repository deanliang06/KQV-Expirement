import torch
import torch.nn as nn
import torch.nn.functional as F


class TruncatedSVDQ(nn.Module):
    def __init__(self, q_weight, rank=128):
        super().__init__()
        rows, cols = q_weight.shape
        max_rank = min(rows, cols)
        if rank < 1 or rank > max_rank:
            raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")

        with torch.no_grad():
            q_weight = q_weight.detach().to(torch.float32)
            u, s, vh = torch.linalg.svd(q_weight, full_matrices=True)

        self.rank = rank
        self.register_buffer("u", u)
        self.register_buffer("s", s)
        self.register_buffer("vh", vh)

    def forward(self, x):
        u_r = self.u[:, : self.rank]
        s_r = self.s[: self.rank]
        vh_r = self.vh[: self.rank, :]

        q_hat = (u_r * s_r) @ vh_r
        return F.linear(x, q_hat.t())
