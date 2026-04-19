import torch
import torch.nn as nn
import torch.nn.functional as F


class TruncatedSVDQ(nn.Module):
    def __init__(self, q_weight, rank=128):
        super().__init__()
        max_rank = min(q_weight.shape)
        if rank < 1 or rank > max_rank:
            raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")

        with torch.no_grad():
            u, s, vh = torch.linalg.svd(q_weight.detach().to(torch.float32), full_matrices=False)
            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]

            left = u_r * s_r.unsqueeze(0)
            right = vh_r

        self.rank = rank
        self.register_buffer("left", left)
        self.register_buffer("right", right)

    def forward(self, x):
        x = F.linear(x, self.right)
        return F.linear(x, self.left)
