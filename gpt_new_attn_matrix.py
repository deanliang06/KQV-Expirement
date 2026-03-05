import torch
import torch.nn.functional as F
from torch import nn
import sys

from unet_auto import CNNBasedFrag

N_EMBED = 768 #Wow I am so good at coding

class GPTAttn(nn.Module):
    def __init__(self, autoencoder, regular_attn, d=64):
        super().__init__()
        self.autoencoder = autoencoder
        self.d = d

        self.register_buffer("base_weights", regular_attn["weights"].detach().to(next(autoencoder.parameters()).device))
        self.register_buffer("base_bias", regular_attn["bias"].detach().to(next(autoencoder.parameters()).device))
        self.k,self.q,self.v = torch.chunk(self.base_weights, 3, dim=1)
        self.k_bias, self.q_bias, self.v_bias = torch.chunk(self.base_bias, 3)

        self.UNet_layer = CNNBasedFrag(autoencoder, {"weights": self.q, "bias": self.q_bias}, ndim=N_EMBED, d=self.d).to(next(autoencoder.parameters()).device)

    def set_autoencoder(self, autoencoder):
        self.UNet_layer.set_autoencoder(autoencoder)

    def forward(self, x):
        qx = self.UNet_layer(x)
        kx = F.linear(x, self.q, self.q_bias)
        vx = F.linear(x, self.v, self.v_bias)
        print(qx.size(), kx.size(), vx.size())
        sys.exit()
        return x