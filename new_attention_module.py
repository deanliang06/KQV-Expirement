import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import crop
import sys


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
    
class CNNBasedFrag(nn.Module):
    def __init__(self, autoencoder, params, d=64):
        super().__init__()
        self.d = d
        self.autoencoder = autoencoder
        self.down = nn.Linear(312, d)
        self.down_ll = nn.LayerNorm(64)
        self.up = nn.Linear(d, 312)
        
        self.register_buffer("base_weights", params["weights"].detach().to(next(autoencoder.parameters()).device))
        self.register_buffer("base_bias", params["bias"].detach().to(next(autoencoder.parameters()).device))

    def set_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder

    def forward(self, x):
        x = self.down_ll(self.down(x))
        x = F.linear(x, self.autoencoder(self.base_weights))
        x = self.up(x) + self.base_bias
        return x

class CNNAutoencoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        
        self.d = d
        self.mirror = nn.ReflectionPad2d(64)

        self.first_cont = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.second_cont = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )

        self.third_cont = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.fourth_cont = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.bottom = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 2, 2),
        )

        self.first_up = nn.Sequential(
            nn.Conv2d(1024, 128, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),
        )

        self.second_up = nn.Sequential(
            nn.Conv2d(128, 32, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 1, 2),
            nn.BatchNorm2d(1),
        )

        self.third_res_pre_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )



    def forward(self, x):
        x = self.mirror(x.view(1, 312, 312)).view(1, 1, 440, 440)
        first_map = self.first_cont(x)
        second_map = self.second_cont(first_map)
        third_map = self.third_cont(second_map)
        fourth_map = self.fourth_cont(third_map)
        bottom_map = self.bottom(fourth_map)
        first_up_map = self.first_up(torch.cat((bottom_map, crop(fourth_map, 0, 0, 38, 38)), 1))
        third_map = self.third_res_pre_conv(third_map)
        second_up_map = self.second_up(torch.cat((first_up_map, crop(third_map, 0, 0, 72, 72)), 1))
        if self.d <= 72:
            return crop(second_up_map, 0, 0, self.d, self.d).view(self.d, self.d)
        else:
            raise SystemError("too big of a d") 





    


