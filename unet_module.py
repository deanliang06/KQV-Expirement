import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import crop


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





    


