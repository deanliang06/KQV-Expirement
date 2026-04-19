import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEmbeddingAutoencoder(nn.Module):
    def __init__(self, ndim=768, embedding_dim=128):
        super().__init__()

        self.ndim = ndim
        self.embedding_dim = embedding_dim
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
        )

        self.embedding_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.q_decoder = nn.Linear(embedding_dim, ndim * ndim)

    def encode(self, x):
        x = self.mirror(x.view(1, self.ndim, self.ndim)).view(
            1, 1, self.ndim + 128, self.ndim + 128
        )
        x = self.first_cont(x)
        x = self.second_cont(x)
        x = self.third_cont(x)
        x = self.fourth_cont(x)
        x = self.bottom(x)
        return self.embedding_proj(self.embedding_pool(x))

    def decode(self, embedding):
        return self.q_decoder(embedding).view(self.ndim, self.ndim)

    def forward(self, x):
        embedding = self.encode(x)
        return self.decode(embedding)
