"""KittenFish v1 SE-ResNet value network for 8x8 chess boards."""
from __future__ import annotations
import torch
from torch import nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.gate(self.pool(x))

class SEResidualBlock(nn.Module):
    """Two 3x3 convolutions, identity skip, and channel-wise SE gating."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.se(y)
        return self.relu(x + y)

class KittenFishResNet(nn.Module):
    """Predict side-to-move evaluation in pawns (clipped training labels ±20)."""
    def __init__(self, input_planes: int = 18, channels: int = 64, blocks: int = 8):
        super().__init__()
        if input_planes != 18:
            raise ValueError("This v1 encoder requires exactly 18 input planes")
        if channels < 8 or blocks < 1:
            raise ValueError("channels must be >= 8 and blocks must be >= 1")
        self.input_planes, self.channels, self.blocks = input_planes, channels, blocks
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*(SEResidualBlock(channels) for _ in range(blocks)))
        # Keep square-specific information for the value head.
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 64, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.input_planes, 8, 8):
            raise ValueError(f"Expected [batch, {self.input_planes}, 8, 8], got {tuple(x.shape)}")
        return self.value_head(self.trunk(self.stem(x))).squeeze(-1)
