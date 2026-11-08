"""Matched-input scalar chess evaluators; outputs are White centipawns / scale."""
import torch
from torch import nn


def planes(pieces, stm):
    # a1..h8, White PNBRQK=1..6, Black=7..12; fixed White orientation.
    x = torch.nn.functional.one_hot(pieces.long(), 13)[..., 1:].float()
    return torch.cat((x, stm.float()[:, None, None].expand(-1, 64, 1)), -1)


class SEBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // 8, 1),
                                nn.ReLU(), nn.Conv2d(channels // 8, channels, 1), nn.Sigmoid())

    def forward(self, x):
        y = self.body(x)
        return torch.relu(x + y * self.se(y))


class SmallResNet(nn.Module):
    def __init__(self, blocks=10, attention_layers=0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(13, 128, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        self.blocks = nn.Sequential(*(SEBlock() for _ in range(blocks)))
        self.attention = nn.ModuleList(nn.TransformerEncoderLayer(
            128, 4, 256, dropout=0.0, activation='gelu', batch_first=True,
            norm_first=True) for _ in range(attention_layers))
        if attention_layers:
            self.position = nn.Parameter(torch.empty(1, 64, 128))
            nn.init.normal_(self.position, std=0.02)
        self.head = nn.Sequential(nn.Conv2d(128, 8, 1), nn.ReLU(), nn.Flatten(),
                                  nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, pieces, stm):
        x = planes(pieces, stm).transpose(1, 2).reshape(-1, 13, 8, 8)
        x = self.blocks(self.stem(x))
        if self.attention:
            # Keep one token per square and feed the same spatial value head.
            tokens = x.flatten(2).transpose(1, 2)
            tokens = tokens + self.position.to(tokens.dtype)
            for block in self.attention:
                tokens = block(tokens)
            x = tokens.transpose(1, 2).reshape_as(x)
        return self.head(x).squeeze(-1)


class SquareTransformer(nn.Module):
    """Transformer value evaluator; layer count is the model's depth."""
    def __init__(self, layers):
        super().__init__()
        self.embed = nn.Linear(13, 256)
        self.position = nn.Parameter(torch.empty(1, 64, 256))
        nn.init.normal_(self.position, std=0.02)
        # Construct independently: avoid identical initial weights across cloned layers.
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            256, 8, 1024, dropout=0.1, activation='gelu', batch_first=True,
            norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, pieces, stm):
        x = self.embed(planes(pieces, stm)) + self.position
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(1)).squeeze(-1)


class SmallTransformer(SquareTransformer):
    def __init__(self):
        super().__init__(layers=8)


class DeepTransformer(SquareTransformer):
    """16-layer, width-256 transformer; keeps the small model's other settings."""
    def __init__(self):
        super().__init__(layers=16)


RESNET_SWEEP = {
    'resnet_6': (6, 0),
    'resnet': (10, 0),
    'resnet_16': (16, 0),
    'resnet_24': (24, 0),
    'hybrid_10_1': (10, 1),
    'hybrid_10_3': (10, 3),
}
TRANSFORMERS = {'transformer': SmallTransformer, 'transformer_deep': DeepTransformer}
MODEL_NAMES = (*RESNET_SWEEP, *TRANSFORMERS)


def make_model(name):
    if name in RESNET_SWEEP:
        return SmallResNet(*RESNET_SWEEP[name])
    return TRANSFORMERS[name]()
