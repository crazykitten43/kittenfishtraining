import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

# Reuse the exact same board encoding as the baseline CNN.
from encoding import encode_fen


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity skip connection."""

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class ChessResNet(nn.Module):
    """Residual CNN that predicts side-to-move evaluation in pawns."""

    def __init__(self, channels=64, blocks=6):
        super().__init__()
        self.input_conv = nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)]
        )
        # Preserve the square-specific information in the flattened board.
        # LeakyReLU keeps gradients flowing if a hidden unit is negative,
        # avoiding the dead dense head observed with ordinary ReLU.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 64, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.relu(self.input_bn(self.input_conv(x)))
        x = self.residual_tower(x)
        return self.head(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description="Train KittenFish residual CNN")
    parser.add_argument("--data", default="data/positions_15k.csv.gz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="AdamW learning rate (default: 0.0001)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--output", default="checkpoints/kittenfish_resnet.pt")
    args = parser.parse_args()

    if args.blocks < 1:
        parser.error("--blocks must be at least 1")
    if args.channels < 1:
        parser.error("--channels must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    print(f"Device: {device}; threads: {args.threads}")
    print(f"Architecture: {args.blocks} residual blocks, {args.channels} channels")

    df = pd.read_csv(args.data)
    required = {"fen", "eval_cp", "game_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df = df.dropna(subset=["fen", "eval_cp", "game_id"]).copy()
    df["eval_cp"] = pd.to_numeric(df["eval_cp"], errors="coerce")
    df = df.dropna(subset=["eval_cp"]).reset_index(drop=True)

    print(f"Positions: {len(df):,}")
    print(f"Unique games: {df['game_id'].nunique():,}")
    print(f"Evaluation range: {df['eval_cp'].min():.0f} to "
          f"{df['eval_cp'].max():.0f} cp")

    # Match train.py: targets are side-to-move evaluations, converted to
    # pawns and clipped to [-20, +20].
    y = np.clip(
        df["eval_cp"].to_numpy(dtype=np.float32) / 100.0, -20.0, 20.0
    )

    # Same split method and seed as train.py, keeping each game in one split.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=args.seed
    )
    train_idx, val_idx = next(splitter.split(df, groups=df["game_id"]))
    print(f"Train: {len(train_idx):,}; validation: {len(val_idx):,}")

    start = time.time()
    features = np.stack([encode_fen(fen) for fen in df["fen"]])
    print(f"Encoded positions in {time.time() - start:.1f}s")

    X = torch.from_numpy(features)
    Y = torch.from_numpy(y)

    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = ChessResNet(channels=args.channels, blocks=args.blocks).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0
        train_count = 0

        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_count += len(yb)

        model.eval()
        val_loss = 0.0
        abs_error = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                predictions = model(xb)
                val_loss += loss_fn(predictions, yb).item() * len(yb)
                abs_error += (predictions - yb).abs().sum().item()
                val_count += len(yb)

        train_mse = train_loss / train_count
        val_mse = val_loss / val_count
        mae_cp = 100.0 * abs_error / val_count

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train MSE={train_mse:.4f} | "
            f"val MSE={val_mse:.4f} | "
            f"val MAE={mae_cp:.1f} cp | "
            f"{time.time() - start:.1f}s"
        )

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "channels": args.channels,
                "blocks": args.blocks,
                "input_planes": 18,
                "target": "side_to_move_pawns",
                "best_val_mse": best_val,
                "epoch": epoch,
                "args": vars(args),
                "architecture": "ChessResNet",
            }, output)
            print(f"Saved best checkpoint: {output}")

    print(f"Finished. Best validation MSE: {best_val:.4f}")
    print(f"Model saved at: {output}")


if __name__ == "__main__":
    main()
