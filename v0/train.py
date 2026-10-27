import argparse
import copy
import random
import time
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit


def encode_fen(fen):
    board = chess.Board(fen)
    x = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes: white P,N,B,R,Q,K then black P,N,B,R,Q,K.
    piece_index = {
        (chess.WHITE, chess.PAWN): 0,
        (chess.WHITE, chess.KNIGHT): 1,
        (chess.WHITE, chess.BISHOP): 2,
        (chess.WHITE, chess.ROOK): 3,
        (chess.WHITE, chess.QUEEN): 4,
        (chess.WHITE, chess.KING): 5,
        (chess.BLACK, chess.PAWN): 6,
        (chess.BLACK, chess.KNIGHT): 7,
        (chess.BLACK, chess.BISHOP): 8,
        (chess.BLACK, chess.ROOK): 9,
        (chess.BLACK, chess.QUEEN): 10,
        (chess.BLACK, chess.KING): 11,
    }

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        x[piece_index[(piece.color, piece.piece_type)], row, col] = 1.0

    # Side to move.
    x[12, :, :] = float(board.turn == chess.WHITE)

    # Castling rights: white K/Q, black K/Q.
    rights = [
        (chess.WHITE, 7),
        (chess.WHITE, 0),
        (chess.BLACK, 7),
        (chess.BLACK, 0),
    ]
    for i, (color, rook_file) in enumerate(rights):
        king_sq = board.king(color)
        if king_sq is not None:
            rook_sq = chess.square(
                rook_file, chess.square_rank(king_sq)
            )
            if board.castling_rights & chess.BB_SQUARES[rook_sq]:
                x[13 + i, :, :] = 1.0

    # En-passant target square, if present.
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        x[17, row, col] = 1.0

    return x


class ChessCNN(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/positions_15k.csv.gz"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/kittenfish.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cpu")
    print(f"Device: {device}; threads: {args.threads}")

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

    # eval_cp is from the side-to-move perspective.
    # Convert centipawns to pawns and clip extreme values.
    y = np.clip(df["eval_cp"].to_numpy(dtype=np.float32) / 100.0,
                -20.0, 20.0)

    # Keep all positions from each game in the same split.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=args.seed
    )
    train_idx, val_idx = next(
        splitter.split(df, groups=df["game_id"])
    )
    print(f"Train: {len(train_idx):,}; validation: {len(val_idx):,}")

    # Encode each board once.
    start = time.time()
    features = np.stack([encode_fen(f) for f in df["fen"]])
    print(f"Encoded positions in {time.time() - start:.1f}s")

    X = torch.from_numpy(features)
    Y = torch.from_numpy(y)

    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0
    )

    model = ChessCNN().to(device)
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
                "channels": 64,
                "input_planes": 18,
                "target": "side_to_move_pawns",
                "best_val_mse": best_val,
                "epoch": epoch,
                "args": vars(args),
            }, output)
            print(f"Saved best checkpoint: {output}")

    print(f"Finished. Best validation MSE: {best_val:.4f}")
    print(f"Model saved at: {output}")


if __name__ == "__main__":
    main()
