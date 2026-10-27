"""Train KittenFish SE-ResNet on the existing Stockfish-labelled CSV dataset."""
from __future__ import annotations
import argparse, copy, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from encoding import encode_fen
from model import KittenFishResNet

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="../data/positions_100k.csv.gz")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", default="checkpoints/kittenfish_l4_resnet.pt")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp != "off"
    if args.amp == "on" and device.type != "cuda":
        p.error("--amp on requires CUDA")
    print(f"Device={device}; AMP={use_amp}; batch={args.batch_size}; workers={args.workers}", flush=True)
    df = pd.read_csv(args.data)
    required = {"fen", "eval_cp", "game_id"}
    if not required.issubset(df.columns): raise ValueError(f"Missing columns: {required - set(df.columns)}")
    df = df.dropna(subset=["fen", "eval_cp", "game_id"]).copy()
    df["eval_cp"] = pd.to_numeric(df["eval_cp"], errors="coerce")
    df = df.dropna(subset=["eval_cp"]).reset_index(drop=True)
    if len(df) < 10: raise ValueError("Dataset has too few valid rows")
    y = np.clip(df["eval_cp"].to_numpy(np.float32) / 100.0, -20.0, 20.0)
    split = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    tr, va = next(split.split(df, groups=df["game_id"]))
    print(f"Positions={len(df):,}; games={df.game_id.nunique():,}; train={len(tr):,}; val={len(va):,}", flush=True)
    start = time.time()
    X = np.stack([encode_fen(f) for f in df["fen"]])
    print(f"Encoded {len(X):,} positions in {time.time()-start:.1f}s", flush=True)
    X = torch.from_numpy(X); Y = torch.from_numpy(y)
    train_ds, val_ds = TensorDataset(X[tr], Y[tr]), TensorDataset(X[va], Y[va])
    loader_args = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_args)
    model = KittenFishResNet(channels=args.channels, blocks=args.blocks).to(device)
    print(f"Parameters={sum(q.numel() for q in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        tic = time.time(); model.train(); train_sum = 0.0; train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(xb); loss = loss_fn(pred.float(), yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_sum += loss.detach().item() * len(yb); train_n += len(yb)
        model.eval(); val_sum = abs_sum = pred_sum = pred_sq_sum = 0.0; val_n = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred = model(xb).float()
                val_sum += loss_fn(pred, yb).item() * len(yb)
                abs_sum += (pred-yb).abs().sum().item()
                pred_sum += pred.sum().item(); pred_sq_sum += (pred*pred).sum().item(); val_n += len(yb)
        train_mse, val_mse = train_sum/train_n, val_sum/val_n
        mae_cp = 100*abs_sum/val_n
        scheduler.step(val_mse)
        pred_mean = pred_sum/val_n
        pred_std = max(0.0, pred_sq_sum/val_n - pred_mean**2) ** 0.5
        print(f"Epoch {epoch:02d}/{args.epochs} train_mse={train_mse:.4f} val_mse={val_mse:.4f} val_MAE={mae_cp:.1f}cp pred_mean={pred_mean:.3f} pred_std={pred_std:.3f} lr={optimizer.param_groups[0]['lr']:.2g} time={time.time()-tic:.1f}s", flush=True)
        if val_mse < best:
            best, best_epoch, stale = val_mse, epoch, 0
            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "architecture": "KittenFishResNet-SE-v1", "input_planes": 18,
                "channels": args.channels, "blocks": args.blocks,
                "target": "side_to_move_pawns", "best_val_mse": best,
                "best_val_mae_cp": mae_cp, "epoch": epoch, "args": vars(args),
            }, output)
            print(f"Saved best checkpoint: {output}", flush=True)
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping: no validation improvement for {args.patience} epochs", flush=True); break
    print(f"Finished. Best val MSE={best:.4f} at epoch {best_epoch}; checkpoint={output}", flush=True)

if __name__ == "__main__": main()
