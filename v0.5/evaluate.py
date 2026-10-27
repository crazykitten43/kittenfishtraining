"""Evaluate a saved checkpoint on a game-grouped holdout split."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from encoding import encode_fen
from model import KittenFishResNet

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/positions_100k.csv.gz")
    p.add_argument("--checkpoint", default="checkpoints/kittenfish_l4_resnet.pt")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.data).dropna(subset=["fen", "eval_cp", "game_id"]).copy()
    df["eval_cp"] = pd.to_numeric(df["eval_cp"], errors="coerce"); df = df.dropna(subset=["eval_cp"]).reset_index(drop=True)
    y = np.clip(df.eval_cp.to_numpy(np.float32)/100, -20, 20)
    _, vi = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed).split(df, groups=df.game_id))
    X = torch.from_numpy(np.stack([encode_fen(f) for f in df.fen.iloc[vi]]))
    Y = torch.from_numpy(y[vi])
    loader = DataLoader(TensorDataset(X,Y), batch_size=a.batch_size, shuffle=False, pin_memory=device.type=="cuda")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = KittenFishResNet(channels=ckpt["channels"], blocks=ckpt["blocks"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    preds=[]
    with torch.inference_mode():
        for xb,_ in loader: preds.extend(model(xb.to(device)).cpu().tolist())
    pred=np.asarray(preds); target=Y.numpy()
    print(f"Device: {device}; positions: {len(target):,}")
    print(f"MSE: {np.mean((pred-target)**2):.4f} pawns²")
    print(f"MAE: {np.mean(np.abs(pred-target))*100:.1f} cp")
    print(f"Prediction mean/std: {pred.mean():.3f}/{pred.std():.3f} pawns")
    print(f"Target mean/std: {target.mean():.3f}/{target.std():.3f} pawns")
if __name__ == "__main__": main()
