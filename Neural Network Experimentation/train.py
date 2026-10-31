#!/usr/bin/env python3
"""Train either small evaluator on existing KFD1 train/validation splits."""
import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import Batches
from models import make_model


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['resnet', 'transformer'], required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--cp-scale', type=float, default=400)
    p.add_argument('--clip-cp', type=float, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--amp', choices=['off', 'fp16', 'bf16'], default='off')
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-train-batches', type=int, default=0, help='0 = full split')
    p.add_argument('--max-val-batches', type=int, default=0, help='0 = full split')
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.lr, a.cp_scale, a.clip_cp) <= 0 or min(a.workers, a.weight_decay, a.max_train_batches, a.max_val_batches) < 0:
        p.error('Invalid training limits')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = a.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device)
    if a.amp != 'off' and device.type != 'cuda':
        p.error('Mixed precision is supported here only on CUDA; use --amp off')
    if a.amp == 'bf16' and not torch.cuda.is_bf16_supported():
        p.error('CUDA device does not support bf16')
    model = make_model(a.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp == 'fp16')
    first, best = 0, float('inf')
    config = vars(a).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config['manifest'] = str(a.manifest.resolve())
    config['device_resolved'] = str(device)
    if a.resume:
        # Checkpoints contain Python/NumPy RNG state: load only your trusted files.
        checkpoint = torch.load(a.resume, map_location='cpu', weights_only=False)
        for key in ('model', 'manifest', 'cp_scale', 'clip_cp', 'seed', 'batch_size',
                    'lr', 'weight_decay', 'epochs', 'amp', 'max_train_batches', 'max_val_batches'):
            if checkpoint['config'][key] != config[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        first, best = checkpoint['epoch'] + 1, checkpoint['best_mae_cp']
        torch.set_rng_state(checkpoint['torch_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.resume and (a.output / 'last.pt').exists():
        raise ValueError('Output already contains a run; use --resume or a new directory')
    (a.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'{a.model}: {sum(p.numel() for p in model.parameters()):,} parameters; device={device}', flush=True)

    def run(split, epoch, limit):
        training = split == 'train'
        model.train(training)
        loader = DataLoader(Batches(a.manifest, split, a.batch_size, a.seed, epoch),
                            batch_size=None, num_workers=a.workers, pin_memory=device.type == 'cuda')
        total_mse, total_abs, count = 0., 0., 0
        for index, (pieces, stm, cp) in enumerate(loader):
            pieces, stm, cp = (t.to(device, non_blocking=True) for t in (pieces, stm, cp))
            target = cp.clamp(-a.clip_cp, a.clip_cp) / a.cp_scale
            if training:
                optimizer.zero_grad(set_to_none=True)
            precision = (torch.autocast('cuda', dtype=torch.float16 if a.amp == 'fp16' else torch.bfloat16)
                         if a.amp != 'off' else nullcontext())
            with torch.set_grad_enabled(training), precision:
                prediction = model(pieces, stm)
                loss = torch.nn.functional.mse_loss(prediction.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at {split} batch {index}')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            n = cp.numel()
            count += n
            total_mse += loss.item() * n
            total_abs += (prediction.detach().float() * a.cp_scale - target * a.cp_scale).abs().sum().item()
            if training and (index + 1) % 100 == 0:
                print(f'epoch {epoch + 1} batch {index + 1}: MAE={total_abs / count:.1f} cp', flush=True)
            if limit and index + 1 >= limit:
                break
        if count == 0:
            raise ValueError(f'No records in {split}')
        return total_mse / count, total_abs / count, count

    for epoch in range(first, a.epochs):
        started = time.perf_counter()
        train_mse, train_mae, train_n = run('train', epoch, a.max_train_batches)
        val_mse, val_mae, val_n = run('val', 0, a.max_val_batches)
        elapsed = time.perf_counter() - started
        improved = val_mae < best
        best = min(best, val_mae)
        scheduler.step()
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                     best_mae_cp=best, config=config, torch_rng=torch.get_rng_state(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        atomic_save(state, a.output / 'last.pt')
        if improved:
            atomic_save(state, a.output / 'best.pt')
        row = dict(epoch=epoch + 1, train_mse=train_mse, train_mae_cp=train_mae,
                   val_mse=val_mse, val_mae_cp=val_mae, train_positions=train_n,
                   val_positions=val_n, seconds=elapsed)
        metrics = a.output / 'metrics.csv'
        exists = metrics.exists()
        with metrics.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
