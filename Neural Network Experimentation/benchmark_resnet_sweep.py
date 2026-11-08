#!/usr/bin/env python3
"""Train the matched SE-ResNet sweep and benchmark its best checkpoints on an L4."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from data import Batches, RECORD
from models import RESNET_SWEEP, make_model


SPLIT_COUNTS = {'train': 1_000_000, 'val': 100_000}
INFERENCE_BATCHES = (1, 16, 64, 256, 1024)


def verify_manifest(manifest):
    meta = json.loads(manifest.read_text())
    counts = {
        split: sum(shard['count'] for shard in meta['shards'] if shard['split'] == split)
        for split in SPLIT_COUNTS
    }
    if counts != SPLIT_COUNTS:
        raise ValueError(f'Expected the original 1M/100k split, got {counts}')
    for shard in meta['shards']:
        path = manifest.parent / shard['path']
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != shard['count'] * RECORD.itemsize or digest.hexdigest() != shard['sha256']:
            raise ValueError(f'Unexpected shard contents: {path}')


@torch.inference_mode()
def benchmark_inference(checkpoint_path, inputs, repeats=50):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_name = checkpoint['config']['model']
    model = make_model(model_name)
    model.load_state_dict(checkpoint['model'])
    model.cuda().eval()
    # FP32 timings use full FP32 math. BF16 uses the same autocast mode as training.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pieces, stm = inputs
    rows = []
    for precision in ('fp32', 'bf16'):
        context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
        with context:
            for batch in INFERENCE_BATCHES:
                board, side = pieces[:batch], stm[:batch]
                for _ in range(10):
                    predictions = model(board, side)
                if not torch.isfinite(predictions).all():
                    raise FloatingPointError(f'Non-finite predictions for {model_name}, {precision}, batch {batch}')
                torch.cuda.synchronize()

                rates = []
                torch.cuda.reset_peak_memory_stats()
                for _ in range(3):
                    start = time.perf_counter()
                    for _ in range(repeats):
                        model(board, side)
                    torch.cuda.synchronize()
                    rates.append(batch * repeats / (time.perf_counter() - start))
                peak_mib = torch.cuda.max_memory_allocated() / 2**20

                latencies = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    model(board, side)
                    torch.cuda.synchronize()
                    latencies.append(time.perf_counter() - start)
                p95 = statistics.quantiles(latencies, n=100, method='inclusive')[94]
                rows.append({
                    'model': model_name,
                    'precision': precision,
                    'batch_size': batch,
                    'p50_ms': statistics.median(latencies) * 1000,
                    'p95_ms': p95 * 1000,
                    'positions_per_second': statistics.median(rates),
                    'peak_allocated_mib': peak_mib,
                })
    return rows


def append_csv(path, rows):
    exists = path.exists()
    with path.open('a', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 5:
        parser.error('Use at least five epochs for the full sweep.')
    if not torch.cuda.is_available() or 'NVIDIA L4' not in torch.cuda.get_device_name(0):
        parser.error('This benchmark is intended for the NVIDIA L4.')
    if not torch.cuda.is_bf16_supported():
        parser.error('This benchmark requires CUDA BF16 support.')

    manifest = args.manifest.resolve()
    verify_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'environment.json').write_text(json.dumps({
        'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'epochs': args.epochs, 'batch_size': 256, 'seed': 42,
    }, indent=2) + '\n')
    pieces, stm, _ = next(iter(Batches(manifest, 'val', max(INFERENCE_BATCHES))))
    inputs = pieces.cuda(), stm.cuda()
    train = Path(__file__).with_name('train.py')

    for name in RESNET_SWEEP:
        output = args.output / name
        print(f'\nTraining {name}', flush=True)
        subprocess.run([
            sys.executable, '-u', str(train), '--model', name,
            '--manifest', str(manifest), '--output', str(output),
            '--epochs', str(args.epochs), '--batch-size', '256', '--workers', '4',
            '--seed', '42', '--device', 'cuda', '--amp', 'bf16',
        ], check=True)
        with (output / 'metrics.csv').open(newline='') as source:
            epochs = list(csv.DictReader(source))
        if len(epochs) != args.epochs or any(
            int(row['train_positions']) != SPLIT_COUNTS['train']
            or int(row['val_positions']) != SPLIT_COUNTS['val'] for row in epochs
        ):
            raise ValueError(f'Unexpected epoch or sample counts in {output}')
        best = min(epochs, key=lambda row: float(row['val_mae_cp']))
        summary = {
            'model': name, 'parameters': best['parameter_count'],
            'best_epoch': best['epoch'], 'best_val_mae_cp': best['val_mae_cp'],
            'final_val_mae_cp': epochs[-1]['val_mae_cp'],
            'train_seconds_total': sum(float(row['train_seconds']) for row in epochs),
            'median_train_positions_per_second': statistics.median(
                float(row['train_positions_per_second']) for row in epochs),
            'peak_train_mib': max(int(row['train_peak_allocated_bytes']) for row in epochs) / 2**20,
        }
        append_csv(args.output / 'training_summary.csv', [summary])
        print(json.dumps(summary), flush=True)
        print(f'Benchmarking inference for {name}', flush=True)
        append_csv(args.output / 'inference.csv', benchmark_inference(output / 'best.pt', inputs))


if __name__ == '__main__':
    main()
