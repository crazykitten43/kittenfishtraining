#!/usr/bin/env python3
"""Mini-batch CUDA/CPU NNUE trainer for KFD1. Exports existing KFF1 format."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

F, H = 16 * 768, 864
DTYPE = np.dtype([('board', 'u1', (64,)), ('stm', 'u1'), ('cp', '<i2')])


def manifest(path, verify=True):
    path = Path(path)
    raw = path.read_bytes()
    m = json.loads(raw)
    if m['format'] != 'KFD1' or m['record_bytes'] != 67 or m['perspective'] != 'white':
        raise ValueError('unsupported dataset format')
    counts = {'train': 0, 'val': 0}
    seen = set()
    for s in m['shards']:
        name = s['path']
        if Path(name).name != name or name in seen or s['split'] not in counts or s['count'] < 1:
            raise ValueError('invalid shard entry')
        seen.add(name)
        file = path.parent / name
        if file.stat().st_size != s['count'] * 67:
            raise ValueError(f'truncated shard: {file}')
        if verify:
            h = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != s['sha256']:
                raise ValueError(f'checksum mismatch: {file}')
        counts[s['split']] += s['count']
    if counts != m['counts'] or not all(counts.values()):
        raise ValueError('invalid split counts')
    return m, hashlib.sha256(raw).hexdigest()


def features(rows):
    b = rows['board'].astype(np.int64)
    if np.any(b > 12) or np.any((b == 6).sum(1) != 1) or np.any((b == 12).sum(1) != 1) or np.any(rows['stm'] > 1):
        raise ValueError('invalid packed board')
    result = []
    for view in (0, 1):
        king = (b == (12 if view else 6)).argmax(1) ^ (56 if view else 0)
        bucket = (king // 16) * 4 + (king % 8) // 2
        pc = np.maximum(b - 1, 0)
        square = np.arange(64) ^ (56 if view else 0)
        f = bucket[:, None] * 768 + (((pc // 6) ^ view) * 6 + pc % 6) * 64 + square
        f[b == 0] = F  # zero padding embedding, excluded from sums
        result.append(torch.from_numpy(f))
    target = rows['cp'].astype(np.float32).clip(-2000, 2000) / 400
    target *= 1 - 2 * rows['stm'].astype(np.float32)
    return (*result, torch.from_numpy(rows['stm'].astype(np.int64)), torch.from_numpy(target))


class Batches(IterableDataset):
    def __init__(self, path, m, split, batch_size, seed, epoch, start=0):
        super().__init__()
        self.root, self.m, self.split = Path(path).parent, m, split
        self.batch_size, self.seed, self.epoch, self.start = batch_size, seed, epoch, start

    def __iter__(self):
        # Workers stride over a deterministic batch schedule, independent of worker count.
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        shards = [s for s in self.m['shards'] if s['split'] == self.split]
        rng = random.Random(self.seed + self.epoch)
        if self.split == 'train':
            rng.shuffle(shards)
        index = 0
        for s in shards:
            blocks = list(range(math.ceil(s['count'] / self.batch_size)))
            if self.split == 'train':
                rng.shuffle(blocks)
            data = None
            try:
                for block in blocks:
                    current = index
                    index += 1
                    if current < self.start or (current - self.start) % workers != wid:
                        continue
                    if data is None:
                        data = np.memmap(self.root / s['path'], dtype=DTYPE, mode='r')
                    rows = data[block*self.batch_size:(block+1)*self.batch_size].copy()
                    if self.split == 'train':
                        np.random.default_rng(self.seed + self.epoch * 1000000007 + current).shuffle(rows)
                    yield features(rows)
            finally:
                if data is not None:
                    data._mmap.close()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # EmbeddingBag avoids materializing [batch,64,864] activations.
        self.embedding = nn.EmbeddingBag(F + 1, H, mode='sum', padding_idx=F)
        self.bias = nn.Parameter(torch.zeros(H))
        self.output = nn.Linear(2 * H, 1)
        nn.init.normal_(self.embedding.weight, std=.015)
        with torch.no_grad():
            self.embedding.weight[F].zero_()
        nn.init.normal_(self.output.weight, std=.015)
        nn.init.zeros_(self.output.bias)

    def forward(self, white, black, stm):
        a = (self.embedding(white) + self.bias).clamp(0, 1)
        b = (self.embedding(black) + self.bias).clamp(0, 1)
        side = stm[:, None].bool()
        return self.output(torch.cat((torch.where(side, b, a), torch.where(side, a, b)), 1)).squeeze(1)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def export(net, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as out:
        out.write(b'KFF1')
        for tensor in (net.embedding.weight[:F], net.bias, net.output.weight, net.output.bias):
            out.write(tensor.detach().cpu().numpy().astype('<f4').tobytes())
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def loader(a, m, split, epoch, start=0):
    return DataLoader(Batches(a.manifest, m, split, a.batch_size, a.seed, epoch, start),
                      batch_size=None, num_workers=a.workers, pin_memory=a.device.startswith('cuda'),
                      **({'prefetch_factor': 2, 'multiprocessing_context': 'spawn'} if a.workers else {}))


@torch.no_grad()
def validate(net, batches, device):
    total, count, absolute = 0., 0, 0.
    net.eval()
    for batch in batches:
        w, b, stm, target = [x.to(device, non_blocking=True) for x in batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            prediction = net(w, b, stm)
        error = (prediction.float() - target) * 400
        total += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += target.numel()
    if not count:
        raise ValueError('empty validation')
    return total / count, absolute / count


def train(a):
    if min(a.batch_size, a.epochs, a.checkpoint_every, a.cpu_threads) < 1 or a.workers < 0 or a.max_steps < 0 or not math.isfinite(a.lr) or a.lr <= 0:
        raise ValueError('invalid training parameters')
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.cpu_threads)
    m, identity = manifest(a.manifest)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install CUDA PyTorch and NVIDIA driver, or use --device cpu')
    net = Net().to(device)
    parameter_count = sum(p.numel() for p in net.parameters())
    print(f'NNUE: {parameter_count} stored parameters, {parameter_count-H} effective excluding padding; CUDA BF16 autocast', flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
    epoch, start, step, best = 0, 0, 0, float('inf')
    config = dict(dataset=identity, batch_size=a.batch_size, seed=a.seed, lr=a.lr)
    if a.resume:
        # Only load checkpoints created by this script from a trusted path.
        state = torch.load(a.resume, map_location=device, weights_only=False)
        if state['config'] != config:
            raise ValueError('checkpoint dataset/batch/seed/lr mismatch')
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch, start, step, best = (state[k] for k in ('epoch', 'next_batch', 'step', 'best'))
    prefix = Path(a.output)
    def checkpoint(e, next_batch):
        atomic_save(dict(version=1, config=config, model=net.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=e, next_batch=next_batch, step=step, best=best), str(prefix) + '.last.pt')
    if a.max_steps and step >= a.max_steps:
        export(net, str(prefix) + '.last.kff')
        return
    for e in range(epoch, a.epochs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started, seen = time.monotonic(), 0
        net.train()
        for i, batch in enumerate(loader(a, m, 'train', e, start), start):
            w, b, stm, target = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                prediction = net(w, b, stm)
                loss = (prediction.float() - target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
            with torch.no_grad():
                for parameter in net.parameters():
                    parameter.clamp_(-8, 8)
            step += 1
            seen += target.numel()
            if step % 100 == 0:
                print(f'step={step} positions_per_second={seen/(time.monotonic()-started):.1f} loss={loss.item():.6f}', flush=True)
            if step % a.checkpoint_every == 0 or (a.max_steps and step >= a.max_steps):
                checkpoint(e, i + 1)
                print(f'epoch={e} next_batch={i+1} step={step} loss={loss.item():.6f}', flush=True)
            if a.max_steps and step >= a.max_steps:
                export(net, str(prefix) + '.last.kff')
                return
        if device.type == 'cuda': torch.cuda.synchronize()
        train_seconds = time.monotonic() - started
        mse, mae = validate(net, loader(a, m, 'val', e), device)
        if device.type == 'cuda': torch.cuda.synchronize()
        metrics = dict(epoch=e+1, train_positions=seen, val_positions=m['counts']['val'], train_seconds=train_seconds, train_positions_per_second=seen/train_seconds, seconds=time.monotonic()-started, val_mae_cp=mae, val_mse_cp2=mse, parameter_count=parameter_count, effective_parameter_count=parameter_count-H, peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0, peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else 0, amp='bf16', batch_size=a.batch_size, lr=a.lr, seed=a.seed, dataset=identity)
        Path(str(prefix)+'.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print(json.dumps(metrics),flush=True)
        if mse < best:
            best = mse
            export(net, str(prefix) + '.best.kff')
        checkpoint(e + 1, 0)
        export(net, str(prefix) + '.last.kff')
        print(f'epoch={e+1} step={step} val_mse_cp2={mse:.4f} val_mae_cp={mae:.4f}', flush=True)
        start = 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--max-steps', type=int, default=0, help='Total optimizer steps; 0 is unlimited')
    p.add_argument('--resume', type=Path)
    train(p.parse_args())
