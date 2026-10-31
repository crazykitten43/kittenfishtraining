"""Memory-mapped KFD1 shards, shuffled in bounded chunks, partitioned by worker."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

RECORD = np.dtype([('pieces', 'u1', (64,)), ('stm', 'u1'), ('cp', '<i2')])


class Batches(IterableDataset):
    def __init__(self, manifest, split, batch_size, seed=42, epoch=0, chunk_size=65536):
        self.manifest = Path(manifest).resolve()
        meta = json.loads(self.manifest.read_text())
        if meta['format'] != 'KFD1' or meta['perspective'] != 'white' or meta['record_bytes'] != 67:
            raise ValueError('Expected KFD1, 67-byte records, White perspective')
        self.shards = [e for e in meta['shards'] if e['split'] == split]
        if not self.shards:
            raise ValueError(f'Empty {split} split')
        self.train = split == 'train'
        self.batch_size, self.seed, self.epoch = batch_size, seed, epoch
        self.chunk_size = chunk_size

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = np.random.default_rng(self.seed + self.epoch)
        shards = list(self.shards)
        if self.train:
            rng.shuffle(shards)
        for entry in shards[wid::workers]:
            path = self.manifest.parent / entry['path']
            if path.stat().st_size != entry['count'] * RECORD.itemsize:
                raise ValueError(f'Truncated shard: {path}')
            records = np.memmap(path, mode='r', dtype=RECORD)
            chunks = list(range(0, len(records), self.chunk_size))
            if self.train:
                rng.shuffle(chunks)
            for start in chunks:
                rows = np.array(records[start:start + self.chunk_size])
                if self.train:
                    rng.shuffle(rows)
                if (rows['pieces'] > 12).any() or (rows['stm'] > 1).any():
                    raise ValueError(f'Invalid piece/side encoding: {path}')
                for offset in range(0, len(rows), self.batch_size):
                    batch = rows[offset:offset + self.batch_size]
                    yield (torch.from_numpy(batch['pieces'].copy()),
                           torch.from_numpy(batch['stm'].copy()),
                           torch.from_numpy(batch['cp'].astype(np.float32)))
