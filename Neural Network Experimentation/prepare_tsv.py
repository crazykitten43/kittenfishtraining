#!/usr/bin/env python3
"""Convert pre-split FEN<TAB>White-cp TSV(.gz) into bounded-size KFD1 shards."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

RECORD = struct.Struct('<64BBh')


def encode(fen, cp):
    fields = fen.split()
    if len(fields) != 6 or fields[1] not in ('w', 'b') or not math.isfinite(cp):
        raise ValueError('Invalid FEN or score')
    ranks = fields[0].split('/')
    if len(ranks) != 8:
        raise ValueError('Expected eight ranks')
    board = []
    for rank in reversed(ranks):
        squares = []
        for char in rank:
            if char in '12345678':
                squares.extend([0] * int(char))
            elif char in 'PNBRQKpnbrqk':
                squares.append(' PNBRQKpnbrqk'.index(char))
            else:
                raise ValueError('Invalid piece')
        if len(squares) != 8:
            raise ValueError('Invalid rank width')
        board.extend(squares)
    if board.count(6) != 1 or board.count(12) != 1 or sum(x != 0 for x in board) > 32:
        raise ValueError('Invalid piece count')
    if any(x in (1, 7) for x in board[:8] + board[56:]):
        raise ValueError('Pawn on promotion rank')
    return RECORD.pack(*board, fields[1] == 'b', round(max(-2000, min(2000, cp))))


def convert(train, val, output, shard_size):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for split, source in [('train', train), ('val', val)]:
        opener = gzip.open if str(source).endswith('.gz') else open
        count, handle = 0, None
        try:
            with opener(source, 'rt') as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip() or line.startswith('#'):
                        continue
                    try:
                        fen, score = line.rstrip('\n').split('\t')
                        record = encode(fen, float(score))
                    except ValueError as exc:
                        raise ValueError(f'{source}:{number}: {exc}') from exc
                    if count % shard_size == 0:
                        if handle:
                            handle.close()
                            entries[-1]['sha256'] = digest.hexdigest()
                        name = f'{split}-{count // shard_size:06d}.kfd'
                        handle = (output / name).open('wb')
                        entries.append(dict(path=name, split=split, count=0))
                        digest = hashlib.sha256()
                    handle.write(record)
                    digest.update(record)
                    entries[-1]['count'] += 1
                    count += 1
        finally:
            if handle:
                handle.close()
        if not count:
            raise ValueError(f'Empty {split} split')
        entries[-1]['sha256'] = digest.hexdigest()
    manifest = dict(format='KFD1', record_bytes=67, perspective='white', shards=entries,
                    counts={s: sum(e['count'] for e in entries if e['split'] == s) for s in ('train', 'val')})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--val', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input-perspective', choices=['white'], required=True)
    p.add_argument('--shard-size', type=int, default=1_000_000)
    a = p.parse_args()
    if a.shard_size < 1:
        p.error('shard-size must be positive')
    convert(a.train, a.val, a.output, a.shard_size)
