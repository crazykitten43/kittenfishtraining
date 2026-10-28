#!/usr/bin/env python3
"""Bounded-memory, restartable Lichess JSONL -> KFD1 packed shards."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct

from convert_lichess_eval import open_input, best_cp

RECORD = struct.Struct('<64BBh')  # a1..h8 pieces, stm, White cp; 67 bytes


def board_fen(text):
    fields = text.split()
    if len(fields) < 4 or fields[1] not in ('w', 'b'):
        raise ValueError('invalid FEN fields')
    board = []
    ranks = fields[0].split('/')
    if len(ranks) != 8:
        raise ValueError('invalid ranks')
    for rank in reversed(ranks):
        squares = []
        for c in rank:
            if c in '12345678':
                squares.extend([0] * int(c))
            elif c in 'PNBRQKpnbrqk':
                squares.append(' PNBRQKpnbrqk'.index(c))
            else:
                raise ValueError('invalid piece')
        if len(squares) != 8:
            raise ValueError('invalid rank width')
        board.extend(squares)
    if board.count(6) != 1 or board.count(12) != 1 or sum(bool(x) for x in board) > 32:
        raise ValueError('invalid piece count')
    if any(x in (1, 7) for x in board[:8] + board[56:]):
        raise ValueError('pawn on promotion rank')
    return board, int(fields[1] == 'b')


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build(source, destination, target=100_000_000, min_depth=20,
          perspective='white', shard_size=1_000_000, val_permille=10,
          commit_every=10000, max_lines=0):
    if min(target, min_depth, shard_size, commit_every) < 1 or not 1 <= val_permille <= 999 or max_lines < 0:
        raise ValueError('invalid generation limits')
    source, destination = Path(source).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    config = dict(source=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                  target=target, min_depth=min_depth, perspective=perspective,
                  shard_size=shard_size, val_permille=val_permille, format='KFD1')
    if perspective not in ('white', 'stm'):
        raise ValueError('invalid perspective')
    with (destination / '.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with contextlib.closing(sqlite3.connect(destination / 'dedup.sqlite')) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
            db.execute('PRAGMA cache_size=-65536')
            db.execute('CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, key BLOB UNIQUE NOT NULL, split TEXT NOT NULL, record BLOB NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
            saved = db.execute('SELECT value FROM state WHERE id=1').fetchone()
            state = json.loads(saved[0]) if saved else dict(config=config, scanned=0, selected=0, bad=0, filtered=0, duplicates=0, clipped=0, white=0, black=0)
            if state['config'] != config:
                raise ValueError('source/config changed; use a new output directory')
            # Source positions and inserted records commit in the SAME transaction.
            # Compressed sources replay the durable prefix on restart, without reparsing it.
            def commit():
                db.execute('INSERT OR REPLACE INTO state VALUES (1, ?)', (json.dumps(state),))
                db.commit()
                print(json.dumps(state), flush=True)
            if state['selected'] < target:
                with open_input(source) as stream:
                    for number, line in enumerate(stream, 1):
                        if number <= state['scanned']:
                            continue
                        if max_lines and number > max_lines:
                            break
                        state['scanned'] = number
                        try:
                            item = json.loads(line)
                            board, stm = board_fen(item['fen'])
                            cp = best_cp(item.get('evals', []), min_depth)
                            if cp is None:
                                state['filtered'] += 1
                            else:
                                state["clipped"] += abs(cp) > 2000
                                cp = max(-2000, min(2000, round(cp)))
                                if perspective == 'stm' and stm:
                                    cp = -cp
                                # Match network identity: rights/EP/counters are not features.
                                key = bytes(board + [stm])
                                h = hashlib.sha256(b'kittenfish-kfd1:' + key).digest()
                                split = 'val' if int.from_bytes(h[:8], 'big') % 1000 < val_permille else 'train'
                                cur = db.execute('INSERT OR IGNORE INTO samples(key,split,record) VALUES (?,?,?)',
                                                 (key, split, RECORD.pack(*board, stm, cp)))
                                state['selected' if cur.rowcount else 'duplicates'] += 1
                                if cur.rowcount:
                                    state['black' if stm else 'white'] += 1
                        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                            state['bad'] += 1
                        if number % commit_every == 0:
                            commit()
                        if state['selected'] >= target:
                            break
                commit()
            # Re-export is deterministic and bounded; partial exports never publish a manifest.
            manifest_path = destination / 'manifest.json'
            manifest_path.unlink(missing_ok=True)
            entries = []
            for split in ('train', 'val'):
                last_id = 0
                cursor = db.execute('SELECT id,record FROM samples WHERE split=? ORDER BY id', (split,))
                while True:
                    rows = cursor.fetchmany(shard_size)
                    if not rows:
                        break
                    name = f'{split}-{last_id:09d}.kfd'
                    path = destination / name
                    digest = hashlib.sha256()
                    with path.with_suffix('.tmp').open('wb') as out:
                        for _, record in rows:
                            out.write(record)
                            digest.update(record)
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(path.with_suffix('.tmp'), path)
                    entries.append(dict(path=name, split=split, count=len(rows), sha256=digest.hexdigest()))
                    last_id += len(rows)
            counts = {s: sum(e['count'] for e in entries if e['split'] == s) for s in ('train', 'val')}
            manifest = dict(format='KFD1', record_bytes=RECORD.size, perspective='white',
                            config=config, stats=state, counts=counts,
                            target_reached=state['selected'] >= target, shards=entries)
            for split in ('train', 'val'):
                listing = destination / f'{split}.list'
                tmp = listing.with_suffix('.tmp')
                tmp.write_text(''.join(e['path'] + '\n' for e in entries if e['split'] == split))
                os.replace(tmp, listing)
            atomic_json(manifest_path, manifest)
            if not all(counts.values()):
                raise ValueError(f'empty split: {counts}; generate more positions')
            return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--target', type=int, default=100_000_000)
    p.add_argument('--min-depth', type=int, default=20)
    p.add_argument('--input-perspective', choices=('white', 'stm'), required=True)
    p.add_argument('--shard-size', type=int, default=1_000_000)
    p.add_argument('--val-permille', type=int, default=10)
    p.add_argument('--commit-every', type=int, default=10000)
    p.add_argument('--max-lines', type=int, default=0)
    a = p.parse_args()
    build(a.source, a.destination, a.target, a.min_depth, a.input_perspective,
          a.shard_size, a.val_permille, a.commit_every, a.max_lines)
