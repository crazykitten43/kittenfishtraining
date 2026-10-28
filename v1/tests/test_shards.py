import contextlib
import gzip
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_shards import build, board_fen, RECORD
from convert_lichess_eval import best_cp


def records(n=200):
    for i in range(n):
        board = [0] * 64
        board[0], board[63] = 6, 12
        board[8 + i % 48] = 2 + (i // 48) % 4
        ranks = []
        for r in range(7, -1, -1):
            rank, empty = '', 0
            for pc in board[r*8:r*8+8]:
                if not pc:
                    empty += 1
                else:
                    rank += (str(empty) if empty else '') + ' PNBRQKpnbrqk'[pc]
                    empty = 0
            ranks.append(rank + (str(empty) if empty else ''))
        yield dict(fen='/'.join(ranks) + (' b' if i % 2 else ' w') + ' - -',
                   evals=[dict(depth=22, pvs=[dict(cp=i - 100)])])


class ShardTests(unittest.TestCase):
    def build(self, source, out, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return build(source, out, target=200, shard_size=17, val_permille=200, commit_every=7, **kw)

    def test_resume_dedup_sign_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = list(records())
            rows += [rows[0], dict(rows[0], fen=rows[0]['fen'] + ' 10 20'), {'fen': 'bad'}, {'fen': rows[0]['fen'], 'evals': [{'depth': 50, 'pvs': [None]}]}]
            source = root / 'source.gz'
            with gzip.open(source, 'wt') as out:
                out.write('\n'.join(json.dumps(r) for r in rows))
            self.build(source, root / 'resume', perspective='stm', max_lines=80)
            actual = self.build(source, root / 'resume', perspective='stm')
            expected = self.build(source, root / 'fresh', perspective='stm')
            self.assertEqual(actual, expected)
            self.assertEqual(actual['stats']['duplicates'], 10)  # 192 unique network inputs
            self.assertEqual(actual['stats']['bad'], 1)
            self.assertEqual(actual['stats']['filtered'], 1)
            keys = {'train': set(), 'val': set()}
            for s in actual['shards']:
                data = (root / 'resume' / s['path']).read_bytes()
                self.assertEqual(data, (root / 'fresh' / s['path']).read_bytes())
                self.assertLessEqual(len(data), 17 * 67)
                for row in RECORD.iter_unpack(data):
                    key = bytes(row[:65])
                    keys[s['split']].add(key)
                    # first occurrence wins, including STM label conversion
                    original = next(r for r in records() if bytes(board_fen(r['fen'])[0] + [board_fen(r['fen'])[1]]) == key)
                    cp = original['evals'][0]['pvs'][0]['cp']
                    self.assertEqual(row[-1], -cp if row[64] else cp)
            self.assertFalse(keys['train'] & keys['val'])
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.build(source, root / 'resume', perspective='white')

    def test_invalid_board_and_labels(self):
        for fen in ('8/8/8/8/8/8/8/8 w - -', '7k/8/8/8/8/8/8/K6P w - -', '7k/8/8/8/8/8/8/K8 w - -'):
            with self.assertRaises(ValueError):
                board_fen(fen)
        self.assertIsNone(best_cp([dict(depth=20, pvs=[None])], 20))
        self.assertIsNone(best_cp([dict(depth=20, pvs=[dict(cp=True)])], 20))

    def test_uncommitted_records_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.write_text('\n'.join(json.dumps(r) for r in records()))
            self.build(source, root / 'out', max_lines=80)
            db = sqlite3.connect(root / 'out/dedup.sqlite')
            db.execute('DELETE FROM samples')
            db.close()  # simulates a crash before transaction commit
            m = self.build(source, root / 'out')
            self.assertEqual(sum(m['counts'].values()), 192)

    def test_zstd_input(self):
        try:
            import zstandard
        except ImportError:
            self.skipTest('zstandard not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.zst'
            source.write_bytes(zstandard.ZstdCompressor().compress('\n'.join(json.dumps(r) for r in records()).encode()))
            m = self.build(source, root / 'out')
            self.assertEqual(sum(m['counts'].values()), 192)


if __name__ == '__main__':
    unittest.main()
