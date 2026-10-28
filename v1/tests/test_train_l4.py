import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    import numpy as np
    import torch
    import train_l4 as gpu
except ImportError:
    gpu = None
from test_shards import records
from build_shards import build


@unittest.skipIf(gpu is None, 'install requirements-l4.txt for GPU pipeline tests')
class L4Tests(unittest.TestCase):
    def test_schedule_corruption_resume_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'input.jsonl'
            src.write_text('\n'.join(json.dumps(r) for r in records(80)))
            with contextlib.redirect_stdout(io.StringIO()):
                m = build(src, root / 'data', target=80, val_permille=200, shard_size=13)
            path = root / 'data/manifest.json'
            gpu.manifest(path)
            def batches(workers=0, start=0):
                d = gpu.Batches(path, m, 'train', 16, 42, 0, start)
                return list(gpu.DataLoader(d, batch_size=None, num_workers=workers))
            baseline = batches()
            multi = batches(2)
            self.assertEqual(len(baseline), len(multi))
            for expected, actual in zip(baseline, multi):
                for x, y in zip(expected, actual):
                    self.assertTrue(torch.equal(x, y))
            for expected, actual in zip(baseline[2:], batches(2, 2)):
                for x, y in zip(expected, actual):
                    self.assertTrue(torch.equal(x, y))
            def args(prefix, max_steps, resume=None):
                return argparse.Namespace(manifest=path, output=root / prefix, device='cpu', batch_size=16,
                                          workers=0, cpu_threads=1, epochs=1, lr=.0001, seed=42,
                                          checkpoint_every=1, max_steps=max_steps, resume=resume)
            with contextlib.redirect_stdout(io.StringIO()):
                gpu.train(args('full', 3))
                gpu.train(args('resume', 1))
                gpu.train(args('resume', 3, root / 'resume.last.pt'))
                gpu.train(args('new-directory/epoch', 0))
            epoch_state = torch.load(root / 'new-directory/epoch.last.pt', weights_only=False)
            self.assertEqual(epoch_state['epoch'], 1)
            self.assertEqual(epoch_state['next_batch'], 0)
            self.assertTrue((root / 'new-directory/epoch.best.kff').exists())
            full = torch.load(root / 'full.last.pt', weights_only=False)
            resumed = torch.load(root / 'resume.last.pt', weights_only=False)
            self.assertEqual(full['next_batch'], resumed['next_batch'])
            for name in full['model']:
                self.assertTrue(torch.equal(full['model'][name], resumed['model'][name]), name)
            net = gpu.Net()
            net.load_state_dict(full['model'])
            # Export bytes and C++ evaluation must agree for BOTH perspectives.
            gpu.export(net, root / 'net.kff')
            self.assertEqual((root / 'net.kff').stat().st_size, 4 + (gpu.F*gpu.H + 3*gpu.H + 1)*4)
            exe = ROOT / 'build/kfnnue'
            if exe.exists():
                tsv = root / 'val.tsv'
                test_rows = list(records(20))
                tsv.write_text(''.join(r['fen'] + ' 0 1\t0\n' for r in test_rows))
                from build_shards import board_fen, RECORD
                raw = b''.join(RECORD.pack(*board_fen(r['fen'])[0], board_fen(r['fen'])[1], 0) for r in test_rows)
                batch = gpu.features(np.frombuffer(raw, dtype=gpu.DTYPE))
                with torch.no_grad():
                    prediction = net(*batch[:3]) * 400
                out = subprocess.check_output([str(exe), 'diagnose', str(root / 'net.kff'), str(tsv)], text=True)
                line = next(l for l in out.splitlines() if l.startswith('float model'))
                mse = float(line.split('MSE=')[1].split()[0])
                self.assertAlmostEqual(mse, prediction.square().mean().item(), delta=max(.01, mse * .0001))
                # Streaming reader accepts full shards and rejects trailing bytes.
                for split in ('train', 'val'):
                    (path.parent / f'{split}.list').write_text('\n'.join(s['path'] for s in m['shards'] if s['split'] == split) + '\n')
                subprocess.run([str(exe), 'stream-train', str(path.parent / 'train.list'), str(path.parent / 'val.list'),
                                str(root / 'cpp'), '1', '.0001', str(root / 'net.kff')], check=True, capture_output=True)
                shard = path.parent / m['shards'][0]['path']
                with shard.open('ab') as f:
                    f.write(b'x')
                result = subprocess.run([str(exe), 'stream-train', str(path.parent / 'train.list'), str(path.parent / 'val.list'),
                                         str(root / 'bad'), '1', '.0001'], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                with self.assertRaisesRegex(ValueError, 'truncated'):
                    gpu.manifest(path)
            # Same-size corruption is detected by SHA-256.
            shard = path.parent / m['shards'][-1]['path']
            raw = bytearray(shard.read_bytes())
            raw[-1] ^= 1
            shard.write_bytes(raw)
            # Restore earlier trailing-byte corruption, if C++ test ran.
            first = path.parent / m['shards'][0]['path']
            first.write_bytes(first.read_bytes()[:m['shards'][0]['count']*67])
            with self.assertRaisesRegex(ValueError, 'checksum'):
                gpu.manifest(path)


if __name__ == '__main__':
    unittest.main()
