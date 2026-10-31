"""Run: python -m unittest -v test_experiments.py"""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from prepare_tsv import encode, convert

FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class EncodingTests(unittest.TestCase):
    def test_orientation_and_score(self):
        board = struct.unpack('<64BBh', encode(FEN, 123))
        self.assertEqual(board[:8], (4, 2, 3, 5, 6, 3, 2, 4))
        self.assertEqual(board[56:64], (10, 8, 9, 11, 12, 9, 8, 10))
        self.assertEqual(board[-2:], (0, 123))
        black = struct.unpack('<64BBh', encode(FEN.replace(' w ', ' b '), 9999))
        self.assertEqual(black[-2:], (1, 2000))

    def test_invalid_inputs(self):
        for fen, cp in [(FEN, float('nan')), (FEN.replace('8/8', '9/8'), 0),
                        (FEN.replace('KBNR', 'BBNR'), 0)]:
            with self.assertRaises(ValueError):
                encode(fen, cp)

    def test_conversion_split_counts_and_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, n in [('train', 5), ('val', 3)]:
                (root / split).write_text('# perspective=white\n' + ''.join(f'{FEN}\t{i}\n' for i in range(n)))
            convert(root / 'train', root / 'val', root / 'out', 2)
            meta = json.loads((root / 'out/manifest.json').read_text())
            self.assertEqual(meta['counts'], {'train': 5, 'val': 3})
            for entry in meta['shards']:
                self.assertEqual((root / 'out' / entry['path']).stat().st_size, entry['count'] * 67)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'PyTorch/NumPy required')
class ModelTests(unittest.TestCase):
    def test_both_models_forward_backward(self):
        import torch
        from models import make_model
        torch.set_num_threads(1)
        for name, minimum, maximum in [('resnet', 3_000_000, 4_000_000), ('transformer', 6_000_000, 7_000_000)]:
            model = make_model(name)
            self.assertTrue(minimum < sum(p.numel() for p in model.parameters()) < maximum)
            x = torch.zeros(2, 64, dtype=torch.uint8)
            y = model(x, torch.tensor([0, 1]))
            self.assertEqual(y.shape, (2,))
            y.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            model.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x, torch.tensor([0, 1])), model(x, torch.tensor([0, 1]))))

    def test_loader_covers_every_record(self):
        from data import Batches
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val'):
                (root / split).write_text(''.join(f'{FEN}\t{i}\n' for i in range(11)))
            convert(root / 'train', root / 'val', root / 'out', 4)
            for workers in (0, 2):
                loader = DataLoader(Batches(root / 'out/manifest.json', 'train', 3, chunk_size=5), batch_size=None, num_workers=workers)
                scores = [int(cp) for _, _, batch in loader for cp in batch]
                self.assertEqual(sorted(scores), list(range(11)))


if __name__ == '__main__':
    unittest.main()
