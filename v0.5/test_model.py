"""Lightweight shape and encoding tests for KittenFish SE-ResNet."""
import unittest
import numpy as np
import torch
import chess
from encoding import encode_fen
from model import KittenFishResNet

class ResNetTests(unittest.TestCase):
    def test_start_position_encoding(self):
        x = encode_fen(chess.STARTING_FEN)
        self.assertEqual(x.shape, (18, 8, 8))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(float(x[12, 0, 0]), 1.0)
        self.assertAlmostEqual(float(x[:12].sum()), 32.0)

    def test_forward_batch_shape(self):
        model = KittenFishResNet(channels=16, blocks=2)
        output = model(torch.zeros(3, 18, 8, 8))
        self.assertEqual(tuple(output.shape), (3,))
        self.assertTrue(torch.isfinite(output).all())

    def test_reject_wrong_input_shape(self):
        model = KittenFishResNet(channels=16, blocks=1)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 12, 8, 8))

if __name__ == "__main__":
    unittest.main()
