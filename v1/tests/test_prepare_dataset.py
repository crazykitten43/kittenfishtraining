import contextlib
import csv
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare", ROOT / "scripts/prepare_dataset.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ConversionTests(unittest.TestCase):
    def test_score_signs_groups_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            rows = []
            for game in range(40):
                board = ["1"] * 64
                board[0], board[game + 16] = "K", "k"
                ranks = []
                for rank in range(7, -1, -1):
                    text, empty = "", 0
                    for piece in board[rank * 8:rank * 8 + 8]:
                        if piece == "1":
                            empty += 1
                        else:
                            text += (str(empty) if empty else "") + piece
                            empty = 0
                    ranks.append(text + (str(empty) if empty else ""))
                placement = "/".join(ranks)
                for side, cp in (("w", 120), ("b", 80)):
                    rows.append({"fen": f"{placement} {side} - - 0 1",
                                 "eval_cp": cp, "game_id": str(game)})
            rows.append(dict(rows[0]))
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("fen", "eval_cp", "game_id"))
                writer.writeheader()
                writer.writerows(rows)
            for perspective in ("stm", "white"):
                destination = root / perspective
                with contextlib.redirect_stdout(io.StringIO()):
                    counts = prepare.convert(source, destination, perspective)
                self.assertEqual(sum(counts.values()), 80)
                game_sets = []
                for split in ("train", "val"):
                    records = [line.split("\t") for line in
                               (destination / f"{split}_white.tsv").read_text().splitlines()
                               if not line.startswith("#")]
                    games = set()
                    for fen, score in records:
                        fields = fen.split()
                        games.add(fields[0])
                        expected = 120 if fields[1] == "w" else (-80 if perspective == "stm" else 80)
                        self.assertEqual(float(score), expected)
                    game_sets.append(games)
                self.assertFalse(game_sets[0] & game_sets[1])

    def test_input_perspective_required(self):
        import subprocess
        result = subprocess.run(["python3", str(ROOT / "scripts/prepare_dataset.py"),
                                 "unused.csv", "unused"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"--input-perspective", result.stderr)


if __name__ == "__main__":
    unittest.main()
