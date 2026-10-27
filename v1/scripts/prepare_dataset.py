#!/usr/bin/env python3
"""Convert labelled CSV(.gz) to White-perspective TSV with game-grouped splits."""
import argparse
import csv
import gzip
import hashlib
import math
from pathlib import Path


def convert(source, destination, perspective):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    counts = {"train": 0, "val": 0}
    duplicates = 0
    opener = gzip.open if str(source).endswith(".gz") else open
    # Distinct filenames keep the previously incorrect TSVs available for audit.
    with opener(source, "rt", newline="") as src, \
         open(destination / "train_white.tsv", "w") as train, \
         open(destination / "val_white.tsv", "w") as val:
        reader = csv.DictReader(src)
        missing = {"fen", "eval_cp", "game_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        outputs = {"train": train, "val": val}
        for stream in outputs.values():
            stream.write(f"# perspective=white; source_perspective={perspective}\n")
        for line, row in enumerate(reader, 2):
            fen = row["fen"].strip()
            fields = fen.split()
            game = row["game_id"].strip()
            score = float(row["eval_cp"])
            if len(fields) != 6 or fields[1] not in ("w", "b") or not game or not math.isfinite(score):
                raise ValueError(f"Invalid row {line}")
            key = " ".join(fields[:4])
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            if perspective == "stm" and fields[1] == "b":
                score = -score
            digest = hashlib.sha256(("kittenfish-split-42:" + game).encode()).digest()
            split = "val" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"
            outputs[split].write(f"{fen}\t{score}\n")
            counts[split] += 1
    if not all(counts.values()):
        raise ValueError(f"Empty split: {counts}; add more games")
    print(f"Source: {perspective}; output: WHITE perspective")
    print(f"Training: {counts['train']:,}; validation: {counts['val']:,}; duplicates: {duplicates:,}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--input-perspective", choices=("white", "stm"), required=True,
                        help="Required explicitly. KittenFish positions_100k.csv.gz uses stm.")
    args = parser.parse_args()
    convert(args.source, args.destination, args.input_perspective)
