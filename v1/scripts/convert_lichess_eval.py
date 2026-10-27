#!/usr/bin/env python3
"""Convert the first usable Lichess evaluation positions into White-score TSVs.

Each input line is one Lichess position record. Uses the deepest eval record,
then its first PV as recommended by database.lichess.org. Mate-only records
are skipped because this trainer's score convention is centipawns.
"""
import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path


def open_input(path):
    if str(path) == "-":
        import io
        try:
            import zstandard
        except ImportError as exc:
            raise SystemExit("Install zstandard first: python3 -m pip install zstandard") from exc
        reader = zstandard.ZstdDecompressor().stream_reader(sys.stdin.buffer)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    raw = open(path, "rb")
    magic = raw.read(4)
    raw.seek(0)
    # Zstandard skippable frames use little-endian magic 0x184D2A50..0x184D2A5F.
    # Some valid .zst exports begin with one before the data frames.
    zstd_skippable = len(magic) == 4 and 0x184D2A50 <= int.from_bytes(magic, "little") <= 0x184D2A5F
    if magic == b"\x28\xb5\x2f\xfd" or zstd_skippable:
        try:
            import zstandard
        except ImportError as exc:
            raise SystemExit("Install zstandard first: python3 -m pip install zstandard") from exc
        import io
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    if str(path).endswith(".gz"):
        return gzip.open(raw, "rt", encoding="utf-8", errors="replace")
    import io
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def valid_fen(fen):
    fields = fen.strip().split()
    if len(fields) < 4 or fields[1] not in ("w", "b"):
        return None
    # Reject invalid editor positions that the C++ trainer cannot load.
    if sum(ch in "pnbrqkPNBRQK" for ch in fields[0]) > 32:
        return None
    # Lichess stores four position fields; the C++ reader expects full FEN.
    return " ".join(fields[:4]) + " 0 1", fields[1]


def best_cp(evals, min_depth):
    candidates = [e for e in evals if isinstance(e, dict)
                  and isinstance(e.get("depth"), int)
                  and e["depth"] >= min_depth
                  and isinstance(e.get("pvs"), list) and e["pvs"]]
    if not candidates:
        return None
    chosen = max(candidates, key=lambda e: e["depth"])
    # Lichess recommends the deepest evaluation's first PV.
    score = chosen["pvs"][0].get("cp")
    return score if isinstance(score, (int, float)) and math.isfinite(score) else None


def convert(source, outdir, target, min_depth, perspective, max_lines=0):
    selected = []
    selected_fens = set()
    scanned = accepted = duplicates = bad = mate_or_shallow = 0
    with open_input(source) as stream:
        for line in stream:
            if max_lines and scanned >= max_lines:
                break
            scanned += 1
            try:
                item = json.loads(line)
                parsed = valid_fen(item["fen"])
                if not parsed:
                    bad += 1
                    continue
                full_fen, stm = parsed
                cp = best_cp(item.get("evals", []), min_depth)
                if cp is None:
                    mate_or_shallow += 1
                    continue
                cp = max(-2000, min(2000, int(round(cp))))
                if perspective == "stm" and stm == "b":
                    cp = -cp
                key = " ".join(full_fen.split()[:4])
                accepted += 1
                if key in selected_fens:
                    duplicates += 1
                    continue
                selected.append((key, full_fen, cp))
                selected_fens.add(key)
                if len(selected) >= target:
                    break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
                bad += 1
    if not selected:
        raise ValueError("No usable centipawn positions found; lower --min-depth or check input")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    outputs = {name: open(outdir / f"{name}_white.tsv", "w") for name in counts}
    try:
        for key, fen, cp in selected:
            # No game IDs exist in this database. Stable position hashing keeps
            # identical positions in one split even if source duplication occurs.
            digest = hashlib.sha256(("lichess-eval-split:" + key).encode()).digest()
            split = "val" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"
            outputs[split].write(f"{fen}\t{cp}\n")
            counts[split] += 1
    finally:
        for output in outputs.values():
            output.close()
    print(f"Read lines: {scanned:,}; usable labels: {accepted:,}; selected: {len(selected):,}")
    print(f"Skipped malformed: {bad:,}; mate-only/shallow: {mate_or_shallow:,}; repeated in sample: {duplicates:,}")
    print(f"Score input perspective: {perspective}; TSV perspective: white")
    print(f"Training: {counts['train']:,}; validation: {counts['val']:,}")
    print(f"Files: {outdir}/train_white.tsv and {outdir}/val_white.tsv")
    if len(selected) >= target:
        print(f"Stopped after reaching target of {target:,} unique positions.")
    else:
        print(f"Reached end of input before target; selected {len(selected):,} positions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Lichess JSONL or .zst export")
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--target", type=int, default=500_000,
                        help="Stop after this many unique usable positions (default: 500000)")
    parser.add_argument("--min-depth", type=int, default=20)
    parser.add_argument("--input-perspective", choices=("white", "stm"), required=True,
                        help="Explicitly set the meaning of pvs[0].cp")
    parser.add_argument("--max-lines", type=int, default=0,
                        help="Read only this many records for a quick format test; 0 reads all")
    args = parser.parse_args()
    if args.target < 1 or args.min_depth < 1 or args.max_lines < 0:
        parser.error("target and min-depth must be positive; max-lines cannot be negative")
    try:
        convert(args.source, args.outdir, args.target, args.min_depth,
                args.input_perspective, args.max_lines)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

