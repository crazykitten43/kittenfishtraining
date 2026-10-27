#!/usr/bin/env python3
"""
Build a stratified human-game chess-position dataset labelled by Stockfish.

Example:
  python make_chess_dataset.py --target 5000 --start-month 2025-01 --months 3 \
      --stockfish /usr/games/stockfish --output dataset_5k.csv.gz

Dependencies:
  pip install python-chess requests zstandard

Notes:
- Streams each monthly Lichess rated-game PGN archive; it does not save the full PGN.
- Each selected game contributes at most 3 candidate positions.
- Rating band = floor(mean(White rating, Black rating) / 100) * 100.
- Stages are based on the move whose position is being evaluated:
    opening: full moves 1-10; middlegame: 11-35; late: move 36 onward.
  Positions are taken BEFORE the move is played. The initial board is move 1.
- The final position quotas are stratified across 15 rating bands and 3 stages.
- Stockfish labels are centipawns from the side-to-move perspective.
- Output includes game_id so future train/validation/test splits can be grouped by game.
- workers × threads should normally not exceed available vCPUs. Hash memory is
  per worker: workers × hash-mb MiB, plus Python/OS overhead.
- Candidates and deduplication are disk-backed; only a bounded evaluation window
  is in RAM. Reservoir selection stops as soon as all quotas are filled.
- A small checkpoint is written after each month for auditing. The disk-backed reservoir
  is not resumed, so an interrupted run must rescan from the start month.
  Source PGNs are streamed and never stored.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import sqlite3
import tempfile
import threading
import signal
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, TextIO

import chess
import chess.engine
import chess.pgn
import requests
import zstandard


LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
RATING_LOWER = 1000
RATING_UPPER = 2500  # exclusive
RATING_STEP = 100
STAGES = ("opening", "middlegame", "late")
STAGE_WEIGHTS = (1, 3, 1)
USER_AGENT = "human-chess-position-dataset/1.0 (streaming PGN sampler)"


def parse_month(value: str) -> date:
    try:
        year, month = map(int, value.split("-"))
        return date(year, month, 1)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Month must be YYYY-MM, e.g. 2025-01") from exc


def month_range(start: date, count: int) -> list[str]:
    result = []
    year, month = start.year, start.month
    for _ in range(count):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return result


def stage_for_ply(ply: int) -> str:
    # ply=0 is the initial position, before White's first move.
    full_move = ply // 2 + 1
    if full_move <= 10:
        return "opening"
    if full_move <= 35:
        return "middlegame"
    return "late"


def rating_band(white: int, black: int) -> int | None:
    mean_rating = (white + black) / 2.0
    lower = int(mean_rating // RATING_STEP) * RATING_STEP
    if RATING_LOWER <= lower < RATING_UPPER:
        return lower
    return None


def position_key(board: chess.Board) -> str:
    # Position identity excludes move counters, but includes side to move,
    # castling rights and en-passant state. FEN's first four fields capture these.
    return " ".join(board.fen().split()[:4])


def hash_position(key: str) -> str:
    return hashlib.blake2b(key.encode("ascii"), digest_size=8).hexdigest()


def quota_map(target: int) -> dict[tuple[int, str], int]:
    """Allocate nearly equal totals per rating band, then 20/60/20 by stage."""
    bands = list(range(RATING_LOWER, RATING_UPPER, RATING_STEP))
    base_band, extra_bands = divmod(target, len(bands))
    result: dict[tuple[int, str], int] = {}

    for i, band in enumerate(bands):
        band_total = base_band + (1 if i < extra_bands else 0)
        # Largest-remainder apportionment using weights 1:3:1.
        raw = [band_total / 5, band_total * 3 / 5, band_total / 5]
        allocated = [math.floor(x) for x in raw]
        remainder = band_total - sum(allocated)
        order = sorted(range(3), key=lambda j: (raw[j] - allocated[j], -j), reverse=True)
        for j in order[:remainder]:
            allocated[j] += 1
        for stage, amount in zip(STAGES, allocated):
            result[(band, stage)] = amount

    return result


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Could not read checkpoint %s: %s", path, exc)
        return None


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=5000, help="Number of labelled positions (default: 5000)")
    parser.add_argument("--start-month", type=parse_month, default=parse_month("2025-01"),
                        help="First Lichess archive month, YYYY-MM (default: 2025-01)")
    parser.add_argument("--months", type=int, default=12,
                        help="Maximum consecutive months to scan; stops early once all quotas are filled (default: 12)")
    parser.add_argument("--stockfish", default="/usr/games/stockfish",
                        help="Path to Stockfish executable (default: /usr/games/stockfish)")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth per position (default: 12)")
    parser.add_argument("--nodes", type=int, default=0,
                        help="If nonzero, use this node limit instead of --depth")
    parser.add_argument("--workers", type=int, default=1, help="Independent Stockfish processes (default: 1); workers x threads should not exceed vCPUs; total hash is workers x hash-mb MiB")
    parser.add_argument("--threads", type=int, default=4, help="Stockfish threads PER WORKER (default: 4)")
    parser.add_argument("--hash-mb", type=int, default=1024, help="Stockfish hash MiB PER WORKER (default: 1024)")
    parser.add_argument("--max-cp", type=int, default=2000,
                        help="Clamp labels to +/- this many cp; set 0 to disable (default: 2000)")
    parser.add_argument("--output", default="chess_positions.csv.gz", help="Output CSV.GZ path")
    parser.add_argument("--work-dir", default="chess_dataset_work", help="Directory for checkpoint and temp metadata")
    parser.add_argument("--seed", type=int, default=43, help="Random seed (default: 43)")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP read/connect and individual engine command timeout in seconds (default: 120)")
    parser.add_argument("--max-games-per-month", type=int, default=0,
                        help="Optional test limit; 0 means scan the whole month")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any previous checkpoint and start fresh")
    parser.add_argument("--keep-checkpoint", action="store_true", help="Keep checkpoint after successful completion")
    parser.add_argument("--keep-partial", action="store_true",
                        help="Export quota-short selections to OUTPUT.partial.csv.gz (never mark complete)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Retain temporary candidate database, checkpoint and unfinished output")
    return parser


def get_ratings(headers: dict[str, str]) -> tuple[int, int] | None:
    try:
        white = int(headers.get("WhiteElo", ""))
        black = int(headers.get("BlackElo", ""))
        if not (0 < white < 4000 and 0 < black < 4000):
            return None
        return white, black
    except (TypeError, ValueError):
        return None


def select_game_candidates(
    game: chess.pgn.Game,
    white_rating: int,
    black_rating: int,
    game_id: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    band = rating_band(white_rating, black_rating)
    if band is None:
        return []

    # Collect positions before each mainline move, grouped by move-number stage.
    by_stage: dict[str, list[str]] = {s: [] for s in STAGES}
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        if board.is_game_over(claim_draw=False) or ply >= 300:
            break
        by_stage[stage_for_ply(ply)].append(board.fen())
        board.push(move)

    available = [s for s in STAGES if by_stage[s]]
    if not available:
        return []

    chosen: list[tuple[str, str]] = []
    # Weighted sampling by stage, independent of how long each stage lasts.
    # This keeps the candidate pool close to 20/60/20 while never exceeding 3/game.
    weights = dict(zip(STAGES, STAGE_WEIGHTS))
    for _ in range(3):
        eligible = [s for s in available if by_stage[s]]
        if not eligible:
            break
        total = sum(weights[s] for s in eligible)
        pick = rng.uniform(0, total)
        cumulative = 0.0
        chosen_stage = eligible[-1]
        for s in eligible:
            cumulative += weights[s]
            if pick <= cumulative:
                chosen_stage = s
                break
        fen = rng.choice(by_stage[chosen_stage])
        by_stage[chosen_stage].remove(fen)
        chosen.append((chosen_stage, fen))

    return [
        {
            "fen": fen,
            "position_key": " ".join(fen.split()[:4]),
            "rating_band": band,
            "mean_rating": (white_rating + black_rating) / 2.0,
            "stage": stage,
            "game_id": game_id,
        }
        for stage, fen in chosen
    ]


def scan_month(
    month: str,
    quotas: dict[tuple[int, str], int],
    reservoirs: dict[tuple[int, str], list[dict[str, Any]]],
    seen_hashes: set[str],
    candidate_counts: Counter,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[int, int]:
    url = LICHESS_URL.format(month=month)
    logging.info("Streaming %s", url)
    games_seen = 0
    eligible_games = 0
    started = time.monotonic()

    headers = {"User-Agent": USER_AGENT}
    with requests.get(url, stream=True, headers=headers, timeout=args.timeout) as response:
        if response.status_code == 404:
            logging.warning("Archive not found for %s; skipping", month)
            return 0, 0
        response.raise_for_status()
        response.raw.decode_content = True
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(response.raw, read_size=1 << 20) as reader:
            text_stream = __import__("io").TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="")
            while True:
                if args.max_games_per_month and games_seen >= args.max_games_per_month:
                    break
                try:
                    game = chess.pgn.read_game(text_stream)
                except (ValueError, UnicodeError) as exc:
                    logging.warning("PGN parse issue in %s after %d games: %s", month, games_seen, exc)
                    continue
                if game is None:
                    break
                games_seen += 1
                if game.errors or game.headers.get("WhiteTitle") == "BOT" or game.headers.get("BlackTitle") == "BOT":
                    continue
                if game.headers.get("Variant", "Standard") != "Standard" or game.headers.get("SetUp") == "1":
                    continue
                ratings = get_ratings(game.headers)
                if ratings is None:
                    continue
                band = rating_band(*ratings)
                if band is None:
                    continue
                eligible_games += 1
                game_id = f"{month}:{games_seen}"
                candidates = select_game_candidates(game, ratings[0], ratings[1], game_id, rng)

                for candidate in candidates:
                    key = (candidate["rating_band"], candidate["stage"])
                    pkey_hash = candidate["position_key"]
                    # Global deduplication across all scanned games/months.
                    if pkey_hash in seen_hashes:
                        continue
                    seen_hashes.add(pkey_hash)

                    cap = quotas[key]
                    if cap <= 0:
                        continue
                    candidate_counts[key] += 1
                    seen = candidate_counts[key]
                    reservoir = reservoirs[key]
                    if len(reservoir) < cap:
                        reservoir.append(candidate)
                    else:
                        # Reservoir sampling over unique candidate positions.
                        j = rng.randrange(seen)
                        if j < cap:
                            reservoir[j] = candidate

                if all(len(reservoirs[k]) >= quotas[k] for k in quotas):
                    break

                if games_seen % 100_000 == 0:
                    filled = sum(len(v) for v in reservoirs.values())
                    logging.info("%s: %s games scanned, %s positions retained, %.1f min",
                                 month, f"{games_seen:,}", f"{filled:,}", (time.monotonic() - started) / 60)

    logging.info("Finished %s: %s games scanned, %s eligible, %.1f min",
                 month, f"{games_seen:,}", f"{eligible_games:,}", (time.monotonic() - started) / 60)
    return games_seen, eligible_games


class DiskSeen:
    """Exact canonical-position deduplication, with a bounded SQLite page cache."""
    def __init__(self, db):
        self.db = db
        db.execute("CREATE TABLE seen (key TEXT PRIMARY KEY) WITHOUT ROWID")

    def __contains__(self, key):
        return self.db.execute("SELECT 1 FROM seen WHERE key=?", (key,)).fetchone() is not None

    def add(self, key):
        self.db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (key,))


class DiskBucket:
    def __init__(self, db, bucket):
        self.db, self.bucket, self.size = db, bucket, 0

    def __len__(self):
        return self.size

    def append(self, row):
        self[self.size] = row
        self.size += 1

    def __setitem__(self, index, row):
        self.db.execute("INSERT OR REPLACE INTO candidates VALUES (?, ?, ?)",
                        (self.bucket, index, json.dumps(row)))

    def __iter__(self):
        for (row,) in self.db.execute("SELECT row FROM candidates WHERE bucket=? ORDER BY slot", (self.bucket,)):
            yield json.loads(row)


def evaluate_and_write(rows, args, output_path):
    """Bounded jobs; one long-lived UCI process per executor thread; sole CSV writer.

    Any failed/timeout evaluation aborts publication. No silent missing labels.
    Startup uses SimpleEngine timeout; a watchdog bounds configuration/analysis.
    """
    local = threading.local()
    engines = []
    lock = threading.Lock()

    def evaluate(row):
        if not hasattr(local, "engine"):
            engine = chess.engine.SimpleEngine.popen_uci(args.stockfish, timeout=args.timeout)
            with lock:
                engines.append(engine)
            local.engine = engine

        engine = local.engine
        expired = threading.Event()
        def abort():
            expired.set()
            engine.close()
        watchdog = threading.Timer(args.timeout, abort)
        watchdog.daemon = True
        watchdog.start()
        try:
            if not getattr(local, "configured", False):
                engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
                local.configured = True
            board = chess.Board(row["fen"])
            limit = chess.engine.Limit(nodes=args.nodes) if args.nodes else chess.engine.Limit(depth=args.depth)
            info = engine.analyse(board, limit)
            if expired.is_set():
                raise TimeoutError("Stockfish evaluation timed out")
        except Exception as exc:
            if expired.is_set():
                raise TimeoutError("Stockfish evaluation timed out") from exc
            raise
        finally:
            watchdog.cancel()
            watchdog.join()
        score = info["score"].pov(board.turn).score(mate_score=10000)
        if score is None:
            raise RuntimeError("Engine returned no valid evaluation")
        if args.max_cp:
            score = max(-args.max_cp, min(args.max_cp, score))
        return {"fen": row["fen"], "eval_cp": int(score),
                "rating_band": row["rating_band"], "mean_rating": f'{row["mean_rating"]:.1f}',
                "stage": row["stage"], "game_id": row["game_id"]}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(output_path.name + ".tmp")
    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = set()
    succeeded = False
    count = 0
    try:
        with gzip.open(temp_output, "wt", encoding="utf-8", newline="") as gz:
            writer = csv.DictWriter(gz, fieldnames=["fen", "eval_cp", "rating_band", "mean_rating", "stage", "game_id"])
            writer.writeheader()
            iterator = iter(rows)
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < args.workers * 2:
                    try:
                        row = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(pool.submit(evaluate, row))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    writer.writerow(future.result())
                    count += 1
                    if count % 1000 == 0:
                        logging.info("Stockfish labelled %s positions", f"{count:,}")
        succeeded = True
    finally:
        for future in pending:
            future.cancel()
        # Close transports immediately (also interrupts outstanding analyses),
        # then join executor threads and await OS process reaping.
        with lock:
            for engine in engines:
                engine.close()
        pool.shutdown(wait=True, cancel_futures=True)
        # A task could have been starting its engine at the time of cancellation.
        for engine in engines:
            engine.close()
            engine.returncode.result(timeout=args.timeout + 5)
        if not succeeded and not (args.keep_partial or args.no_cleanup):
            temp_output.unlink(missing_ok=True)
    temp_output.replace(output_path)
    return count


def main() -> int:
    args = make_parser().parse_args()
    if args.target < 1 or args.months < 1 or args.depth < 1 or args.threads < 1 or args.hash_mb < 1 or args.workers < 1 or args.timeout < 1:
        print("target, months, depth, workers, threads, hash-mb and timeout must be positive.", file=sys.stderr)
        return 2
    if args.nodes < 0 or args.max_cp < 0 or args.max_games_per_month < 0:
        print("nodes, max-cp and max-games-per-month cannot be negative.", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rng = random.Random(args.seed)
    output_path = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_dir / "checkpoint.json"

    quotas = quota_map(args.target)
    scratch = tempfile.TemporaryDirectory(prefix="run-", dir=work_dir)
    db = sqlite3.connect(str(Path(scratch.name) / "candidates.sqlite"))
    db.execute("PRAGMA cache_size=-8192")
    db.execute("CREATE TABLE candidates (bucket INTEGER, slot INTEGER, row TEXT, PRIMARY KEY(bucket, slot))")
    reservoirs = {key: DiskBucket(db, i) for i, key in enumerate(quotas)}
    seen_hashes = DiskSeen(db)
    candidate_counts: Counter = Counter()
    scanned_months: list[str] = []
    month_names = month_range(args.start_month, args.months)

    if args.workers * args.threads > (os.cpu_count() or 1):
        logging.warning("workers x threads exceeds available logical CPUs")
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Checkpoint is metadata-only; candidate storage is not resumed.
        if args.no_resume and checkpoint_path.exists():
            checkpoint_path.unlink()
        prior = load_checkpoint(checkpoint_path) if not args.no_resume else None
        if prior and prior.get("scanned_months"):
            logging.warning("Checkpoint found; candidate storage is not resumed, so this run rescans from --start-month.")
    
        for month in month_names:
            scan_month(month, quotas, reservoirs, seen_hashes, candidate_counts, rng, args)
            scanned_months.append(month)
            counts = {f"{r}-{r+99}/{stage}": len(reservoirs[(r, stage)])
                      for r in range(RATING_LOWER, RATING_UPPER, RATING_STEP) for stage in STAGES}
            save_checkpoint(checkpoint_path, {
                "target": args.target, "start_month": args.start_month.isoformat()[:7],
                "months_requested": args.months, "scanned_months": scanned_months,
                "counts": counts, "complete": False,
            })
            # scan_month stops within the archive when every quota is full.
            if all(len(reservoirs[key]) >= quotas[key] for key in quotas):
                logging.info("All quotas filled after %s; stopping before the next month.", month)
                break
    
        missing = {key: quotas[key] - len(reservoirs[key]) for key in quotas if len(reservoirs[key]) < quotas[key]}
        if missing:
            logging.warning("Some quotas remain unfilled after %d months.", len(scanned_months))
            for (band, stage), deficit in sorted(missing.items()):
                if deficit:
                    logging.warning("  %d-%d %s: short by %d", band, band + 99, stage, deficit)
            if not args.keep_partial:
                logging.error("No output written. Increase --months and rerun; use --keep-partial to export a partial dataset.")
                return 3
    
        rows = (row for key in quotas for row in reservoirs[key])
        if missing:
            output_path = output_path.with_name(output_path.name + ".partial.csv.gz")
        logging.info("Labelling with %d workers (%d total threads, %d MiB total hash).",
                     args.workers, args.workers * args.threads, args.workers * args.hash_mb)
        actual = evaluate_and_write(rows, args, output_path)
        logging.info("Wrote %s", output_path)
    
        save_checkpoint(checkpoint_path, {
            "target": args.target, "output": str(output_path),
            "start_month": args.start_month.isoformat()[:7], "months_requested": args.months,
            "scanned_months": scanned_months,
            "actual_positions": actual,
            "complete": not bool(missing),
            "missing_quotas": {f"{r}-{r+99}/{s}": n for (r, s), n in missing.items()},
        })
    
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted; no complete output published by this run")
        return 130
    except Exception:
        logging.exception("Run failed; no complete output published by this run")
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        db.commit()
        db.close()
        if args.no_cleanup:
            scratch._finalizer.detach()
            logging.info("Retained candidates at %s", scratch.name)
        else:
            scratch.cleanup()
        if not (args.keep_checkpoint or args.no_cleanup):
            checkpoint_path.unlink(missing_ok=True)
            checkpoint_path.with_suffix(".json.tmp").unlink(missing_ok=True)
        try:
            work_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
