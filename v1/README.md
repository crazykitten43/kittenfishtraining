# KittenFish NNUE training

Original C++17 CPU NNUE baseline: training, quantization, incremental evaluation,
model IO and benchmarks. **No search, UCI, move generator or trained weights.**
This is Stockfish-inspired, not a copy of Stockfish's network or file format.

## Architecture

- 16 king buckets (2x2 square regions) per perspective.
- 12 relative piece types x 64 oriented squares per bucket; kings included.
- Shared 12,288 x 864 sparse feature transformer and 864 biases.
- Two 864-element accumulators: White and Black perspectives.
- Clipped ReLU [0,1], concatenated side-to-move first, linear scalar output.
- 10,619,425 trainable parameters. Float training; int16 weights and int32
  accumulators for deployment, with int64 output sum. No SIMD intrinsics needed;
  the optimized compiler can vectorize loops.
- King bucket changes rebuild the affected perspective. Other changes subtract
  removed pieces and add new pieces. Final-board updates cover captures,
  castling, en passant and promotions without special move encodings.

The small output head is intentionally simpler than modern Stockfish. Strength
and 100k single-thread evaluations/s are targets, not measured guarantees.
The update API scans 64 squares; a future engine can pass explicit dirty pieces.

## Build and test

Linux/macOS, GNU Make and a C++17 compiler; no third-party libraries:

```sh
make
make test
```

`-march=native` targets the build CPU. For a portable binary:
`make CXXFLAGS='-O3 -std=c++17 -Wall -Wextra -Wpedantic'`.

## Important: score perspective fix

The KittenFish `positions_100k.csv.gz` dataset uses **side-to-move** scores.
`PositionGeneration.py` explicitly calls `info["score"].pov(board.turn)`.
The C++ TSV loader expects **White** scores and internally converts to side to
move. Passing the CSV values directly as White scores reverses Black-to-move
training targets. Old conversion instructions did this incorrectly. Do not
resume weights trained on those incorrectly converted files for the clean rerun.

Use the explicit converter (Python standard library only):

```sh
python3 scripts/prepare_dataset.py data/positions_100k.csv.gz data --input-perspective white
./build/kfnnue train data/train_white.tsv data/val_white.tsv checkpoints/fixed 10 0.0001
./build/kfnnue diagnose checkpoints/fixed.best.kff data/val_white.tsv
```

For a future dataset that genuinely contains White-perspective labels, specify
`--input-perspective white` instead. The converter never guesses. It preserves
the existing deterministic game-grouped 80/20 split, deduplicates by FEN's first
four fields, and writes new `_white.tsv` filenames so old files remain available.
It loads deduplication keys in RAM and is intended for curated datasets, not
hundreds of millions of records in a single invocation. Validation rows must
not be used for training, including tiny-set overfit experiments.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the diagnosis and measured results.

The trainer now prints zero/material baselines, training-sample metrics,
validation prediction mean and standard deviation, target standard deviation,
prediction correlation, epoch time, and validation metrics for each newly saved
quantized model. `diagnose` additionally reports activation clipping on up to
256 positions. Training metrics use a shuffled sample of up to 4096 rows, not
the complete training dataset. Baseline material is 100/320/330/500/900cp and does
not use any learned parameters.

## Convert Lichess's evaluated-position database

The database is JSONL compressed with Zstandard: one JSON position per line.
Lichess recommends taking the evaluation record with the greatest `depth`, then
using its first PV. The converter follows that rule, skips mate-only or too
shallow positions, clamps scores to +/-2000cp, and stops as soon as it collects
the requested number of unique usable positions (500,000 by default). It writes
the first eligible positions in archive order rather than scanning the entire
archive for a uniform reservoir sample, so the resulting dataset may reflect
the archive's ordering. It splits selected positions by a stable position hash;
identical positions therefore cannot leak across the train/validation split.
The export has no game IDs, so it cannot make game-grouped splits. Lichess FENs
have four fields; the converter appends the two move counters expected by this
trainer.

The converter requires an explicit input score perspective. This workflow uses
`white`: the Lichess analysis board normalizes scores to White's perspective.
Check a Black-to-move sample against Lichess analysis if you want to verify the
sign. Choosing `stm` instead flips Black-to-move values.

Install the one Python dependency and download the archive:

```sh
python3 -m pip install --user zstandard
curl --fail --location --retry 3 \
  https://database.lichess.org/lichess_db_eval.jsonl.zst \
  | gcloud storage cp - gs://YOUR_BUCKET/lichess_db_eval.jsonl.zst
```

Replace `YOUR_BUCKET` with the name of a bucket you own. This streams the
compressed archive from Lichess straight into Cloud Storage, without a local
archive file.

Try the first 100,000 lines before processing the full archive:

```sh
gcloud storage cat gs://YOUR_BUCKET/lichess_db_eval.jsonl.zst | \
  python3 scripts/convert_lichess_eval.py - data/sample \
  --target 20000 --max-lines 100000 --input-perspective white
```

Review a few converted rows and verify the sign against Lichess analysis. If
scores are White-perspective, rerun with `--input-perspective white`. Once
confirmed, process the whole archive, for example:

```sh
gcloud storage cat gs://YOUR_BUCKET/lichess_db_eval.jsonl.zst | \
  python3 scripts/convert_lichess_eval.py - data/lichess500k \
  --target 500000 --min-depth 20 --input-perspective white
./build/kfnnue train data/lichess500k/train_white.tsv \
  data/lichess500k/val_white.tsv checkpoints/lichess500k 20 0.0001
```

The converter streams decompression and keeps only the selected positions
in memory. Both `--max-lines` and `--target` can stop early, so results come
from the beginning of the archive and may reflect its ordering. See
[database.lichess.org](https://database.lichess.org/) for the schema and its
recommendation about depth and PV selection.

## Training data contract

Uncompressed TSV, no header: full FEN, one tab, numeric centipawns **from White's
perspective**. Blank lines and lines beginning with # are ignored.

```text
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1	20
```

Use an actual tab, not the two characters backslash+t. Mate scores must be
converted upstream using an explicit chosen convention, or excluded. Labels
are clipped to +/-2000cp, converted to side-to-move, and divided by 400.
Lichess JSONL conversion is not included; do not pass its raw archive here.
Prepare independent training/validation files, deduplicate positions across
splits, and group by source game when source identity is available.

```sh
./build/kfnnue train train.tsv val.tsv kittenfish 10 0.0001
# Resume float weights (optimizer is plain SGD, no momentum state):
./build/kfnnue train train.tsv val.tsv continued 10 0.0001 kittenfish.last.kff
./build/kfnnue export kittenfish.best.kff kittenfish.nnue
./build/kfnnue eval kittenfish.nnue 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
./build/kfnnue bench kittenfish.nnue 1000000
```

Online sparse SGD, deterministic epoch shuffling (seed 42), MSE loss,
clipped output gradient, bounded parameters. Validation prints float-model
MSE in cp squared and MAE in cp. The best float validation MSE selects
`.best.kff` and quantized `.nnue`; `.last.kff` supports continuation. On resume,
shuffle order and best-metric tracking restart. Quantized validation is printed when a new best model is saved; quantization
can change accuracy. The checkpoint is still selected by float-model MSE.

This is a correctness-first CPU trainer, not a high-throughput GPU training
pipeline. It loads datasets into RAM and updates one sample at a time. Do not
load hundreds of millions of rows at once. Use curated subsets initially.

## Benchmarks and API

`bench` measures a single thread, printing separate cached-accumulator forward
throughput and incremental-update-plus-eval throughput. It alternates start
and e4 positions, so it is a warm-cache microbenchmark, not a search NPS claim
or a diverse-position benchmark. Full FEN parsing/rebuild cost is excluded.
Benchmark your eventual target CPU; no speed promise is implied.

`QuantNet::start(position)` builds state; `update(state, next_position)` updates
it; `eval(state)` returns centipawns from side-to-move perspective. Copy a State
before an update to implement restore/undo later. One network is read-only and
can be shared; each worker must own its State. This code validates basic FEN
syntax, both kings and piece count, but does not validate legal chess positions.
The evaluator learns static position scores; it does not adjudicate draws,
checkmate, repetition, castling rights or the fifty-move rule.

Model formats KFQ1 and KFF1 are fixed-architecture native-endian files, not
Stockfish-compatible .nnue files. Use trusted local models on the same endian
architecture. There is no checksum, compression, or architecture conversion.

`make test` also checks converter perspective handling, game grouping,
deduplication and the required explicit perspective argument.

`selftest` checks finite-difference gradients for feature, bias and output
weights, 200 accumulator updates against complete rebuilds (including
king bucket changes), one SGD descent step, and model roundtripping. Its board
mutations intentionally exercise tensor bookkeeping rather than legal play.
