# Neural Network Experimentation

Two PyTorch scalar evaluators, with the same inputs, White-perspective targets,
MSE objective and shared training loop:

| Model | Body | Parameters (approximately) |
| --- | --- | ---: |
| `resnet` | 10 SE residual blocks, 128 channels, two 3x3 convolutions/block; SE reduction 8 | 3.1M |
| `transformer` | 64 square tokens, 8 pre-norm layers, width 256, 8 heads, FFN 1024, dropout 0.1 | 6.4M |

The trainer prints the exact parameter count. Both use fixed White orientation,
12 piece indicators plus a side-to-move feature per square. ResNet uses a spatial
flattened value head; transformer uses learned square positions and mean pooling.
No policy head. These are GPU evaluation experiments, not incremental NNUEs or
exports compatible with the current C++ engine.

## Setup

Use Python 3.10+ and install the appropriate CUDA build of PyTorch for your machine,
then:

```bash
cd 'Neural Network Experimentation'
python -m pip install -r requirements.txt
python -m unittest -v test_experiments.py
```

## Data

Use `v1/scripts/build_shards.py`'s existing `manifest.json` directly. Its KFD1
format stores 64 piece bytes (a1..h8), one side-to-move byte (0=White, 1=Black),
and an int16 White-perspective centipawn target. No full dataset is loaded into
RAM; memory is bounded by 65,536 records per worker plus prefetched batches.
Shard order, chunk order and records within chunks shuffle each training epoch.
Workers partition shards without duplication. Use enough shards to keep workers
busy. This is a bounded local shuffle, not a uniform permutation of all positions.

For the existing pre-split, headerless `FEN<TAB>White_cp` TSV files (comments allowed):

```bash
python prepare_tsv.py --train /data/train_white.tsv.gz --val /data/val_white.tsv.gz \
  --input-perspective white --output /data/spatial-shards
```

The output directory must be new. Scores are clipped to +/-2000 cp, matching v1.
Invalid rows fail with filename and line number. Splits are preserved; conversion
does not deduplicate or check overlap. Use the existing deduplicated train/val
splits for both models. Do not pass side-to-move labels as White labels.

**KFD1 omits castling rights, en passant and move counters.** These experiments
therefore cannot distinguish positions differing only in those fields. Keeping
this format permits a direct comparison on the existing dataset; richer features
would require regenerating data for both models. Repetition history is absent too.

## Train

From this directory, run the models separately (same dataset and sample budget):

```bash
python train.py --model resnet --manifest /data/spatial-shards/manifest.json \
  --output runs/resnet --epochs 20 --batch-size 256 --workers 4 --device cuda --amp bf16

python train.py --model transformer --manifest /data/spatial-shards/manifest.json \
  --output runs/transformer --epochs 20 --batch-size 256 --workers 4 --device cuda --amp bf16
```

Use `--device cpu --amp off --workers 0` for CPU, or `--device mps --amp off`
on Apple Silicon. CUDA bf16 requires supported hardware. Lower batch size if you
run out of VRAM. Defaults: AdamW, learning rate 0.0003, weight decay 0.0001,
cosine decay over the requested epochs, gradient norm clipping at 1. Targets are
cp / 400; network outputs are unbounded. Training uses MSE in scaled units;
reported validation MAE is in cp against the clipped labels, weighted by sample
count. Validation disables dropout and gradients and uses the entire split unless
explicitly capped. No Elo claim follows from validation MAE alone.

Each output directory contains `config.json`, `metrics.csv`, `best.pt` (lowest
validation MAE) and `last.pt` (latest completed epoch), saved atomically. Full
checkpoints include weights, optimizer, scheduler, AMP scaler and RNG states.
To resume at an epoch boundary, repeat the original command with
`--resume runs/resnet/last.pt`. Keep the original total `--epochs` and training
settings. Only load checkpoints you trust. Cross-device/worker configurations and
GPU kernels can affect reproducibility; bitwise reproducibility is not promised.
An interrupted epoch is replayed from its previous completed checkpoint.

For a quick end-to-end check before a long run, use a fresh output directory with
`--epochs 1 --max-train-batches 2 --max-val-batches 2 --workers 0`.
Do not compare capped validation numbers with full-validation results.

To load for inference:

```python
import torch
from models import make_model
checkpoint = torch.load('runs/resnet/best.pt', map_location='cpu', weights_only=False)
model = make_model(checkpoint['config']['model'])
model.load_state_dict(checkpoint['model'])
model.eval()
# pieces: uint8/int64 [batch,64], stm: [batch], using the KFD1 encoding above
with torch.inference_mode():
    white_cp = model(pieces, stm) * checkpoint['config']['cp_scale']
```

## Completed L4 comparison

See [the 31 October 2026 benchmark](benchmarks/2026-10-31-1m/README.md) for one-epoch SE-ResNet, transformer, and NNUE results on the same 1,000,000 training and 100,000 validation positions. Logs, configurations, exact metrics, checksums, smoke-run checkpoints, full-run checkpoints, and the selected data records are included. Binary weights and data use Git LFS; install Git LFS and run `git lfs pull` after cloning.

The instrumented `train.py` measures training time, throughput, and peak CUDA allocation. `train_nnue.py` is the existing v1 NNUE trainer with BF16 autocast and equivalent measurements; its native optimizer and batch size differ from the spatial models. The benchmark's historical scripts and configuration retain the original VM paths as provenance. The following commands use portable dataset paths and fresh output directories:

```bash
cd 'Neural Network Experimentation'
python -m unittest -v test_experiments.py
python train.py --model resnet --manifest benchmarks/2026-10-31-1m/manifest.json --output runs/replay-resnet --epochs 1 --batch-size 256 --workers 4 --device cuda --amp bf16
python train.py --model transformer --manifest benchmarks/2026-10-31-1m/manifest.json --output runs/replay-transformer --epochs 1 --batch-size 256 --workers 4 --device cuda --amp bf16
mkdir -p runs/replay-nnue
python train_nnue.py benchmarks/2026-10-31-1m/nnue-subset/manifest.json runs/replay-nnue/model --epochs 1 --batch-size 4096 --workers 4 --cpu-threads 4 --device cuda --checkpoint-every 10000
```
