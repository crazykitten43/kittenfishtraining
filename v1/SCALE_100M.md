# 100M positions and NVIDIA L4

Use `scripts/build_shards.py` and `scripts/train_l4.py` for large runs. The original
TSV converter/trainer remain useful for small experiments and still load all rows
into RAM. Do not use the original `train` command for 100M positions.

## Machine and storage

Recommended starting point: **g2-standard-8**, one NVIDIA L4 (24 GB), 8 vCPUs,
32 GB host RAM. Conversion is single-process CPU/SQLite work; it can run on the
existing e2-standard-2 with a smaller shard size (100000), but will be slower.
An e2-standard-8 is an alternative for conversion without paying for an idle GPU.
Benchmark the first million records before committing to the full run.

Keep the compressed archive: decompression is streamed. 100M packed records take
6.7 GB decimal (6.24 GiB), including validation. Budget **100 GiB of free space
in addition to the archive**, for SQLite data/index/WAL, shards, temporary
exports, checkpoints and headroom. Start with **500 GiB pd-ssd**, provided the
archive plus 100 GiB fits; otherwise size to `archive GiB + 100 GiB`, rounded up.
This is capacity planning, not a measurement of your archive. SQLite's actual
size depends on page utilization; measure `du` and `df` during the pilot.
`pd-balanced` is a cheaper option; SSD helps random dedup index accesses.
Local SSD is optional scratch, not the only copy of the dataset/checkpoints.

G2 supports pd-ssd/pd-balanced, but not pd-standard. You cannot simply add an L4
to an E2 machine: use a G2 instance in a supported zone with available quota.
Keep the existing archive disk and attach it to the G2 in the same zone after
stopping the writer and detaching it from the old VM. If it is pd-standard,
make a snapshot and create a pd-ssd copy instead. Avoid assuming a cross-family
machine-type change is supported. Do not delete the source VM/disk until verified.

Reference specifications and setup:
- [G2 configurations and supported disks](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines)
- [GPU creation restrictions](https://docs.cloud.google.com/compute/docs/gpus/create-vm-with-gpus)
- [Disk resize and filesystem growth](https://docs.cloud.google.com/compute/docs/disks/resize-persistent-disk)
- [PyTorch CUDA installation](https://pytorch.org/get-started/previous-versions/)

The following commands are for the operator; this change launches no resources.
Replace uppercase values. The resize command only enlarges a disk. If the existing
disk already exceeds 500 GiB, retain its size. Inspect before resizing:

```sh
gcloud compute instances describe OLD_VM --zone=ZONE
gcloud compute disks describe DATA_DISK --zone=ZONE
gcloud compute disks resize DATA_DISK --zone=ZONE --size=500GB
# On the mounted VM: identify the actual device, filesystem and partition first.
lsblk -f
df -h /mnt/data
# ext4 partition example, ONLY after verifying its actual device name:
sudo growpart /dev/sdb 1
sudo resize2fs /dev/sdb1
# For whole-device ext4, use resize2fs on that device without growpart.
# For XFS use sudo xfs_growfs /mnt/data instead.
```

For an operator-created G2, use Ubuntu 22.04/24.04 and Google's documented NVIDIA
driver installation procedure, not a Deep Learning VM boot image (G2 restriction).
Select a zone with L4 capacity, use a 50 GiB boot disk, and keep data on a separate
persistent disk. Mount the data disk without formatting it. Confirm `nvidia-smi`
shows the L4 before installing the CUDA wheel.

Example operator commands for a new G2 in the **same zone** as an existing compatible
data disk (replace names, check quota/capacity and preserve the disk first):

```sh
gcloud compute instances create TRAIN_VM --zone=ZONE --machine-type=g2-standard-8 \
  --maintenance-policy=TERMINATE --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud --boot-disk-type=pd-balanced --boot-disk-size=50GB
# After safely detaching the data disk from the stopped source VM:
gcloud compute instances attach-disk TRAIN_VM --zone=ZONE --disk=DATA_DISK
```

If the archive is on the old VM's boot disk, prefer a snapshot/copy and mount it as
a data disk on the new VM. Take special care to preserve the original boot disk;
never run mkfs on a disk containing the archive. Driver installation is a separate
step; follow the Google GPU guide linked above and verify with `nvidia-smi`.

## Installation and pilot

From `v1` (Python 3.10+):

```sh
python3 -m venv .venv
. .venv/bin/activate
# Known published CUDA wheel; driver must support CUDA 12.8.
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-l4.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
make
make test
```

The CUDA runtime is in the wheel; install a compatible NVIDIA driver on the VM.
Use the PyTorch selector if a different CUDA wheel is needed. No torchvision or
full CUDA compiler toolkit is required.

```sh
# Pilot in its own directory. Confirm archive score convention before choosing
# white vs stm; the database eval cp convention is White perspective.
python scripts/build_shards.py /mnt/data/lichess_db_eval.jsonl.zst /mnt/data/pilot \
  --target 1000000 --input-perspective white --min-depth 20
cat /mnt/data/pilot/manifest.json
du -sh /mnt/data/pilot
df -h /mnt/data
python scripts/train_l4.py /mnt/data/pilot/manifest.json /mnt/data/models/pilot \
  --device cuda --batch-size 4096 --workers 4 --epochs 1 --max-steps 100 \
  --checkpoint-every 100
nvidia-smi
```

Start at batch 4096; reduce to 2048 or 1024 if GPU memory is exhausted. Try workers
2/4 and batch 4096/8192 on the pilot; compare logged positions/second and GPU
utilization. No claim of L4 throughput or total training time is made without a
run on your hardware. CPU smoke tests cover the same model/data/checkpoint path;
CUDA kernels and L4 performance must be verified on the VM. FP32 is intentional:
EmbeddingBag sums are small, stable and fit easily; mixed precision is not assumed
to improve this sparse-input workload.

## Full conversion and resume

```sh
python scripts/build_shards.py /mnt/data/lichess_db_eval.jsonl.zst /mnt/data/kf100m \
  --target 100000000 --input-perspective white --min-depth 20 \
  --shard-size 1000000 --val-permille 10 --commit-every 10000
```

Re-run the exact command after interruption. A SQLite WAL transaction stores both
records and the source line cursor every 10000 lines. Uncommitted records roll
back. Compressed input is replayed from the beginning to the saved line (without
JSON parsing for the prefix); a zstd byte seek is not a valid restart strategy.
Restart latency can be long on a large archive. Plain JSONL also replays the prefix.
The source path/size/mtime and generation options must match; these checks detect
accidental replacement, not adversarial same-size/same-time modification. Keep the
archive immutable. Do not use stdin for restartable generation.

Unique network inputs are deduplicated using the exact board+side bytes in SQLite,
not an in-memory set or a probabilistic Bloom filter. Rights, EP and move counters
are ignored because v1's network cannot see them. The first qualifying label wins.
SHA-256 over that same identity deterministically assigns 1% to validation by
default, so equivalent network inputs cannot leak between splits. This is a
position split; related positions from the same game can still cross splits,
since the evaluation database does not supply game IDs. Selection stops at the
first 100M qualifying unique inputs, not a uniform sample of the archive.

Deepest qualifying eval, first PV cp is used; mate-only/shallow, malformed,
non-finite labels, bad rank widths, missing kings, >32 pieces, and pawns on first/
last ranks are rejected. This is structural validation, not full chess legality.
Scores are clipped to +/-2000 cp and converted explicitly to White perspective.
Stats include malformed/filtered/duplicate/clipped counts and side counts. Review
these and split counts in `manifest.json`. The target includes train+validation;
416M source positions do not guarantee 100M pass your depth filter.
`target_reached: false` explicitly identifies an undersized dataset; a nonempty
partial dataset is usable for pilots. `--max-lines` limits absolute source lines,
not additional lines after a restart.

Export is bounded by shard size (default about a few hundred MB in Python), not
total dataset size; SQLite's page cache is capped at 64 MiB. Shards are written to
temporary files, fsynced and renamed; the manifest is published last with per-shard
counts and SHA-256. An export interruption is recovered by deterministic re-export
from SQLite. Do not edit/remove the dedup database if conversion may need resuming.
Do not generate and train concurrently in the same destination, or run multiple
trainers with the same output prefix. Only manifest-listed shards are consumed.

## Full GPU training

```sh
python scripts/train_l4.py /mnt/data/kf100m/manifest.json /mnt/data/models/kf100m \
  --device cuda --batch-size 4096 --workers 4 --epochs 10 --lr 0.0001 \
  --checkpoint-every 1000
# Resume, keeping batch size/seed/lr/dataset identical:
python scripts/train_l4.py /mnt/data/kf100m/manifest.json /mnt/data/models/kf100m \
  --device cuda --batch-size 4096 --workers 4 --epochs 10 --lr 0.0001 \
  --checkpoint-every 1000 --resume /mnt/data/models/kf100m.last.pt
```

Startup verifies all shard lengths/hashes. Memory-mapped shard access, bounded
batches, pinned host buffers and worker prefetch feed the GPU. Shards and batch
blocks shuffle each epoch; rows shuffle within each block. This is bounded local
mixing, not a global 100M-row permutation. Every row, including each final short
batch, is used once per epoch. Each worker opens shards lazily and closes them.
`EmbeddingBag` sums sparse piece features without a dense 12288-wide input or a
batch x 64 x 864 embedding intermediate. The architecture, perspective ordering,
clipped activations and KFF1 layout match C++. Adam mini-batch optimization differs
from the C++ single-row SGD; learning rates and learning curves are not interchangeable.
Validation reports full-split MSE/MAE in cp units. Learning-rate tuning and actual
playing-strength testing remain necessary.

`.last.pt` atomically saves model, Adam moments, epoch, next batch, global step,
best validation MSE and dataset/options identity. The schedule is derived from
seed+epoch, so resume skips completed batches independently of worker count. CPU
resume equivalence is tested; CUDA reduction ordering may cause small numerical
differences. A crash replays only steps since the last checkpoint. Checkpoints
are roughly 130 MB. Only load trusted `.pt` files (PyTorch pickle format).
`--epochs` is the total desired epoch count on resume; `--max-steps` is a total
step cap for smoke runs. To resume a pilot into full training omit the step cap.

`.best.kff` is selected by validation MSE; `.last.kff` is the latest epoch (or
step-capped run). Export the selected model through the existing C++ quantizer:

```sh
./build/kfnnue export /mnt/data/models/kf100m.best.kff /mnt/data/models/kf100m.nnue
```

## C++ streaming reference path

```sh
./build/kfnnue stream-train /mnt/data/kf100m/train.list /mnt/data/kf100m/val.list \
  /mnt/data/models/reference 1 0.0001
# Optional final argument: a .kff to warm-start model weights.
```

Lists are generated next to the manifest and contain relative shard paths. This
path bounds the shuffle buffer to 8192 positions and streams full validation. It
checks packed record structure/truncation, but does not verify manifest hashes;
use the Python training startup for hash verification. C++ saves model weights
at epoch boundaries; its optional warm start does not restore shuffle/best-score
state. Use the PyTorch checkpoint path for precise batch progress and the L4 for
practical large-scale training. KFD1 records are little-endian 67-byte rows:
64 bytes piece IDs at a1..h8 (0 empty, PNBRQK=1..6, pnbrqk=7..12), one byte stm
(0 white/1 black), signed int16 White cp; no header. Manifest format is `KFD1`.

## Validation

`make test` runs the existing gradient/incremental/quantization checks plus
conversion tests. With requirements installed it also tests multi-worker coverage,
mid-epoch Adam resume equivalence, shard corruption, packed C++ training and
PyTorch-to-C++ float evaluation compatibility. Run it from the activated environment
to include all tests. A local CPU pass is not a 100M-scale or L4 hardware benchmark.
