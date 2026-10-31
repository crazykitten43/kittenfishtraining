# Kittenfish training comparison — 31 October 2026

All three models trained from scratch, sequentially, for one complete epoch on the same 1,000,000 positions, then evaluated on the same held-out 100,000 positions. CUDA BF16 autocast on NVIDIA L4 (23,034 MiB VRAM). Seed 42.

| Model | Effective parameters | Batch | Training seconds | Positions/s | Peak allocated GPU MiB | Sampled device peak MiB | Validation MAE cp |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet | 3,078,697 | 256 | 104.314574 | 9586.388 | 276.354492 | 566 | 259.39040234 |
| transformer | 6,371,585 | 256 | 251.433040 | 3977.202 | 1537.231934 | 1922 | 197.07463445 |
| nnue | 10,619,425 | 4096 | 7.326152 | 136497.305 | 372.313965 | 748 | 214.10278250 |

Timing includes data loading and training, excludes validation and final checkpoint writes. GPU allocation peaks use PyTorch counters; whole-device memory was sampled every 2 seconds and may miss brief peaks. Exact bytes and unrounded measurements are in summary.json.

NNUE stores 10,620,289 parameter values, including an unused 864-value embedding padding row; effective parameter count is 10,619,425. BF16 autocast retains FP32 embedding accumulation and optimizer weights.

SE-ResNet and transformer use AdamW, learning rate 0.0003, weight decay 0.0001, gradient clipping 1, and one-epoch cosine scheduling. NNUE uses its existing Adam training recipe: learning rate 0.0001, gradient clipping 10, and parameter bounds [-8, 8]. Thus this compares the included training recipes, rather than isolating architecture under identical optimization. NNUE converts targets and predictions to side-to-move perspective; absolute centipawn error is invariant under the sign conversion. All labels are clipped to ±2000 cp.

Subset selection: seed-42 sample without replacement of ten complete 100,000-record training shards and one 100,000-record validation shard from the existing kf100m manifest. Shard SHA-256 checksums verified. NNUE uses symlinks to precisely the same files; subset membership is identical, shuffle schedules differ. This is a shard sample, not a uniform sample across individual records.

Code commit: 957d7bdfc30d1a47f096c42672562505384bc45b. Current main fetched into isolated checkout; existing branch and datasets preserved. Existing PyTorch 2.9.1+cu128 and NumPy 2.5.4 satisfied requirements. No dependency installation was needed. Instrumentation adds synchronized training timing, memory peaks, and BF16 support for NNUE; changes are preserved in instrumentation.patch.

Validation: five experiment tests passed; all seven NNUE pipeline Python tests passed in the configured environment; C++ self-tests passed. The make test invocation also runs Python tests using system Python, where the optional GPU pipeline test was skipped; it passed separately in the configured environment. Both spatial models passed CUDA BF16 smoke runs. All final PyTorch checkpoints strictly reloaded and produced finite inference outputs. Checkpoint sizes and SHA-256 hashes are in summary.json.

Persistent VM artifact directory: /home/kittenfishchess/kittenfish-experiments-20261031/benchmark-20261031/

Spatial models: resnet/{best,last}.pt and transformer/{best,last}.pt. NNUE: nnue/model.last.pt plus model.best.kff and model.last.kff. This GitHub benchmark folder includes logs, metrics, configuration, checksums, exact selected data, and every checkpoint; binary artifacts use Git LFS. The original artifacts also remain on the VM persistent disk.

Project: project-845b9345-31a5-43ef-8ea. Zone: asia-northeast3-a. VM: kittenfishgpu. Stop requested after verification; no VM or disk deletion performed.

The portable manifest.json refers to nnue-subset/ files in this folder; manifest.vm.json preserves the original absolute-path manifest. Both contain the same records and shard SHA-256 values. Original configuration and checkpoint dataset identities remain unchanged. Source snapshots are in the experiment folder; instrumentation.patch records the measured VM changes.
