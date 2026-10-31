#!/bin/bash
set -Eeuo pipefail
ROOT=/home/kittenfishchess/kittenfish-experiments-20261031
OUT=$ROOT/benchmark-20261031
PY=/home/kittenfishchess/kittenfishtraining-100m/.venv/bin/python
mkdir -p "$OUT"
exec > >(tee -a "$OUT/orchestration.log") 2>&1
trap 'rc=$?; echo "FINAL_EXIT_CODE=$rc"; date -u; echo "$rc" > "$OUT/exit-code"; sudo shutdown -c || true; sudo shutdown -h +10; exit "$rc"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$ROOT/Neural Network Experimentation"
date -u
git rev-parse HEAD > "$OUT/commit.txt"
git diff > "$OUT/instrumentation.patch"
$PY -m pip freeze > "$OUT/packages.txt"
nvidia-smi > "$OUT/gpu.txt"
$PY -m unittest -v test_experiments.py 2>&1 | tee "$OUT/tests.log"
for MODEL in resnet transformer; do
  $PY train.py --model "$MODEL" --manifest "$OUT/manifest.json" --output "$OUT/smoke-$MODEL" --epochs 1 --batch-size 256 --workers 2 --device cuda --amp bf16 --max-train-batches 3 --max-val-batches 2 2>&1 | tee "$OUT/smoke-$MODEL.log"
done
for MODEL in resnet transformer; do
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 2 > "$OUT/$MODEL-gpu.csv" &
  MONITOR=$!
  $PY train.py --model "$MODEL" --manifest "$OUT/manifest.json" --output "$OUT/$MODEL" --epochs 1 --batch-size 256 --workers 4 --device cuda --amp bf16 2>&1 | tee "$OUT/$MODEL.log"
  kill "$MONITOR" || true
done
echo COMPLETE
