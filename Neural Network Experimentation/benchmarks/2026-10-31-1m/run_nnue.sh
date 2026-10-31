#!/bin/bash
set -Eeuo pipefail
ROOT=/home/kittenfishchess/kittenfish-experiments-20261031
OUT=$ROOT/benchmark-20261031
PY=/home/kittenfishchess/kittenfishtraining-100m/.venv/bin/python
exec > >(tee -a "$OUT/nnue-orchestration.log") 2>&1
trap 'rc=$?; echo "$rc" > "$OUT/nnue-exit-code"; sudo shutdown -c || true; sudo shutdown -h +10; exit "$rc"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
while [ ! -f "$OUT/exit-code" ]; do sleep 2; done
test "$(cat "$OUT/exit-code")" = 0
sudo shutdown -c
sudo shutdown -h +120
cd "$ROOT/v1"
$PY -m unittest discover -v -s tests 2>&1 | tee "$OUT/nnue-tests.log"
make test 2>&1 | tee "$OUT/nnue-cpp-tests.log"
git diff > "$OUT/instrumentation.patch"
mkdir -p "$OUT/nnue"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 2 > "$OUT/nnue-gpu.csv" &
MONITOR=$!
$PY scripts/train_l4.py "$OUT/nnue-subset/manifest.json" "$OUT/nnue/model" --device cuda --batch-size 4096 --workers 4 --cpu-threads 4 --epochs 1 --checkpoint-every 10000 2>&1 | tee "$OUT/nnue.log"
kill "$MONITOR" || true
echo NNUE_COMPLETE
