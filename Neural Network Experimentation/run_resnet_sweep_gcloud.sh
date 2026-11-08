#!/usr/bin/env bash
# Run from a machine with authenticated gcloud access to kittenfishgpu.
set -Eeuo pipefail

vm=kittenfishgpu
ssh_target=kittenfishchess@$vm
flags=(--project=project-845b9345-31a5-43ef-8ea --zone=asia-northeast3-a --quiet)
repo=https://github.com/crazykitten43/kittenfishtraining.git
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
revision=$(git -C "$source_dir" rev-parse HEAD)
run_id="resnet-sweep-$(date -u +%Y%m%dT%H%M%SZ)-$$"
local_run="$PWD/$run_id"
remote_run="kittenfish-benchmarks/$run_id"

if ! command -v gcloud >/dev/null; then
    PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi
command -v gcloud >/dev/null || { echo 'gcloud is required.' >&2; exit 1; }

description=$(gcloud compute instances describe "$vm" "${flags[@]}" \
    --format='value(status,machineType.basename())')
read -r status machine_type <<< "$description"
if [[ $machine_type != g2-standard-8 ]]; then
    echo "Expected g2-standard-8, found $machine_type." >&2
    exit 1
fi
case "$status" in
    RUNNING) gcloud compute instances stop "$vm" "${flags[@]}" ;;
    TERMINATED) ;;
    *) echo "Cannot start $vm from state $status." >&2; exit 1 ;;
esac

cleanup() {
    local result=$?
    trap - EXIT
    trap '' INT TERM
    for attempt in 1 2 3; do
        if gcloud compute instances stop "$vm" "${flags[@]}" &&
            [[ $(gcloud compute instances describe "$vm" "${flags[@]}" --format='value(status)') == TERMINATED ]]; then
            echo "$vm is TERMINATED."
            exit "$result"
        fi
        sleep 5
    done
    echo "Could not verify shutdown of $vm. Check its state in Google Cloud." >&2
    exit 1
}

mkdir "$local_run"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
gcloud compute instances start "$vm" "${flags[@]}"

ssh_flags=(--ssh-flag=-oConnectTimeout=10 --ssh-flag=-oServerAliveInterval=15 --ssh-flag=-oServerAliveCountMax=4)
ready=0
for attempt in {1..18}; do
    if gcloud compute ssh "$ssh_target" "${flags[@]}" "${ssh_flags[@]}" --command='nvidia-smi -L'; then
        ready=1
        break
    fi
    sleep 10
done
(( ready == 1 )) || { echo 'The GPU VM did not become ready.' >&2; exit 1; }

result=0
gcloud compute ssh "$ssh_target" "${flags[@]}" "${ssh_flags[@]}" \
    --command="bash -s -- $remote_run $revision $repo" <<'REMOTE' 2>&1 | tee "$local_run/run.log" || result=$?
set -Eeuo pipefail
run="$HOME/$1"
revision=$2
repo=$3
mkdir -p "$run"
printf 'VM artifacts: %s\n' "$run"

archive_reports() {
    local result=$?
    trap - EXIT
    if [[ -d "$run/results" ]]; then
        (
            cd "$run"
            find results -type f \( -name '*.csv' -o -name '*.json' \) -print0 |
                tar -czf reports.tar.gz --null -T -
        ) || exit 1
    fi
    exit "$result"
}
trap archive_reports EXIT

export GIT_LFS_SKIP_SMUDGE=1
git init -q "$run/source"
cd "$run/source"
git fetch --depth 1 "$repo" "$revision"
git checkout --detach FETCH_HEAD
git remote add origin "$repo"
git lfs pull --include='Neural Network Experimentation/benchmarks/2026-10-31-1m/nnue-subset/*.kfd' --exclude=''
cd 'Neural Network Experimentation'

has_cuda() {
    "$1" -c 'import numpy, torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()' >/dev/null 2>&1
}

python_bin=
if has_cuda python3; then
    python_bin=$(command -v python3)
else
    while IFS= read -r -d '' candidate; do
        if [[ $candidate == */bin/python || $candidate == */bin/python3 ]] && has_cuda "$candidate"; then
            python_bin=$candidate
            break
        fi
    done < <(find "$HOME" -maxdepth 6 \( -type f -o -type l \) \
        \( -name python -o -name python3 \) -print0)
fi
if [[ -z $python_bin ]]; then
    echo 'No installed Python with CUDA PyTorch, BF16 and NumPy was found.' >&2
    exit 1
fi
nvidia-smi
export OMP_NUM_THREADS=4
timeout --kill-after=30s 3h "$python_bin" -u benchmark_resnet_sweep.py \
    --manifest benchmarks/2026-10-31-1m/manifest.json --output "$run/results"
REMOTE

if ! gcloud compute scp "$ssh_target:$remote_run/reports.tar.gz" "$local_run/" "${flags[@]}"; then
    echo "Reports could not be copied. See $local_run/run.log and $remote_run on the VM." >&2
    exit 1
fi
tar -xzf "$local_run/reports.tar.gz" -C "$local_run"
printf 'Local reports and log: %s\n' "$local_run"
exit "$result"
