#!/usr/bin/env bash
# Queue the frozen supplementary 2D arm after a CPU config and recovery check.
set -euo pipefail

root="$HOME/20261018_g1_stage/20261017"
run_dir="$root/logs/training/g5_2d_20261019"

verify_sha() {
    local expected="$1" file="$2" actual
    test -f "$file" || { printf 'Missing file: %s\n' "$file" >&2; exit 1; }
    actual="$(sha256sum "$file")"
    actual="${actual%% *}"
    test "$actual" = "$expected" || {
        printf 'SHA-256 mismatch: %s\nexpected %s\nactual   %s\n' \
            "$file" "$expected" "$actual" >&2
        exit 1
    }
}

verify_sha e308af1271885fca84b7007114572a6f00b40ceb88e1298840b65ce94df3ce96 \
    "$root/code/tools/train_g5_2d.py"
verify_sha 3ae7717569b6434d250b6b906c60e62fac23670e400341153b837aaf322f0ec2 \
    "$root/code/config/g5_2d_training_frozen.json"
verify_sha b98fbb0db5936a6260bc1d1f74b1446411e4ba8e1a9584ca715fb48d6a1624f3 \
    "$root/code/tools/audit_g5_2d_sampling.py"
verify_sha 99054855ecd2869796af9a4ce5decbdb2522d5402dac0fa7b4782676aa3d9f08 \
    "$root/code/hipmri_rewrite/sampling_2d.py"
verify_sha e6e7019bb4b896ac32851b9172e5da5346810622572c2068f8b64ae016e2da9b \
    "$root/logs/audits/g5_2d_sampling_full_20261018/manifest.json"
verify_sha e823a918f08672736028b5c47eefc2d901ff145e0824dd9fb86405e63a23af27 \
    "$root/logs/audits/g5_2d_sampling_full_20261018/patches.csv"
test -f "$root/logs/audits/train_full_20261018_job671208.json"
test -d /home/groups/comp3710/HipMRI_Study_open
test ! -e "$run_dir" || {
    printf '2D run directory already exists; inspect before submitting: %s\n' "$run_dir" >&2
    exit 1
}

test_job="$(sbatch --parsable --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=00:10:00 --job-name=g5-2d-check \
    --output="$root/logs/profiles/g5_2d_check_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B -m unittest tests.test_sampling_2d tests.test_training_recovery -v
python -B tools/train_g5_2d.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --config config/g5_2d_training_frozen.json \
    --run-dir ../logs/training/g5_2d_20261019 \
    --check-config-only
SBATCH
)"
test_job="${test_job%%;*}"
printf '2D CPU check: %s\n' "$test_job"

gpu_job="$(sbatch --parsable --dependency="afterok:$test_job" \
    --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=06:00:00 --job-name=g5-2d-40 \
    --output="$root/logs/profiles/g5_formal_2d_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/train_g5_2d.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --config config/g5_2d_training_frozen.json \
    --run-dir ../logs/training/g5_2d_20261019
SBATCH
)"
gpu_job="${gpu_job%%;*}"
printf '2D 40 epochs: %s\n' "$gpu_job"
squeue -j "$test_job,$gpu_job" -h -o '%i|%P|%T|%M|%l|%R'
