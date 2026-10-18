#!/usr/bin/env bash
# Submit one CPU recovery check, then both full 40-epoch A100 jobs.
set -euo pipefail

root="$HOME/20261018_g1_stage/20261017"
dataset=/home/groups/comp3710/HipMRI_Study_open

verify_sha() {
    local expected="$1" file="$2" actual
    test -f "$file" || { printf 'Missing file: %s\n' "$file" >&2; exit 1; }
    actual="$(sha256sum "$file")"
    actual="${actual%% *}"
    test "$actual" = "$expected" || {
        printf 'SHA-256 mismatch: %s\nexpected %s\nactual   %s\n' "$file" "$expected" "$actual" >&2
        exit 1
    }
}

verify_sha e5bd98a99213f4fe6cf6e3e8c6db3b95d5ef334a143a91444426d8ea019923ce \
    "$root/code/config/g5_controlled_training_frozen.json"
verify_sha 6a6c51bfa5a8979e1914ad6ac2879f08b7220b48ac59045f9b9d131f932966bb \
    "$root/code/tools/train_g5_cohort.py"
verify_sha 3b4e85dc364e3e194f63cf194e0f7162bc60e34be326fcc3b0783984794b7555 \
    "$root/code/tests/test_training_recovery.py"
verify_sha 952c493a2f52635225a35208cd664d9fd32db91fe7b01ed15c0f01437fe8f845 \
    "$root/code/hipmri_rewrite/training.py"
test -d "$dataset" || { printf 'Dataset unavailable: %s\n' "$dataset" >&2; exit 1; }
test -f "$root/logs/audits/train_full_20261018_job671208.json" || {
    printf 'Verified G1 audit is missing\n' >&2; exit 1;
}
test -d "$root/logs/profiles" || { printf 'Profile log directory is missing\n' >&2; exit 1; }
for arm in standard improved; do
    test ! -e "$root/logs/training/g5_${arm}_20261018" || {
        printf 'Run directory already exists for %s; inspect before resubmission\n' "$arm" >&2
        exit 1
    }
done

test_job="$(sbatch --parsable --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=00:10:00 \
    --job-name=g5-recovery-test \
    --output="$root/logs/profiles/g5_recovery_test_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B -m unittest tests.test_training_recovery -v
SBATCH
)"
test_job="${test_job%%;*}"
printf 'Recovery test: %s\n' "$test_job"

standard_job="$(sbatch --parsable --dependency="afterok:$test_job" \
    --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=06:00:00 --job-name=g5-standard-40 \
    --output="$root/logs/profiles/g5_formal_standard_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/train_g5_cohort.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --config config/g5_controlled_training_frozen.json \
    --model standard_3d \
    --run-dir ../logs/training/g5_standard_20261018
SBATCH
)"
standard_job="${standard_job%%;*}"
printf 'Standard 40 epochs: %s\n' "$standard_job"

improved_job="$(sbatch --parsable --dependency="afterok:$test_job" \
    --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=06:00:00 --job-name=g5-improved-40 \
    --output="$root/logs/profiles/g5_formal_improved_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/train_g5_cohort.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --config config/g5_controlled_training_frozen.json \
    --model improved_3d \
    --run-dir ../logs/training/g5_improved_20261018
SBATCH
)"
improved_job="${improved_job%%;*}"
printf 'Improved 40 epochs: %s\n' "$improved_job"
squeue -j "$test_job,$standard_job,$improved_job" -h -o '%i|%P|%T|%M|%l|%R'
