#!/usr/bin/env bash
# Queue one held-out Test evaluation only after both 3D continuations succeed.
set -euo pipefail

root="$HOME/20261018_g1_stage/20261017"
script="$root/code/tools/evaluate_g6_test.py"
output="$root/results/tables/comparisons/g6_test_3d.json"
test "$(sha256sum "$script" | cut -d' ' -f1)" = \
    1614e0374f30b1272a0978d9402d3849bc86f044854939c7f70b2fe4aa5d3404
test "$(sha256sum "$root/code/config/g5_controlled_training_frozen.json" | cut -d' ' -f1)" = \
    e5bd98a99213f4fe6cf6e3e8c6db3b95d5ef334a143a91444426d8ea019923ce
test ! -e "$output" || { echo "Held-out Test result already exists: $output" >&2; exit 1; }
if squeue -u "$USER" -h -o '%j' | grep -Fxq g6-test-3d; then
    echo "Held-out Test evaluation is already queued or running" >&2
    exit 1
fi

job="$(sbatch --parsable --dependency=afterok:674102:674103 \
    --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=02:00:00 --job-name=g6-test-3d \
    --output="$root/logs/profiles/g6_test_3d_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/evaluate_g6_test.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --output ../results/tables/comparisons/g6_test_3d.json
SBATCH
)"
job="${job%%;*}"
printf 'One-time 3D held-out Test evaluation: %s\n' "$job"
squeue -j "$job" -h -o '%i|%P|%T|%M|%l|%R'
