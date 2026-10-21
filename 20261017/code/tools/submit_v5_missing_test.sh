#!/usr/bin/env bash
set -euo pipefail

root="$HOME/20261018_g1_stage/20261017"
script="$root/code/tools/evaluate_v5_missing.py"
output="$root/results/experiments/improved_3d_v5/comparisons/test_assessment_20261021"
python="$HOME/miniconda3/envs/torch/bin/python"
dataset="/home/groups/comp3710/HipMRI_Study_open"

test -s "$script"
test -d "$dataset"
test ! -e "$output" || { echo "Output exists: $output" >&2; exit 1; }
if squeue -u "$USER" -h -o '%j' | grep -Fxq v5-missing-test; then
    echo "v5-missing-test is already queued or running" >&2
    exit 1
fi
"$python" -B "$script" --dataset-root "$dataset" --output "$output" --preflight-only

job="$(sbatch --parsable \
    --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=03:00:00 --job-name=v5-missing-test \
    --output="$root/logs/profiles/v5_missing_test_%j.out" \
    --export=ALL,V5_SCRIPT="$script",V5_DATASET="$dataset",V5_OUTPUT="$output",V5_PYTHON="$python" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$V5_PYTHON" -B -u "$V5_SCRIPT" --dataset-root "$V5_DATASET" --output "$V5_OUTPUT"
SBATCH
)"
job="${job%%;*}"
printf 'Submitted v5 Test completion job: %s\n' "$job"
squeue -j "$job" -h -o '%i|%P|%T|%M|%l|%R'
