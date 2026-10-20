#!/usr/bin/env bash
# One isolated v2 run; optional explicit resume, never replace historical runs.
set -euo pipefail
root="$HOME/20261018_g1_stage/20261017"
mode="${1:-new}"
case "$mode" in new|resume) ;; *) echo 'Use new or resume'; exit 1;; esac
cd "$root/code"
python_bin="$HOME/miniconda3/envs/torch/bin/python"
"$python_bin" -B -c 'import hashlib,json,pathlib; r=pathlib.Path.cwd(); d=json.loads((r/"config/v2_transfer_checksums.json").read_text()); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in d.items()), "v2 source SHA mismatch"; print("Verified v2 files")'
run="$root/logs/training/improved_3d_v2_ds_20261020"
if [ "$mode" = new ]; then
    test ! -e "$run" || { echo 'Run exists: inspect before using resume'; exit 1; }
else
    test -f "$run/manifest.json" || { echo 'Resume manifest missing'; exit 1; }
fi
if squeue -u "$USER" -h -o '%j' | grep -qx 'v2-ds-50'; then
    echo 'A v2 training job is already queued or running'; exit 1
fi
mkdir -p "$root/logs/profiles"
check=$(sbatch --parsable --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 \
    --time=00:10:00 --job-name=v2-ds-check --chdir="$root/code" \
    --output="$root/logs/profiles/v2_check_%j.out" \
    --wrap="export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; $python_bin -B -m unittest discover -s tests -p test_improved_v2.py -v")
check="${check%%;*}"
resume=''
if [ "$mode" = resume ]; then resume='--resume'; fi
job=$(sbatch --parsable --dependency="afterok:$check" --kill-on-invalid-dep=yes \
    --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 \
    --time=06:00:00 --job-name=v2-ds-50 --chdir="$root/code" \
    --output="$root/logs/profiles/v2_training_%j.out" \
    --wrap="export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; $python_bin -B -u tools/train_v2_cohort.py --dataset-root /home/groups/comp3710/HipMRI_Study_open --config config/v2_independent_supervision_frozen.json --model improved_3d_v2 --run-dir $run $resume")
job="${job%%;*}"
printf 'v2 CPU checks: %s\nv2 50 epochs: %s\n' "$check" "$job"
squeue -j "$check,$job" -h -o '%i|%P|%T|%M|%l|%R'
