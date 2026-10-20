#!/usr/bin/env bash
set -euo pipefail
root="$HOME/20261018_g1_stage/20261017"
cd "$root/code"
py="$HOME/miniconda3/envs/torch/bin/python"
"$py" -B -c 'import hashlib,json,pathlib; r=pathlib.Path.cwd(); d=json.loads((r/"config/best_checkpoint_transfer_checksums.json").read_text()); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in d.items()), "Audit source mismatch"; print("Verified fixed audit package")'
test ! -e "$root/results/experiments/improved_3d_v3/best_checkpoint_audit_20261020" || {
    echo 'Audit output exists; inspect before any resubmission'; exit 1;
}
if squeue -u "$USER" -h -o '%j' | grep -qx 'best-checkpoint-audit'; then
    echo 'Audit already queued/running'; exit 1
fi
mkdir -p "$root/logs/profiles"
check=$(sbatch --parsable --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 \
    --time=00:10:00 --job-name=best-audit-check --chdir="$root/code" \
    --output="$root/logs/profiles/best_checkpoint_audit_check_%j.out" \
    --wrap="export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; $py -B -m unittest discover -s tests -p test_best_checkpoint_audit.py -v")
check="${check%%;*}"
job=$(sbatch --parsable --dependency="afterok:$check" --kill-on-invalid-dep=yes \
    --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 \
    --time=00:45:00 --job-name=best-checkpoint-audit --chdir="$root/code" \
    --output="$root/logs/profiles/best_checkpoint_audit_%j.out" \
    --wrap="export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; $py -B -u tools/audit_best_checkpoints.py --dataset-root /home/groups/comp3710/HipMRI_Study_open")
job="${job%%;*}"
printf 'CPU diagnostic tests: %s\nFixed Validation/Test audit (no training): %s\n' "$check" "$job"
squeue -j "$check,$job" -h -o '%i|%P|%T|%M|%l|%R'
