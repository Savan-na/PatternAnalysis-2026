#!/usr/bin/env bash
set -euo pipefail
root="$HOME/20261018_g1_stage/20261017"
cd "$root/code"
py="$HOME/miniconda3/envs/torch/bin/python"
"$py" -B -c 'import hashlib,json,pathlib; r=pathlib.Path.cwd(); d=json.loads((r/"config/all_model_audit_checksums.json").read_text()); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in d.items()), "Audit source mismatch"; print("Verified seven-model audit files")'
test ! -e "$root/results/experiments/improved_3d_v4/comprehensive_audit_20261021" || { echo 'Output exists; inspect before resubmission'; exit 1; }
if squeue -u "$USER" -h -o '%j' | grep -qE '^all-model-(audit|check)$'; then
    echo 'Audit already queued/running'; exit 1
fi
# Large historical weights have been backed up; reserve working room for diagnostics.
free=$(df -Pk "$root" | awk 'NR==2 {print $4}')
test "$free" -ge 262144 || { echo 'Less than 256 MiB free; reclaim space before evaluation'; exit 1; }
mkdir -p "$root/logs/profiles"
check=$(sbatch --parsable --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:10:00 --job-name=all-model-check --chdir="$root/code" --output="$root/logs/profiles/all_model_check_%j.out" --wrap="export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; $py -B -m unittest discover -s tests -p test_all_model_audit.py -v")
check="${check%%;*}"
job=$(sbatch --parsable --dependency="afterok:$check" --kill-on-invalid-dep=yes --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=01:00:00 --job-name=all-model-audit --chdir="$root/code" --output="$root/logs/profiles/all_model_audit_%j.out" --wrap="export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; $py -B -u tools/audit_all_models.py --dataset-root /home/groups/comp3710/HipMRI_Study_open")
printf 'CPU check: %s\nSeven-model fixed evaluation: %s\n' "$check" "${job%%;*}"
squeue -j "$check,${job%%;*}" -h -o '%i|%P|%T|%M|%l|%R'
