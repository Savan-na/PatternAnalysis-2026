#!/usr/bin/env bash
set -euo pipefail
root="$HOME/20261018_g1_stage/20261017"
cd "$root/code"
py="$HOME/miniconda3/envs/torch/bin/python"
mode="${1:-audit}"
case "$mode" in audit|train|resume-a0|resume-a1) ;; *) echo 'Use audit, train, resume-a0 or resume-a1'; exit 1;; esac
"$py" -B -c 'import hashlib,json,pathlib; r=pathlib.Path.cwd(); d=json.loads((r/"config/v4_transfer_checksums.json").read_text()); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in d.items()), "V4 source hash mismatch"; print("Verified v4 files")'
mkdir -p "$root/logs/profiles"
if squeue -u "$USER" -h -o '%j' | grep -qE '^v4-(audit|a0|a1)$'; then
    echo 'A v4 job is queued/running; inspect it before submitting more'; exit 1
fi
if [ "$mode" = audit ]; then
    test ! -e "$root/results/experiments/improved_3d_v4/augmentation_audit_20261020" || { echo 'Audit exists; inspect, do not overwrite'; exit 1; }
    job=$(sbatch --parsable --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 --job-name=v4-audit --chdir="$root/code" --output="$root/logs/profiles/v4_audit_%j.out" --wrap="export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; $py -B -m unittest discover -s tests -p test_v4_training.py -v && $py -B -u tools/audit_v4_augmentation.py --dataset-root /home/groups/comp3710/HipMRI_Study_open")
    echo "V4 Train-only augmentation audit: ${job%%;*}"
    exit 0
fi
"$py" -B -c 'import json,pathlib; p=pathlib.Path("../results/experiments/improved_3d_v4/augmentation_audit_20261020/audit.json"); d=json.loads(p.read_text()); assert d["eligible"] and all(d["gates"].values()), "Train augmentation audit did not pass"'
arms='a0 a1'
resume=''
if [ "$mode" != train ]; then arms="${mode#resume-}"; resume='--resume'; fi
for arm in $arms; do
    run="$root/logs/training/improved_3d_v4_${arm}_20261020"
    if [ "$mode" = train ]; then
        test ! -e "$run" || { echo "$arm run exists; use explicit resume"; exit 1; }
    else
        test -f "$run/manifest.json" || { echo 'Resume manifest missing'; exit 1; }
    fi
done
for arm in $arms; do
    run="$root/logs/training/improved_3d_v4_${arm}_20261020"
    job=$(sbatch --parsable --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=06:00:00 --job-name="v4-$arm" --chdir="$root/code" --output="$root/logs/profiles/v4_${arm}_%j.out" --wrap="export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; $py -B -u tools/train_v4_cohort.py --dataset-root /home/groups/comp3710/HipMRI_Study_open --config config/v4_${arm}_frozen.json --model improved_3d_v4_$arm --run-dir $run $resume")
    echo "V4 $arm 50 epochs: ${job%%;*}"
done
