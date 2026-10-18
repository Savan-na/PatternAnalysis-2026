#!/usr/bin/env bash
# Queue both frozen 3D arms to continue from completed epoch 40 through epoch 50.
set -euo pipefail

root="$HOME/20261018_g1_stage/20261017"
dataset=/home/groups/comp3710/HipMRI_Study_open
code="$root/code"
test "$(sha256sum "$code/config/g5_controlled_training_frozen.json" | cut -d' ' -f1)" = \
    e5bd98a99213f4fe6cf6e3e8c6db3b95d5ef334a143a91444426d8ea019923ce
test "$(sha256sum "$code/tools/train_g5_cohort.py" | cut -d' ' -f1)" = \
    6a6c51bfa5a8979e1914ad6ac2879f08b7220b48ac59045f9b9d131f932966bb
test "$(sha256sum "$code/tools/review_g5_validation_failures.py" | cut -d' ' -f1)" = \
    1bac7b2e16a631731a840df064a2cee347d25b2daf6045346d12dcc43d1ed249
test -d "$dataset"
test -f "$root/logs/audits/train_full_20261018_job671208.json"
for arm in standard improved; do
    run="$root/logs/training/g5_${arm}_20261018"
    for name in manifest.json epoch_040.json epoch_040.pt latest.pt best.pt; do
        test -s "$run/$name" || { echo "Missing $run/$name" >&2; exit 1; }
    done
done

"$HOME/miniconda3/envs/torch/bin/python" -B - "$root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for arm, model, job in (("standard", "standard_3d", "672954"),
                        ("improved", "improved_3d", "672955")):
    run = root / "logs/training" / f"g5_{arm}_20261018"
    manifest = json.loads((run / "manifest.json").read_text())
    record = json.loads((run / "epoch_040.json").read_text())
    if (manifest["model"] != model or manifest["job_id_initial"] != job
            or manifest["max_epochs"] != 40
            or record["epoch"] != 40 or record["best_epoch"] != 30
            or record["validation"] is None):
        raise SystemExit(f"{arm}: completed 40-epoch run identity differs")
    snapshots = sorted(path.name for path in run.glob("epoch_*.pt"))
    if snapshots != [f"epoch_{index:03d}.pt" for index in range(1, 41)]:
        raise SystemExit(f"{arm}: immutable epoch snapshots are incomplete")
print("Both frozen runs have 40 complete epoch snapshots and selected checkpoints")
PY

if squeue -u "$USER" -h -o '%j' | grep -Eq '^g5-(std|imp)-50$'; then
    echo "An epoch-50 extension is already queued or running" >&2
    exit 1
fi

standard_job="$(sbatch --parsable --partition=a100 --gres=gpu:a100:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --time=02:00:00 \
    --job-name=g5-std-50 \
    --output="$root/logs/profiles/g5_extend_standard_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/train_g5_cohort.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --config config/g5_controlled_training_frozen.json \
    --model standard_3d --run-dir ../logs/training/g5_standard_20261018 \
    --resume --extend-to-epochs 50
SBATCH
)"
standard_job="${standard_job%%;*}"
printf 'Standard epochs 41-50: %s\n' "$standard_job"

improved_job="$(sbatch --parsable --partition=a100 --gres=gpu:a100:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --time=02:00:00 \
    --job-name=g5-imp-50 \
    --output="$root/logs/profiles/g5_extend_improved_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/train_g5_cohort.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --config config/g5_controlled_training_frozen.json \
    --model improved_3d --run-dir ../logs/training/g5_improved_20261018 \
    --resume --extend-to-epochs 50
SBATCH
)"
improved_job="${improved_job%%;*}"
printf 'Improved epochs 41-50: %s\n' "$improved_job"

review_job="$(sbatch --parsable --dependency="afterok:$standard_job:$improved_job" \
    --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=00:30:00 --job-name=g5-val-panels \
    --output="$root/logs/profiles/g5_validation_panels_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/review_g5_validation_failures.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --standard-run-dir ../logs/training/g5_standard_20261018 \
    --improved-run-dir ../logs/training/g5_improved_20261018 \
    --output-dir ../results/case_studies/comparisons
SBATCH
)"
review_job="${review_job%%;*}"
printf 'Validation case panels after both runs: %s\n' "$review_job"
squeue -j "$standard_job,$improved_job,$review_job" -h -o '%i|%P|%T|%M|%l|%R'
