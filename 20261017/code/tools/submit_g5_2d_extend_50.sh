#!/usr/bin/env bash
# Resume the completed frozen 2D run for a bounded Validation-only 41–50 check.
set -euo pipefail

root="$HOME/20261018_g1_stage/20261017"
run="$root/logs/training/g5_2d_20261019"
code="$root/code"
dataset=/home/groups/comp3710/HipMRI_Study_open

verify_sha() {
    local expected="$1" file="$2" actual
    test -s "$file" || { echo "Missing $file" >&2; exit 1; }
    actual="$(sha256sum "$file")"
    actual="${actual%% *}"
    test "$actual" = "$expected" || {
        echo "Source hash differs: $file" >&2
        exit 1
    }
}

verify_sha e308af1271885fca84b7007114572a6f00b40ceb88e1298840b65ce94df3ce96 \
    "$code/tools/train_g5_2d.py"
verify_sha 3ae7717569b6434d250b6b906c60e62fac23670e400341153b837aaf322f0ec2 \
    "$code/config/g5_2d_training_frozen.json"
test -d "$dataset"
test -s "$run/manifest.json"
test -s "$run/epoch_040.pt"
test -s "$run/epoch_040.json"
test -s "$run/latest.pt"
test -s "$run/best.pt"

"$HOME/miniconda3/envs/torch/bin/python" -B - "$run" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
manifest = json.loads((run / "manifest.json").read_text())
record = json.loads((run / "epoch_040.json").read_text())
snapshots = sorted(path.name for path in run.glob("epoch_*.pt"))
records = sorted(path.name for path in run.glob("epoch_*.json"))
expected_pt = [f"epoch_{epoch:03d}.pt" for epoch in range(1, 41)]
expected_json = [f"epoch_{epoch:03d}.json" for epoch in range(1, 41)]
if (manifest["kind"] != "G5_frozen_2D_Train_Validation_cohort_run"
        or manifest["model"] != "baseline_2d"
        or manifest["job_id_initial"] != "673908"
        or manifest["max_epochs"] != 40
        or record["epoch"] != 40
        or record["validation"] is None
        or record["best_epoch"] != 40
        or snapshots != expected_pt
        or records != expected_json):
    raise SystemExit("2D run is not the expected complete 40-epoch state")
print("Verified 2D run: 40 epoch weights and records; best epoch 40")
PY

if squeue -u "$USER" -h -o '%j' | grep -Eq '^g5-2d-50$'; then
    echo "2D continuation already queued or running" >&2
    exit 1
fi

job="$(sbatch --parsable --partition=a100 --gres=gpu:a100:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --time=02:00:00 \
    --job-name=g5-2d-50 \
    --output="$root/logs/profiles/g5_extend_2d_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/train_g5_2d.py \
    --dataset-root /home/groups/comp3710/HipMRI_Study_open \
    --config config/g5_2d_training_frozen.json \
    --run-dir ../logs/training/g5_2d_20261019 \
    --resume --extend-to-epochs 50
SBATCH
)"
job="${job%%;*}"
printf '2D epochs 41–50: %s\n' "$job"
squeue -j "$job" -h -o '%i|%P|%T|%M|%l|%R'
