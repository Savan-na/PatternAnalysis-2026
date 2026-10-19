#!/bin/bash
# One bounded Validation diagnostic; no training or Test evaluation.
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python="$HOME/miniconda3/envs/torch/bin/python"
test -x "$python"
for base in logs/validation results/diagnostics; do
  if [ -e "$project/$base/inference_context/p2b_prediction_mechanisms_20261020" ]; then
    echo 'P2B output exists; inspect partial/completed results before resubmission' >&2
    exit 1
  fi
done
if [ -n "$(squeue -u "$USER" -h -n p2b-mechanisms)" ]; then
  echo 'P2B is already queued or running' >&2
  exit 1
fi
job=$(sbatch --parsable --partition=a100-test --gres=gpu:a100:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:10:00 \
  --job-name=p2b-mechanisms --chdir="$project/code" \
  --output="$project/logs/profiles/p2b_mechanisms_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python="$HOME/miniconda3/envs/torch/bin/python"
hostname
"$python" -B -m unittest discover -s tests -p test_prediction_diagnostics.py -v
"$python" -B -u tools/audit_p2b_prediction_mechanisms.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open
SBATCH
)
echo "P2B fixed-weight mechanism diagnostic: $job"
squeue -j "$job" -h -o '%i|%P|%T|%M|%l|%R'
