#!/bin/bash
# Fixed four-case diagnostic: no model training and no Test evaluation.
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python="$HOME/miniconda3/envs/torch/bin/python"
run=p2_four_cases_20261020
test -x "$python"
for path in "$project/logs/validation/inference_context/$run" "$project/results/diagnostics/inference_context/$run"; do
  if [ -e "$path" ]; then
    echo "Output exists; inspect before submitting: $path" >&2
    exit 1
  fi
done
if [ -n "$(squeue -u "$USER" -h -n p2-context)" ]; then
  echo 'A P2 diagnostic is already queued or running' >&2
  exit 1
fi
mkdir -p "$project/logs/profiles"
job=$(sbatch --parsable --partition=a100-test --gres=gpu:a100:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:10:00 \
  --job-name=p2-context --chdir="$project/code" \
  --output="$project/logs/profiles/p2_inference_context_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python="$HOME/miniconda3/envs/torch/bin/python"
hostname
"$python" -B -m unittest discover -s tests -p test_inference_context.py -v
"$python" -B -u tools/audit_p2_inference_context.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open
SBATCH
)
echo "P2 fixed-weight Validation diagnostic: $job"
squeue -j "$job" -h -o '%i|%P|%T|%M|%l|%R'
