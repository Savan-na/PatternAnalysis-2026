#!/bin/bash
# Submit the Train-only CPU dry run; this never submits model training.
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:-smoke}
case "$mode" in
  smoke) wall=00:20:00 ;;
  full) wall=00:45:00 ;;
  *) echo 'Use: bash submit_g7_edge_audit.sh smoke|full' >&2; exit 1 ;;
esac
python="$HOME/miniconda3/envs/torch/bin/python"
output="$project/logs/audits/g7_edge_sampling_${mode}_20261019"
jobname="g7-edge-$mode"
test -x "$python"
test -d "$project/logs/audits"
if [ -e "$output" ]; then
  echo "Output already exists; inspect before resubmission: $output" >&2
  exit 1
fi
if [ -n "$(squeue -u "$USER" -h -n "$jobname")" ]; then
  echo "A $jobname job is already queued or running" >&2
  exit 1
fi
if [ "$mode" = full ]; then
  "$python" -B - "$project" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads((p / 'logs/audits/g7_edge_sampling_smoke_20261019/manifest.json').read_text())
if not (d['smoke_pass'] and d['epochs'] == 2 and d['patients'] == 2):
    raise SystemExit('Successful two-patient/two-epoch smoke is required')
for key, relative in (
    ('script_sha256', 'code/tools/audit_g7_edge_sampling.py'),
    ('edge_sampling_sha256', 'code/hipmri_rewrite/sampling_edges.py'),
    ('candidate_config_sha256', 'code/config/g7_edge_sampling_candidate.json'),
    ('frozen_sampling_sha256', 'code/hipmri_rewrite/sampling.py'),
    ('data_sha256', 'code/hipmri_rewrite/data.py'),
):
    if hashlib.sha256((p / relative).read_bytes()).hexdigest() != d['sources'][key]:
        raise SystemExit(f'Source changed after smoke: {relative}')
print('Verified smoke and unchanged candidate sources')
PY
fi
job=$(sbatch --parsable --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --time="$wall" --job-name="$jobname" --chdir="$project/code" \
  --export="ALL,G7_STAGE=$project,G7_MODE=$mode" \
  --output="$project/logs/audits/g7_edge_sampling_${mode}_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
python="$HOME/miniconda3/envs/torch/bin/python"
"$python" -B -m unittest discover -s tests -p test_sampling_edges.py -v
args=()
if [ "$G7_MODE" = smoke ]; then
  args=(--epochs 2 --limit-patients 2)
fi
"$python" -B -u tools/audit_g7_edge_sampling.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --output-dir "$G7_STAGE/logs/audits/g7_edge_sampling_${G7_MODE}_20261019" "${args[@]}"
SBATCH
)
echo "G7 Train-only CPU $mode audit: $job"
squeue -j "$job" -h -o '%i|%P|%T|%M|%l|%R'
