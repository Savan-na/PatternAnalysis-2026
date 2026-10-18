# Rangpur G2 and G3 command handoff

Use Windows PowerShell at `Final project` for `scp` and `ssh`. Use the Rangpur
login shell only to check files and request a Slurm CPU node. Decode MRI arrays
only after `hostname` shows an allocated compute node. The active source is
`/home/groups/comp3710/HipMRI_Study_open`; the existing research stage is
`~/20261018_g1_stage/20261017`. The G1 record and its evidence are in the
[research report](../../results/reports/HipMRI_Research_Report.md).

## Transfer the G2 slice audit script

From Windows PowerShell:

```powershell
scp '.\20261017\code\tools\audit_g2_train_slices.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/audit_g2_train_slices.py'
ssh s4906923@login0.compute.eait.uq.edu.au
```

Compare the local `Get-FileHash -Algorithm SHA256` result with Rangpur
`sha256sum` before running the transferred file. The script also verifies the
complete G1 audit and the hash of every Train MRI/Mask file it processes.

## Two-case smoke run on a CPU node

From the Rangpur login shell:

```bash
sha256sum "$HOME/20261018_g1_stage/20261017/code/tools/audit_g2_train_slices.py"
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 --pty bash
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/audit_g2_train_slices.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --limit 2 \
  --output-dir ../logs/audits/g2_slice_smoke_20261018
```

The `--limit 2` output is a software check, not a full G2 result. Check its
row counts and per-case voxel totals against the G1 JSON before requesting a
longer CPU allocation for all 143 Train cases. The full output belongs in the
existing `results/data_analysis/g2_train_summary/` group; transfer only
approved derived CSVs to Windows for analysis. G2 makes no preprocessing,
sampling, loss, architecture, or model-training change.

## Full Train run after the smoke check

Starting from the smoke run's compute shell, exit to `login0` and request a
fresh one-hour CPU allocation:

```bash
exit
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --time=01:00:00 --pty bash
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
out_dir="$HOME/20261018_g1_stage/20261017/results/data_analysis/g2_train_summary"
if [ -e "$out_dir/per_slice_class_coverage.csv" ] || [ -e "$out_dir/per_volume_intensity_histogram.csv" ]; then
  echo "Full G2 CSV destination already exists; inspect it before rerunning"
  exit 1
fi
python -B tools/audit_g2_train_slices.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --output-dir "$out_dir"
wc -l "$out_dir/per_slice_class_coverage.csv" "$out_dir/per_volume_intensity_histogram.csv"
sha256sum "$out_dir/per_slice_class_coverage.csv" "$out_dir/per_volume_intensity_histogram.csv"
```

Stop if either destination CSV already exists. A complete run should create
18,320 slice records and 17,160 intensity-bin records, plus one header row in
each CSV. Transfer these two derived CSVs to the local folder of the same name
only after the full run passes; then reconcile class totals with G1 and add
plots and findings to the formal report.

## G3 existing-pipeline smoke run

The current restart implements downsampling and whole-volume model arrays;
it has no Patch Sampler to audit yet. Transfer the G3 script from Windows
PowerShell at `Final project`:

```powershell
scp '.\20261017\code\tools\audit_g3_existing_pipeline.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/audit_g3_existing_pipeline.py'
ssh s4906923@login0.compute.eait.uq.edu.au
```

On `login0`, compare its `sha256sum` with the local `Get-FileHash` output. Run
`srun` there; after allocation, continue the remaining commands on the CPU
node:

```bash
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 --pty bash
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/audit_g3_existing_pipeline.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --limit 2 \
  --output ../logs/audits/g3_existing_pipeline_smoke_20261018.json
```

The script refuses to overwrite an existing output and verifies the hashes of
the frozen split, G1 audit, reviewed preprocessing code and every source pair
it processes. A smoke result has `pipeline_pass=false` by design; check
`case_error_count` and per-case measurements before scheduling the full cohort.
