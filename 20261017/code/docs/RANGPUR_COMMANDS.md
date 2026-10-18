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

## G3 full Train run and local analysis

The two-case smoke run passed. On a Rangpur CPU node with `torch` active and
the working directory at `20261017/code`, run the full cohort without
`--limit`:

```bash
python -B tools/audit_g3_existing_pipeline.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --output ../logs/audits/g3_existing_pipeline_full_20261018_job671892.json
```

The completed job `671892` reported 143 audited cases, zero case errors and
`pipeline_pass=true`. Its JSON SHA-256 is
`319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9`.
After downloading that JSON into local `logs/audits/`, regenerate the derived
table and figure with `python -B code/tools/summarize_g3_pipeline.py` from an
existing Matplotlib environment. The tool reconciles raw counts with G1 and
rejects unexpected input hashes.

## G3 candidate-crop smoke run

From Windows PowerShell at `Final project`, transfer only the two new audit
sources to the existing staged project, then reconnect:

```powershell
scp '.\20261017\code\tools\audit_g3_candidate_sampling.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/audit_g3_candidate_sampling.py'
scp '.\20261017\code\config\g3_patch_audit.json' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/config/g3_patch_audit.json'
ssh s4906923@login0.compute.eait.uq.edu.au
```

On `login0`, verify both remote `sha256sum` values against local
`Get-FileHash`. The first interactive attempt (job `671962`) failed while
creating the `srun` job step, leaving the shell on `login0`; the subsequent
Python process was interrupted during OpenBLAS thread creation and produced
no valid smoke result. Submit the smoke audit as a CPU batch job instead:

```bash
sbatch --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:30:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/audits/g3_patch_candidate_smoke_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/audit_g3_candidate_sampling.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --limit 2 \
  --output-dir ../logs/audits/g3_patch_candidate_smoke_20261018
SBATCH
```

Read the returned job ID and inspect its `.out` file after `squeue -j ID`
shows the job has finished. The audit script itself rejects execution without
an allocated Slurm job or on a login host. Check the smoke manifest and patch
CSV before allocating time for all 143 cases. Keep the full run in a separate
new output directory.

The successful smoke job `671975` saved 40 patch rows across two Train cases.
Full CPU job `671984` saved 2,860 rows across all 143 Train cases under
`logs/audits/g3_patch_candidate_full_20261018/`. The manifest and patch CSV
SHA-256 values are respectively
`e142be157e692572a7fd96084ec80c241edc6f165498dec25ea13919abb1889b`
and `96ac1118b0109a21e3bc943df71094026dcf1852535d79124f200383d790dce2`.
After downloading both files into the matching local `logs/audits/` directory,
run `code/tools/summarize_g3_candidate_sampling.py` in the existing local
Matplotlib environment with those two hashes. The verifier checks all 2,860
rows against G3 before writing derived CSVs and a figure.

## G4 patient-balanced tensor dry run

The G4 source uses the already staged G1/G3 full JSON files. Transfer two new
files from Windows PowerShell at `Final project`, then connect to Rangpur:

```powershell
scp '.\20261017\code\hipmri_rewrite\sampling.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/hipmri_rewrite/sampling.py'
scp '.\20261017\code\tools\audit_g4_patient_sampling.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/audit_g4_patient_sampling.py'
ssh s4906923@login0.compute.eait.uq.edu.au
```

On login0, compare the remote `sha256sum` of both files with local
`Get-FileHash`. Submit a two-patient smoke run; the script refuses login-node
computation. Use a new output directory if this name already exists:

```bash
sbatch --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:20:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/audits/g4_patient_smoke_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/audit_g4_patient_sampling.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --limit-patients 2 \
  --output-dir ../logs/audits/g4_patient_smoke_20261018
SBATCH
```

Inspect the job output and `manifest.json`: expect two patients, ten patches
per policy per patient, and 40 CSV rows. This is a CPU *dry run*, not GPU
training or evidence of model benefit. A full patient-epoch run follows only
after reviewing the smoke output.

The two-patient smoke job `672082` passed. The full 26-patient CPU job
`672092` saved 520 model-ready patch records under
`logs/audits/g4_patient_full_20261018/`, 260 per policy. Its manifest and CSV
SHA-256 values are respectively
`cafd3593efa5880a4a61b6281bac5d177f58cd713e734cf8dad422d191f0b90c`
and `04a6822f15940b658f0c3eb062528aa4fa17721552030dc5dc6d45ce5c441968`.
The files were downloaded and verified locally. Run
`code/tools/summarize_g4_patient_sampling.py` from a local environment with
NumPy and Matplotlib to reproduce the G4 tables and SVG. This is still a
one-epoch dry run without optimizer steps.

## G5 patch GPU preflight

The Train-only probe source is `code/tools/probe_g5_patch_gpu.py`. It uses the
already transferred G4 sampler, one approved Train case, two 64×64×32 patches,
and the same provisional CE+foreground-Dice objective for both 3D models.
It records shape, loss, every parameter gradient, runtime and CUDA peak memory
without taking an optimizer step. From Windows PowerShell at `Final project`:

```powershell
scp '.\20261017\code\tools\probe_g5_patch_gpu.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/probe_g5_patch_gpu.py'
ssh s4906923@login0.compute.eait.uq.edu.au
```

Check the transferred SHA-256 before submitting. On 18 October 2026,
`scontrol show partition a100-test` reported nodes `a100-a,a100-b`, and
`sinfo -N -p a100-test -o '%N %G %m %t'` showed one `gpu:a100:1` GRES on each
idle node. The partition's default test time is ten minutes. Submit this
single-GPU, ten-minute probe from login0 after confirming the output path is
unused:

```bash
sbatch --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:10:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/profiles/g5_patch_probe_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/probe_g5_patch_gpu.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --case-id B006_Week0 --batch-size 2 \
  --output ../logs/profiles/g5_patch_probe_20261018.json
SBATCH
```

The `.out` file identifies the allocated node and both model measurements.
The JSON is a capacity record, not a model checkpoint. Do not run the CUDA
workload directly on login0.
