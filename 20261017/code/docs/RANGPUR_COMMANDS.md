# Rangpur G2 and G3 command handoff

This is the historical command record for the 2026 restart. G1–G6 have since
completed; check the [formal research report](../../results/reports/HipMRI_Research_Report.md)
for current evidence before submitting any job. Do not rerun the one-time Test
evaluation or old training commands as a continuation.

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

GPU probe job `672173` completed on `a100-a`. The JSON at
`logs/profiles/g5_patch_probe_20261018.json` has SHA-256
`f8d2a059d31f5d50302399dd8e92d1218c6159352775545f5f02b38ceee651eb`.
Both 3D models produced finite losses and gradients at batch size two; see
the formal research report for interpretation. Their first-run timings are
not a controlled speed comparison.

## G5 two-patch learning check

Transfer `code/tools/tiny_overfit_g5.py` from Windows PowerShell at
`Final project` to the existing staged Rangpur code directory, then confirm
its SHA-256. The check takes optimizer steps on two fixed *Train* patches and
saves metrics only, not weights. After verifying the output path is unused,
submit on the same A100 test partition:

```bash
sbatch --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:10:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/profiles/g5_tiny_overfit_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/tiny_overfit_g5.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --case-id B006_Week0 --steps 100 \
  --output ../logs/profiles/g5_tiny_overfit_20261018.json
SBATCH
```

Review initial/final fixed-batch loss, Rectum and Prostate Dice/recall and
predicted voxel counts before planning a full training run. The result shows
only whether these two patches are learnable under the temporary objective.

Job `672194` completed 100 steps per model. Its JSON SHA-256 is
`097b047bd9d2a7c7f6c713cbbb98ba42558a2577f661aed744cfc8feba43c5ad`.
The Standard arm predicted only Body at step 100; Improved fitted all three
small organs on these fixed patches. To check whether Standard is simply
slower, transfer the revised `tiny_overfit_g5.py` (SHA-256
`ae9c851a79773e84045812c0ccebd0eb176469d3342c02eff2207b2066990045`)
and verify its hash,
then run only Standard for 500 steps with the same case, patches, seed,
objective and optimizer. Use a new output path:

```bash
sbatch --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:10:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/profiles/g5_standard_500_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/tiny_overfit_g5.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --case-id B006_Week0 --model standard_3d --steps 500 \
  --output ../logs/profiles/g5_standard_500_20261018.json
SBATCH
```

This changes the step budget only and does not replace the original 100-step
record or create a checkpoint. Its diagnostic purpose is limited to fixed
Train patches; full training remains gated on reviewing the result.

Job `672228` completed. The returned JSON SHA-256 is
`d10c454a24f1bdb1876fec034784a53fbb4d0a5c0da6fcfdcdc0dee75a72c53f`.
At step 500, Standard still predicted zero Bladder, Rectum and Prostate voxels
on the fixed batch. See the research report for the class counts and limits.

## G5 controlled weighted-CE diagnostic

The next check keeps the Standard model, two Train patches, seed, optimizer,
500 steps and the 0.5 CE / 0.5 foreground Dice mixture fixed. Only the CE
class weights change. `code/config/g5_weighted_ce_diagnostic.json` records the
weights and their Train-only G4 patient-uniform exposure source. This is a
learning diagnostic, not an approved full-training objective.

From **Windows PowerShell** in `Final project`, transfer the updated script
(SHA-256 `393dff3896fc8949ffd0feb9433163fbe8d9bc3d70401e7d9d87bbfce05a404c`)
and new config (SHA-256
`4daa7f5b202e6446a31a5780247ff2e93c3e79390c61137621ecdb097c4e85f`):

```powershell
scp '.\20261017\code\tools\tiny_overfit_g5.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/tiny_overfit_g5.py'
scp '.\20261017\code\config\g5_weighted_ce_diagnostic.json' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/config/g5_weighted_ce_diagnostic.json'
ssh s4906923@login0.compute.eait.uq.edu.au
```

On the **Rangpur login node**, verify the two transferred hashes, check the
result path is free, then submit the small GPU job:

```bash
sha256sum "$HOME/20261018_g1_stage/20261017/code/tools/tiny_overfit_g5.py" \
  "$HOME/20261018_g1_stage/20261017/code/config/g5_weighted_ce_diagnostic.json"
test ! -e "$HOME/20261018_g1_stage/20261017/logs/profiles/g5_standard_weighted_500_20261018.json" && echo 'Weighted diagnostic output path is free'
sbatch --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:10:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/profiles/g5_standard_weighted_500_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/tiny_overfit_g5.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --case-id B006_Week0 --model standard_3d --steps 500 \
  --class-weights-config config/g5_weighted_ce_diagnostic.json \
  --output ../logs/profiles/g5_standard_weighted_500_20261018.json
SBATCH
```

After completion, report the job ID, final ten output lines and JSON SHA-256.
Compare the final Bladder, Rectum and Prostate predicted voxel counts with the
unweighted 500-step JSON before deciding any further experiment.

Job `672730` completed and its returned JSON SHA-256 matched Rangpur:
`f9f53f6eaaabb922f29e04d4905d5529ad24c668b3b2733620c9de340008b7c7`.
The comparison and its limits are recorded in the formal research report.

## G5 full-volume inference capacity gate

Before freezing cohort training and Validation schedules, check that both
current 3D networks can infer an entire preprocessed volume on one allocated
A100. The Train case `K019_Week1` is the sole 256×256×144 raw Train MRI in the
G1 inventory, so it covers the largest observed Z extent. The script uses
random weights and measures geometry, execution time and peak memory only;
its predictions are not segmentation evidence. It reads the frozen G1 hash
and refuses changed source files.

From **Windows PowerShell** in `Final project`, transfer the source (SHA-256
`d47d4a4508f4ee7236927444696384bb25b8cbdc5a7bb6b55b939b4ac71e7040`):

```powershell
scp '.\20261017\code\tools\probe_g5_full_volume.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/probe_g5_full_volume.py'
ssh s4906923@login0.compute.eait.uq.edu.au
```

On the **Rangpur login node**, verify the hash and empty destination before
submitting the short GPU job:

```bash
sha256sum "$HOME/20261018_g1_stage/20261017/code/tools/probe_g5_full_volume.py"
test ! -e "$HOME/20261018_g1_stage/20261017/logs/profiles/g5_full_volume_probe_20261018.json" && echo 'Full-volume probe output path is free'
sbatch --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:10:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/profiles/g5_full_volume_probe_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/probe_g5_full_volume.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --case-id K019_Week1 --repeats 2 \
  --output ../logs/profiles/g5_full_volume_probe_20261018.json
SBATCH
```

Use the job ID printed by `sbatch` to inspect the `.out` file. Once the JSON
exists, send its SHA-256 and the two model rows. A pass supports selecting a
whole-volume Validation inference strategy; it does not approve a training
budget or show model accuracy.

Job `672782` completed; its JSON SHA-256 matched Rangpur:
`263c2309e00d21666e69418a8c82d436585e3037285c0e02f17f959488b21c53`.
Both model outputs had the expected XYZ shape. See the formal report for
measured capacity and interpretation.

## G5 paired training-candidate smoke

The candidate protocol is `code/config/g5_controlled_training_candidate.json`.
Before a 40-epoch cohort run, use `code/tools/probe_g5_training_smoke.py` to
take exactly two weighted optimizer steps per model. It selects two distinct
Train patients under the G4 visit rule, draws two uniform patches per patient,
mixes both patients in each batch, and records delivered-mask class exposure,
loss, gradient norm, prediction counts, time and CUDA memory. No checkpoint
or Validation metric is produced. It verifies the wiring and initial runtime,
not learnability or generalization.

From **Windows PowerShell** in `Final project`, upload the two new files:

```powershell
scp '.\20261017\code\tools\probe_g5_training_smoke.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/probe_g5_training_smoke.py'
scp '.\20261017\code\config\g5_controlled_training_candidate.json' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/config/g5_controlled_training_candidate.json'
ssh s4906923@login0.compute.eait.uq.edu.au
```

On the **Rangpur login node**, verify the script SHA-256
`8ee48773309e8240557fbd860ab9cec69def4776a4d9e16c3f40fe168aaf9a2a`
and configuration SHA-256
`aafdcc4e695de24fec9c6748db3448287b9e5040d036bd7eb4422776521f0ee7`.
If the result path is free, submit the bounded A100 test job:

```bash
sha256sum "$HOME/20261018_g1_stage/20261017/code/tools/probe_g5_training_smoke.py" \
  "$HOME/20261018_g1_stage/20261017/code/config/g5_controlled_training_candidate.json"
test ! -e "$HOME/20261018_g1_stage/20261017/logs/profiles/g5_training_smoke_20261018.json" && echo 'Training smoke output path is free'
sbatch --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:10:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/profiles/g5_training_smoke_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/probe_g5_training_smoke.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --output ../logs/profiles/g5_training_smoke_20261018.json
SBATCH
```

The job should print two successful model rows and a JSON path. Transfer only
the JSON after SHA-256 verification; do not copy source MRI or masks.

Job `672807` completed. The returned JSON matched Rangpur SHA-256
`4161b30138243dbf3f52ba44cb097aead6046e50d60e37b60d64329bf6a336d5`.
Two mixed-patient batches per model passed; the formal report records the
exposure and interpretation. Full-epoch I/O, Validation and checkpoints
remain unchecked.

## G5 one-epoch budget and exposure probe

The revised `probe_g5_training_smoke.py` adds `--full-epoch`. It runs epoch
zero of the candidate Train sampler across all 26 patients, 10 uniform
patches per patient, batch size two, for 130 temporary optimizer updates
per model. It measures NIfTI/hash/preprocessing/patch preparation separately
from each model loop and records every delivered mask. It saves **no** model
checkpoint and does not inspect Validation or Test. This is the remaining
runtime/exposure check before freezing the cohort training protocol.

From **Windows PowerShell** in `Final project`, transfer the revised script
(SHA-256 `6d89a7055afad98cc1288db212b8c9dd12fd5d65deb6968264ca441160d5a4fb`):

```powershell
scp '.\20261017\code\tools\probe_g5_training_smoke.py' 's4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_stage/20261017/code/tools/probe_g5_training_smoke.py'
ssh s4906923@login0.compute.eait.uq.edu.au
```

On the **Rangpur login node**, verify the hash, confirm the regular A100
partition is available, and check the fresh JSON path before submitting:

```bash
sha256sum "$HOME/20261018_g1_stage/20261017/code/tools/probe_g5_training_smoke.py"
scontrol show partition a100 | head -n 5
test ! -e "$HOME/20261018_g1_stage/20261017/logs/profiles/g5_epoch_budget_20261018.json" && echo 'Epoch probe output path is free'
sbatch --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:30:00 \
  --output="$HOME/20261018_g1_stage/20261017/logs/profiles/g5_epoch_budget_%j.out" <<'SBATCH'
#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/20261018_g1_stage/20261017/code"
python -B tools/probe_g5_training_smoke.py \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --full-epoch --output ../logs/profiles/g5_epoch_budget_20261018.json
SBATCH
```

Check the job log for 130 batches per model and the JSON SHA-256. Its first
epoch loss cannot be used as a Validation result or an optimal epoch count.

## G5 formal training transition

Job 672837 was still PENDING when the protocol was frozen. Its no-checkpoint
full-epoch probe has been superseded by a first formal epoch that saves
`latest.pt`. Cancel job 672837 while it is pending and confirm it has left the
queue before starting a formal job. The absence of a Slurm `.out` file and
JSON while PENDING is expected.

The executable protocol is now
`code/config/g5_controlled_training_frozen.json`. Transfer this file,
`code/tools/train_g5_cohort.py`, and `code/hipmri_rewrite/training.py` to their
matching paths under `~/20261018_g1_stage/20261017/`. Verify their SHA-256
hashes against local files. The first formal command is the trainer with
`--model standard_3d`, `--stop-after-epoch 1`, and a new directory under
`logs/training/`. It saves model and optimizer weights plus a Train epoch log.
Run the Improved arm with the same config and its own new directory; resume
both with `--resume` to the frozen budget. Validation first occurs at epoch 5,
and Test remains untouched during training and selection.

The pending one-epoch jobs 672872 and 672886 were subsequently cancelled.
The revised frozen configuration requires one atomic recovery checkpoint per
completed optimizer batch and an immutable snapshot for each epoch. For a new
40-epoch job, omit `--stop-after-epoch`; `--resume` continues from the latest
saved batch if a Slurm time limit interrupts it. Verify the new script/config
hashes before submission. Request a walltime allowed by both partition and
QoS; `MaxTime=UNLIMITED` at the partition does not prove a particular job can
run forever or start promptly.

The reviewed submission entry point is `code/tools/submit_g5_40.sh`. Run it
only on the Rangpur login node after transferring the revised config,
trainer, recovery test and the script itself. It verifies the source hashes
and unused output directories, submits a short CPU checkpoint recovery test,
then submits both six-hour, 40-epoch A100 jobs with `afterok` dependencies on
that test. If the test fails, the GPU jobs do not start. The script prints all
three job IDs. It does not submit the optional 2D arm.

On 18 October, recovery test 672953 cleared the dependency for A100 jobs
672954 (Standard) and 672955 (Improved). Both remained PENDING for priority
at the queue observation. Both later completed 40 epochs with exit code 0;
their logs and checkpoints must be reviewed before comparing per-class
Validation or opening held-out Test data.

## Optional 2D arm: Train-only exposure gate

The two 2D source files have been transferred to Rangpur. Their local
SHA-256 values are `b98fbb0db5936a6260bc1d1f74b1446411e4ba8e1a9584ca715fb48d6a1624f3`
for `code/tools/audit_g5_2d_sampling.py` and
`99054855ecd2869796af9a4ce5decbdb2522d5402dac0fa7b4782676aa3d9f08`
for `code/hipmri_rewrite/sampling_2d.py`. Run the audit in a separate CPU
Slurm job while the 3D jobs queue or train. It compares the two candidate
policies across one visit from each of the 26 Train patients, yielding 260
crops per policy. Job 673872 completed and its `manifest.json` and
`patches.csv` passed local hash and row-count checks. Organ-aware sampling
raised small-organ patch coverage, so the separate 2D protocol is frozen in
`code/config/g5_2d_training_frozen.json`; it cannot serve as a segmentation
quality score. Transfer that config, `code/tools/train_g5_2d.py` and
`code/tools/submit_g5_2d.sh` to their matching Rangpur staging paths. The
submission script verifies hashes, runs a CPU config/recovery check, then
queues the optional 40-epoch A100 job on success. It does not modify the
completed 3D runs. The 2D run's Validation score is separate from the
architecture-controlled 3D comparison.

The first 2D CPU check (673901) failed because the staged checkout lacked
`tests/test_sampling_2d.py`; its dependent GPU job 673902 never started and
was cancelled. After transferring the missing test, CPU check 673907 passed
and formal A100 job 673908 started. The first epoch completed with durable
batch and epoch records. This is an ongoing training run, not a completed
Validation or Test result.
