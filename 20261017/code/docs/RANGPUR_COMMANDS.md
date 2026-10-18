# Rangpur G2 command handoff

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
