# Rangpur G1 commands and terminal boundaries

The user's initial prompt is Windows PowerShell at `Final project`. Passwords
belong only in the terminal. On 18 October 2026, read-only checks confirmed
that `/home/groups/comp3710/HipMRI_Study_open` exists with `semantic_MRs` and
`semantic_labels_only`. The old `~/20261017_g1_stage` and
`~/20261017_source.zip` also exist and must remain untouched. Directory
existence does not yet establish the current file count or data integrity.

## 1. Read-only location check

From Windows PowerShell:

```powershell
ssh s4906923@login0.compute.eait.uq.edu.au
```

After the prompt becomes a Rangpur Linux prompt:

```bash
pwd
hostname
ls -ld /home/groups/comp3710 /home/groups/comp3710/HipMRI_Study_open 2>&1
ls -ld ~/20261017 ~/20261017_g1_stage 2>&1
```

These are small metadata checks on the login node. Confirm the active data
root, available project storage, Python environment, and current COMP3710
Slurm CPU allocation syntax before transfer or array decoding. Do not run
full NIfTI decoding on the shared login node.

## 2. Source transfer after checking the new names are unused

On the Rangpur login shell, confirm both new names are absent and inventory
file counts without opening NIfTI arrays:

```bash
for p in "$HOME/20261018_g1_source.zip" "$HOME/20261018_g1_stage"; do if [ -e "$p" ]; then echo "EXISTS: $p"; else echo "FREE: $p"; fi; done
find /home/groups/comp3710/HipMRI_Study_open/semantic_MRs -maxdepth 1 -type f -name '*.nii.gz' | wc -l
find /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only -maxdepth 1 -type f -name '*.nii.gz' | wc -l
```

If both names are free, transfer the freshly generated source-only ZIP from
a separate Windows PowerShell at `Final project`:

```powershell
scp .\20261017\code\transfer\20261017_source.zip s4906923@login0.compute.eait.uq.edu.au:~/20261018_g1_source.zip
```

Then, in the Rangpur login shell, extract only into the new stage:

```bash
mkdir "$HOME/20261018_g1_stage"
python -m zipfile -e "$HOME/20261018_g1_source.zip" "$HOME/20261018_g1_stage"
ls -l "$HOME/20261018_g1_stage/20261017/code/tools/audit_train_raw.py"
```

Do not run full MRI decoding on `login0`. First confirm the course's current
CPU Slurm allocation syntax and the new source environment. In the allocated
CPU shell, run synthetic checks, a two-case smoke audit, then the full
143-case raw Train audit with unique JSON outputs in `logs/audits/`. The audit
script receives `--dataset-root` explicitly. It inventories all split
filenames and decodes Train arrays only. G1 does not run preprocessing, GPU
probes, or model training.
