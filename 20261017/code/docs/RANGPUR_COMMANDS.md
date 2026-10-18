# Rangpur G1 commands and terminal boundaries

The user's initial prompt is Windows PowerShell at `Final project`. Passwords
belong only in the terminal. The previously documented dataset root and
staging directory are historical and must be checked before reuse. This file
records the procedure; the user receives each executable command after its
paths and prerequisites are confirmed.

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

## 2. Pending confirmed paths

Once the active paths are known, build a fresh source-only ZIP using
`code/tools/package_transfer.py`, transfer it to a new staging location, and
run synthetic checks. Submit the two-case smoke audit and then the full
143-case raw Train audit on a CPU compute node. Each audit writes a unique
JSON result under `logs/audits/`; an existing result is never overwritten.
The script receives `--dataset-root` explicitly, so no old path is compiled
into the code. It inventories all split filenames and decodes Train arrays
only. G1 does not run preprocessing, GPU probes, or model training.
