# Submission verification archive

These scripts and receipts document the preparation of the Standard/v3 submission. They are research records, not part of the runnable coursework package in `recognition/HipMRI_UNet_s4906923/`.

- `verify_evidence.py`: recomputes the ten-model results and checks original clinical figure hashes. It also checks the migration receipt and the executable equivalence of the submitted model/data/metric code.
- `verify_replay.py`: repeats the read-only Standard/v3 Validation comparison using authorised MRI and local weights.
- `migration_receipt.json`: the unchanged receipt from GPU job 682858 (76 matching Validation results).
- `editorial_source_equivalence.json`: records comment/docstring changes after the GPU check. The original replay driver is preserved in Git commit `e1d4306`; the archived driver now locates the submission from this archive.
- `preparation_verification.json`: local checkpoint identity, CPU component parity, backup and GPU verification status.
- `remote_backup_verification.json`: confirms the complete private backups against the server manifests.
- `submission_verification.json`: file checksums and submission-preparation checklist. It describes the current submission; it is refreshed when the report or package changes.

Run from the workspace root:

```powershell
python -B 20261017/archives/submission_verification/verify_evidence.py
python -B 20261017/code/tools/verify_final_model_backups.py
python -B 20261017/archives/submission_verification/verify_replay.py --dataset-root /path/to/HipMRI_Study_open --standard-checkpoint /path/to/standard/best.pt --v3-checkpoint /path/to/v3/best.pt --output /path/to/new_receipt.json --device cuda
```

The replay command requires a fresh output path. The archived scripts read the current submission folder; the submission itself has no dependency on this archive. An isolated HPC extraction can set `HIPMRI_SUBMISSION_ROOT` to its extracted code directory.
