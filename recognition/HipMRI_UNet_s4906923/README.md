# HipMRI Standard 3D U-Net and Improved v3

## Preparation status — not yet the complete submission package

This directory stages two self-contained PyTorch model components. Historical
training code, model files and research evidence have not been removed. No model
has been retrained or selected using a new criterion during this migration.

- `modules.py`: `StandardUNet3D` and `ImprovedUNet3DV3`; PyTorch-only imports,
  with no dependency on the old Improved, 2D, or research model packages.
- Standard: base width 8, 1,412,550 parameters, original state-dictionary keys.
- Improved v3: base width 8, 2,366,946 parameters, residual context modules,
  14 one-group GroupNorm layers, dropout 0.3, summed main/half/quarter logits.
- Original selected checkpoints remain epoch 50 for both models.

## Verification completed

Four CPU tests compare the standalone definitions against the historical models:
default initial tensors/parameter keys/RNG, strict loading and evaluation logits,
dropout/gradient/Adam update parity, and a PyTorch-only import check.

The actual downloaded best checkpoints also match their recorded SHA-256 values,
load strictly into both implementations, and produce exactly equal CPU logits on
a fixed synthetic input. Fifty downloaded epoch JSON records per model match
the previously verified local records. The Standard and v3 backup directories
contain 103 and 105 files respectively, including all 50 epoch weights.

The machine receipt is [preparation_verification.json](preparation_verification.json).
Re-run the local backup/weight check from the workspace root:

```powershell
python -B 20261017/code/tools/verify_final_model_backups.py
```

On 21 October 2026, server-side `sha256sum -c` confirmed all 208 files
(103 Standard and 105 v3) against the local manifests. See
[remote verification receipt](remote_backup_verification.json). Weights are
private research backups, not coursework GitHub deliverables.

**Pending:** replay real MRI predictions after migration on the GPU, and implement/verify the final
`dataset.py`, `train.py`, and `predict.py` entry points. This page is a staging
record, not a claim that all course submission requirements are complete.

## Research interpretation

v3 is the original Validation-selected overall candidate. It does not dominate
every organ, patient, or boundary metric. Standard remains the comparison model;
all ten experiments and negative results remain research evidence, even though
only these two models are intended for the final runnable package.

- [Pre-final readiness review (Chinese)](../../20261017/results/reports/model_performance_comparison/PRE_FINAL_READINESS_REVIEW_ZH.md)
- [Ten-model foreground / prostate analysis](../../20261017/results/reports/model_performance_comparison/ASSESSMENT_FOREGROUND_PROSTATE_ZH.md)

No weights or MRI files are committed. Local backups remain under
`20261017/archives/remote_training_backups/` and are excluded from Git. They are
not a separate external archive yet; deletion or relocation requires a verified
copy and preservation of the documented evidence.

## Artificial Intelligence Usage Disclosure

ChatGPT/Codex assisted with extracting standalone components, tests, backup
checks and this preparation note. Verification includes comparison to the
historical definitions and actual recorded checkpoint hashes. CPU parity is not
an independent clinical review or a replacement for the pending real-MRI/GPU
replay. The student remains responsible for reviewing and explaining the work.
