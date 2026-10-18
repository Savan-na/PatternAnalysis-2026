# HipMRI Research Report

## Dataset Characteristics

The frozen patient-level split contains 26 Train, 6 Validation, and 6 Test
patients, corresponding to 143, 38, and 30 MRI/mask volumes. The current
source is `/home/groups/comp3710/HipMRI_Study_open` on Rangpur. G1 inventoried
filenames for all three partitions and decoded the 143 Train MRI/mask pairs;
it did not decode Validation or Test arrays. The earlier Train class analysis
is retained separately in [historical data analysis](../data_analysis/HipMRI_Pretraining_Data_Analysis.md)
and is not treated as this audit's output.

### G1 raw integrity evidence — 18 October 2026

| Check | Current result |
|---|---|
| Filename inventory and patient split | 143 Train / 38 Validation / 30 Test; no inventory errors |
| Train pair coverage | 143/143 pairs audited |
| Mechanical case errors | 0 |
| Identical file-hash groups among Train cases | 0 |
| Warnings | 143, all reporting an unspecified Mask spatial unit |
| Mechanical audit status | Passed |
| Anatomical review status | Incomplete |

The audit ran on Rangpur CPU node `vcpu-2`, Slurm job `671208`, using the
existing `torch` environment. All 20 synthetic source tests passed before the
full scan. The source-only transfer ZIP SHA-256 was
`0ef4437163c023cfb87474300a6ceb49b18b4290d53fcec9a5cf9e83709cd12f`.
The immutable full audit result is held on Rangpur at
`20261018_g1_stage/20261017/logs/audits/train_full_20261018_job671208.json`;
its SHA-256 is
`6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1`.
This report records derived findings without copying raw MRI arrays into Git.

## Problems Identified

Every Train mask has an unspecified spatial unit in its NIfTI header. This is
a metadata warning, not proof of image–mask misalignment. The audit reported
no mechanical case errors, but aggregate intensity, label-presence, and
geometry findings still need review from the full result. Affine agreement
alone cannot certify that labels match the anatomy. Representative and
flagged MRI/mask overlays remain required before G1 can be closed.

## Interventions

None has been selected. G1 does not change source files, preprocessing,
sampling, loss, or network architecture.

## Experimental Results

No restarted model has been trained. The G1 audit is a data-quality result,
not a segmentation performance comparison.

## Failure Analysis

Pending model experiments. Raw-data review flags will be recorded here after
the full audit summary and visual checks are assessed.

## Conclusions

The current Train cohort passed the scripted mechanical integrity checks.
G1 remains open for metadata interpretation and anatomical alignment review.
No causal conclusion about Prostate or Rectum prediction failure follows from
this integrity pass alone.
