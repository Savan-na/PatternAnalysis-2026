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

The complete Train JSON also reports zero non-finite MRI voxels, zero
non-finite Mask voxels, zero illegal Masks, and all six class IDs present in
every Train volume. MRI and Mask affines match exactly across the cohort
(maximum absolute difference 0). All 143 MRI volumes have LPS orientation
and declare millimetres; 142 shapes are 256×256×128 and one is 256×256×144.
In-plane spacing ranges from 1.40625 to 1.875 mm, while the third-axis
spacing is approximately 1.56 mm.

Across cases, the maximum finite MRI intensity ranges from 335 to 1114,
nonzero-voxel standard deviation from 52.21 to 144.76, and zero-voxel share
from 14.09% to 52.64%. These are case-level ranges, not diagnosed artifacts;
zero intensity can be legitimate background. Rectum and Prostate masks are
single connected components with no gaps in occupied Z slices. Two Bladder
masks have a second, very small component (`N010_Week3` and `D031_Week7`);
both require image review before being classified as annotation defects.

The audit ran on Rangpur CPU node `vcpu-2`, Slurm job `671208`, using the
existing `torch` environment. All 20 synthetic source tests passed before the
full scan. The source-only transfer ZIP SHA-256 was
`0ef4437163c023cfb87474300a6ceb49b18b4290d53fcec9a5cf9e83709cd12f`.
The full audit result is held on Rangpur at
`20261018_g1_stage/20261017/logs/audits/train_full_20261018_job671208.json`;
its SHA-256 is
`6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1`.
The frozen split JSON SHA-256 is
`81531ec7cb84ef58c41393ff23e1105e7cf03aa97828fb12853ce8624fcb8900`;
the G1 audit script SHA-256 is
`98ccab2d8e71eae477914192cd3d5751b9ba4b8601f366f6d734774abff1494f`.
This report records derived findings without copying raw MRI arrays into Git.

## Problems Identified

Every Train mask has an unspecified spatial unit in its NIfTI header. This is
a metadata warning, not proof of image–mask misalignment. The audit reported
no mechanical case errors. The intensity spread and two tiny Bladder islands
are review flags, not established corruption. Affine agreement alone cannot
certify that labels match the anatomy. Representative and flagged MRI/mask
overlays remain required before G1 can be closed.

The visual review set is prespecified from the current audit and historical
flags: `B038_Week0` (small Prostate label), `K019_Week1` (unusual shape),
`N010_Week3` and `D031_Week7` (tiny Bladder islands), `S033_Week1`
(highest case MRI maximum), and `H017_Week7` (highest zero-voxel share).
These are review targets, not confirmed data defects. The raw-image overlay
tool is `code/tools/review_g1_overlays.py`; no overlay has yet been judged.

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
