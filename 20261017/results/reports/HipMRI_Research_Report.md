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
masks have a second, very small 6-neighbour component (`N010_Week3` and
`D031_Week7`); the targeted review below explains this finding.

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
no mechanical case errors. The intensity spread and tiny Bladder components
were flagged for review, not established corruption. Affine agreement alone
cannot certify that labels match the anatomy. The selected overlays below provide an
initial visual screen; the flagged components need further review.

### G1 visual review — 18 October 2026

Six prespecified Train cases were rendered directly from the source NIfTI
files, without preprocessing. Each figure shows axial sections through the
Prostate and Bladder plus coronal and sagittal sections through the Prostate.
On these displayed sections, the organ outlines appear broadly consistent
with visible anatomy; no gross whole-image/mask shift is apparent. This is a
qualitative screen, not expert confirmation of every boundary or every slice.

| Case and raw overlay | Reason selected | What the displayed sections show |
|---|---|---|
| [B038_Week0](../data_analysis/g1_overlay_B038_Week0.png) | Historical small Prostate label | Small central Prostate outline is visible below the Bladder; no obvious gross displacement. |
| [K019_Week1](../data_analysis/g1_overlay_K019_Week1.png) | Only 144-slice Train volume | Organ outlines remain within the displayed anatomy despite the longer scan. |
| [N010_Week3](../data_analysis/g1_overlay_N010_Week3.png) | Tiny second Bladder component | Main Bladder aligns visually; targeted component view below resolves the flag. |
| [D031_Week7](../data_analysis/g1_overlay_D031_Week7.png) | Tiny second Bladder component | Main Bladder aligns visually; targeted component view below resolves the flag. |
| [S033_Week1](../data_analysis/g1_overlay_S033_Week1.png) | Highest case MRI maximum | No obvious gross misalignment in the shown planes; these views do not locate or explain the volume maximum. |
| [H017_Week7](../data_analysis/g1_overlay_H017_Week7.png) | Highest zero-voxel share | Extensive black exterior is visible; the panels do not establish whether every zero voxel is background. |

#### Targeted Bladder component check

The follow-up tool used the same 6-neighbour component definition as the full
audit and plotted each flagged voxel at its actual MRI location. In
`N010_Week3`, the secondary component contains 2 voxels (8.80 mm³), at
voxel coordinates x=126, y=129, z=67–68; see the
[targeted view](../data_analysis/g1_bladder_island_N010_Week3.png). In
`D031_Week7`, it contains 1 voxel (4.40 mm³), at x=126, y=136, z=80; see the
[targeted view](../data_analysis/g1_bladder_island_D031_Week7.png). Each
secondary component touches the main Bladder component diagonally. Thus each
mask has one connected Bladder region under 26-neighbour connectivity; the
earlier count of two is specific to 6-neighbour connectivity. The zoomed MRI
views place these voxels beside the main Bladder outline, without showing a
grossly misplaced region. Pixel-level boundary correctness remains uncertain,
so neither component is classified as an annotation defect and no mask is
changed.

The component output is on Rangpur at
`20261018_g1_stage/20261017/logs/audits/g1_bladder_islands_20261018.json`.
The follow-up script SHA-256 is
`b2eb4be2c428bf2f4db37c443fcc1dfcb81061e98da8554e7453ec9aad70c29b`.
The unspecified Mask spatial-unit header remains a metadata caveat despite
exact affine agreement. Six selected cases cannot establish anatomical
correctness for all 143 Train volumes, so broader anatomical review remains
open.

The PNGs are medical-image-derived local review artifacts. HipMRI Study data
use terms prohibit re-identification and commercial use and require an
acknowledgment and citation when presenting derived results publicly. For this
segmentation study, cite the HipMRI Study MRI-alone treatment-planning paper:
https://doi.org/10.1016/j.ijrobp.2015.08.045. Acknowledgment for any public
presentation: "Data were provided in part by HipMRI Study."

## Interventions

None has been selected. G1 does not change source files, preprocessing,
sampling, loss, or network architecture.

## Experimental Results

No restarted model has been trained. The G1 audit is a data-quality result,
not a segmentation performance comparison.

## Failure Analysis

Pending model experiments. The G1 raw-data findings and remaining visual
checks are recorded under Problems Identified above.

## Conclusions

The current Train cohort passed the scripted mechanical integrity checks.
The two apparent Bladder islands are a 6-neighbour connectivity effect, not
evidence of separated organs or confirmed annotation errors. G1 remains open
for broader anatomical review and interpretation of the Mask unit header.
No causal conclusion about Prostate or Rectum prediction failure follows from
this integrity pass alone.
