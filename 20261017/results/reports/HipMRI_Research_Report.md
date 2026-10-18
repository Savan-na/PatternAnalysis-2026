# HipMRI Research Report

## Dataset Characteristics

The frozen patient-level split contains 26 Train, 6 Validation, and 6 Test
patients, corresponding to 143, 38, and 30 MRI/mask volumes. The current
source is `/home/groups/comp3710/HipMRI_Study_open` on Rangpur. G1 inventoried
filenames for all three partitions and decoded the 143 Train MRI/mask pairs;
it did not decode Validation or Test arrays. The earlier Train class analysis
is retained separately in [historical data analysis](HipMRI_Pretraining_Data_Analysis.md)
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
| Gross alignment screen | 143/143 Train volumes, two axial sections each; no obvious whole-image/mask displacement |

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
The user downloaded the complete audit JSON to the local `logs/audits/` folder;
its local SHA-256 matches the Rangpur value above.

### G2 raw Train statistics — first pass, 18 October 2026

The [G2 tables and figures](../data_analysis/g2_train_summary/README.md) were
recomputed from the verified complete G1 JSON, without reusing the historical
summary or reading Validation/Test label distributions. All shares below use
the 1,200,619,520 raw Train voxels across 143 volumes as their denominator.

| Class | Voxels | Raw Train share |
|---|---:|---:|
| Background | 720,230,975 | 59.988278% |
| Body | 429,188,773 | 35.747276% |
| Bones | 40,906,813 | 3.407142% |
| Bladder | 7,253,501 | 0.604147% |
| Rectum | 1,743,644 | 0.145229% |
| Prostate | 1,295,814 | 0.107929% |

These values reproduce the earlier Train-only class percentages to their
reported precision. This confirms the historical calculation against the
current raw audit; it does not measure what the model receives after
preprocessing and patch sampling. The global Background/Prostate voxel-count
ratio is about 556:1.

Across the 26 Train patients, pooled-visit Prostate share ranges from 0.03667%
to 0.24809%, and Rectum share from 0.08296% to 0.45474%. Patients contribute
1–8 scans each, so the pooled global share weights patients by their scan
counts and image sizes. Per-volume Prostate share ranges from 0.03484% to
0.29290%. These are observed label-size differences, not an explanation of
model failure by themselves. The per-volume nonzero MRI median ranges from
7 to 47 in source intensity units; the zero-voxel share and spacing plots show
additional between-scan variation. The [class-frequency chart](../data_analysis/g2_train_summary/class_frequency_log.svg)
uses a log axis so that all six nonzero categories remain visible. The
[patient plot](../data_analysis/g2_train_summary/patient_rare_organ_share.svg),
[intensity plot](../data_analysis/g2_train_summary/intensity_case_summary.svg),
and [spacing plot](../data_analysis/g2_train_summary/voxel_spacing.svg) are
case-level summaries, not voxel histograms.

This first pass does not include per-slice class exposure, voxel-level MRI
histograms, longitudinal registration, boundary contrast measurements, or
actual post-transform patches. Those require a further Train-only Rangpur CPU
analysis and the G3 data-pipeline audit. G2 is therefore still in progress.

## Problems Identified

Every Train mask has an unspecified spatial unit in its NIfTI header. This is
a metadata warning, not proof of image–mask misalignment. The audit reported
no mechanical case errors. The intensity spread and tiny Bladder components
were flagged for review, not established corruption. Affine agreement alone
cannot certify that labels match the anatomy. The visual screens below add
section-level evidence without claiming expert verification of every boundary.

### G1 visual review — 18 October 2026

Six prespecified Train cases were rendered directly from the source NIfTI
files, without preprocessing. Each figure shows axial sections through the
Prostate and Bladder plus coronal and sagittal sections through the Prostate.
On these displayed sections, the organ outlines appear broadly consistent
with visible anatomy; no gross whole-image/mask shift is apparent. This is a
qualitative screen, not expert confirmation of every boundary or every slice.

| Case and raw overlay | Reason selected | What the displayed sections show |
|---|---|---|
| [B038_Week0](../data_analysis/g1_selected_cases/g1_overlay_B038_Week0.png) | Historical small Prostate label | Small central Prostate outline is visible below the Bladder; no obvious gross displacement. |
| [K019_Week1](../data_analysis/g1_selected_cases/g1_overlay_K019_Week1.png) | Only 144-slice Train volume | Organ outlines remain within the displayed anatomy despite the longer scan. |
| [N010_Week3](../data_analysis/g1_selected_cases/g1_overlay_N010_Week3.png) | Tiny second Bladder component | Main Bladder aligns visually; targeted component view below resolves the flag. |
| [D031_Week7](../data_analysis/g1_selected_cases/g1_overlay_D031_Week7.png) | Tiny second Bladder component | Main Bladder aligns visually; targeted component view below resolves the flag. |
| [S033_Week1](../data_analysis/g1_selected_cases/g1_overlay_S033_Week1.png) | Highest case MRI maximum | No obvious gross misalignment in the shown planes; these views do not locate or explain the volume maximum. |
| [H017_Week7](../data_analysis/g1_selected_cases/g1_overlay_H017_Week7.png) | Highest zero-voxel share | Extensive black exterior is visible; the panels do not establish whether every zero voxel is background. |

#### Targeted Bladder component check

The follow-up tool used the same 6-neighbour component definition as the full
audit and plotted each flagged voxel at its actual MRI location. In
`N010_Week3`, the secondary component contains 2 voxels (8.80 mm³), at
voxel coordinates x=126, y=129, z=67–68; see the
[targeted view](../data_analysis/g1_bladder_review/g1_bladder_island_N010_Week3.png). In
`D031_Week7`, it contains 1 voxel (4.40 mm³), at x=126, y=136, z=80; see the
[targeted view](../data_analysis/g1_bladder_review/g1_bladder_island_D031_Week7.png). Each
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
exact affine agreement. The header warning does not, by itself, establish a
spatial error or justify changing source Masks.

#### Full Train gross-alignment screen

One patient contact sheet was generated for each of the 26 Train patients,
covering all 143 Train volumes. For each volume, the sheet shows one axial
section through the Prostate and one through the Bladder, with all four organ
contours visible where present. All 26 sheets (286 displayed sections) were
visually screened. No obvious whole-image/mask displacement or organ contour
falling wholly outside the imaged anatomy was seen in those sections. Examples
include [C032](../data_analysis/g1_patient_screen/g1_train_patient_C032.png),
[M036](../data_analysis/g1_patient_screen/g1_train_patient_M036.png), and
[W012](../data_analysis/g1_patient_screen/g1_train_patient_W012.png). The other sheets are listed
in `PROJECT_MAP.md` and remain local review artifacts.

The contact-sheet index in `logs/audits/g1_train_contact_sheets_20261018.json`
contains 26 unique patients and 143 unique case IDs. Its figure paths record
the original flat Rangpur output location; the 26 local PNGs are now grouped
under `results/data_analysis/g1_patient_screen/`.
Its local SHA-256 is
`96561d0ee6766305dd2e95369e248cb8a3dba7f06c4a623d4df8532d11fce17f`.
This screen does not examine every slice, quantify contour accuracy, or replace
expert annotation review. Bladder size and appearance vary visibly across
scans; G2 will quantify Train-only distributions before any data-pipeline
decision.

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

Pending model experiments. G1 raw-data findings and visual-screen limits are
recorded under Problems Identified above.

## Conclusions

G1 passed its raw-data integrity gate: all 143 Train pairs passed mechanical
checks, and the two-section-per-volume gross-alignment screen found no obvious
whole-image/mask displacement. The two apparent Bladder islands are a
6-neighbour connectivity effect, not evidence of separated organs or
confirmed annotation errors. The unspecified Mask spatial-unit header is a
documented non-blocking metadata warning because MRI/Mask affines match exactly
and spacings passed the matching check. Boundary correctness across every
slice remains unverified.
G2 can now measure Train-only class, intensity and spatial distributions; no
causal conclusion about Prostate or Rectum prediction failure follows from G1.
The first G2 pass confirms the severe global class imbalance and measures
patient-level variation. It does not yet measure sampling imbalance or justify
a preprocessing or model change.
