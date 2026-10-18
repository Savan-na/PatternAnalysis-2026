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

The [longitudinal organ-volume table](../data_analysis/g2_train_summary/longitudinal_organ_variation.csv)
uses physical volume from the unchanged raw Masks. Eighteen Train patients have
more than one scan. Among them, the median within-patient maximum/minimum
volume ratio is 2.90 for Bladder, 2.33 for Rectum, and 1.22 for Prostate.
These are differences between visits, without image registration or a
biological/annotation attribution. In particular, a fixed crop or volume
prior must be checked against the observed range before use.
The [organ-box center plot](../data_analysis/g2_train_summary/organ_bbox_centers.svg)
shows positions normalized to each raw voxel grid. Prostate box-center Z
fractions range from 0.307 to 0.701 across Train volumes. These are positions
within individual scan grids, not registered anatomical coordinates; the
per-volume box edges in `spatial_statistics.csv` are the relevant input for
testing any proposed ROI crop.

This first pass does not include per-slice class exposure, voxel-level MRI
histograms, longitudinal registration, boundary contrast measurements, or
actual post-transform patches. Those require a further Train-only Rangpur CPU
analysis and the G3 data-pipeline audit. G2 is therefore still in progress.
The G2 slice-audit smoke run on Rangpur CPU node `vcpu-2` (Slurm job `671819`)
processed `B006_Week0` and `B006_Week1`. It emitted 256 axial-slice rows and
240 ten-unit intensity-bin rows, matching two 128-slice volumes and 120 bins
per volume. The transferred script SHA-256 was
`ee45d5cd146710e67649e98f5f82bb130ae8bfd875a6cba66062f2165f434b05`.
The script checks the G1 source-file hashes and per-case class totals before
writing. This is a two-case software check, not a complete G2 distribution.

### G2 full Train axial and intensity audit — 18 October 2026

The same script completed all 143 Train volumes on Rangpur CPU node `vcpu-2`
(Slurm job `671833`). It produced 18,320 axial-slice rows and 17,160
per-volume intensity-bin rows. The two CSV SHA-256 values are respectively
`d8b0966e9965797991f2313858d8088c1e2c32a62bcc483090be1a0dede17e87`
and `c1b39892873b1b61dca3f03f8abcb0ec69a48154609a017a420f0502523d9930`.
Local hashes match Rangpur. Every case's slice class totals reconcile exactly
with G1, and each positive-intensity histogram total matches its G1 zero
fraction. The [summary CSVs and figures](../data_analysis/g2_train_summary/README.md)
are derived from these verified inputs.

| Class | Raw axial slices containing class | Share of 18,320 slices | Median slices per Train volume |
|---|---:|---:|---:|
| Background | 18,320 | 100.00% | 128 |
| Body | 18,170 | 99.18% | 127 |
| Bones | 18,170 | 99.18% | 127 |
| Bladder | 5,317 | 29.02% | 35 |
| Rectum | 7,680 | 41.92% | 53 |
| Prostate | 3,852 | 21.03% | 26 |

Prostate appears on 18–41 axial slices per volume (median 26), Rectum on
39–77 (median 53), and Bladder on 19–74 (median 35). The
[slice-coverage chart](../data_analysis/g2_train_summary/raw_axial_slice_hit_rate.svg)
and [volume-level chart](../data_analysis/g2_train_summary/organ_slices_per_volume.svg)
show this raw anatomical coverage. These rates describe individual *raw axial
slices*, not the probability that a training 3D patch contains an organ. Patch
depth, crop placement, and sampling policy determine that later probability.

The [pooled intensity histogram](../data_analysis/g2_train_summary/pooled_positive_intensity_histogram.svg)
counts 788,672,496 positive-intensity Train voxels in fixed ten-unit bins;
411,947,024 voxels have intensity zero (34.31% of all raw Train voxels). Zeros
are retained as observed values, not treated as missing data. The intensity
distribution provides a reference for a Train-derived normalization choice,
but no normalization rule has been selected yet. Raw class, slice, intensity,
spacing, organ-size, position and visit-level statistics now provide enough
Train-only evidence to enter G3 pipeline diagnosis. Boundary contrast remains
unmeasured and will be revisited if later errors are boundary-dominated. G3
must measure post-transform organ survival. Actual training-patch exposure
cannot be measured until the restart's sampler is identified or specified.
The original plan also names `recognition/hipmri_paper_reproduction/` as the
older model-source location. That path is absent from the current checkout;
the historical Baseline–V4 artifact inventory and final model-source location
still need verification before controlled model experiments. This does not
invalidate the completed raw-data audit.

### G3 existing Train pipeline audit — 18 October 2026

The implemented `load_case` → factor-two `downsample` → `zscore_nonzero` →
whole-volume `to_3d_model_arrays` path was audited on Rangpur CPU node
`vcpu-2`, Slurm job `671892`. The two-case smoke run found no case errors;
its `pipeline_pass=false` reflected the deliberately incomplete `--limit 2`
scope. The full run covered all 143 Train volumes, found zero case errors,
and returned `pipeline_pass=true`. The verified full JSON is in
`logs/audits/g3_existing_pipeline_full_20261018_job671892.json` with SHA-256
`319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9`.
Its raw label counts match G1 for every case and class. The
[per-volume table and figure](../data_analysis/g3_pipeline/README.md) are
reproducible Train-only summaries of that JSON.

| Organ | Raw voxels | Downsampled voxels | Raw share | Downsampled share | Median physical-volume ratio | Ratio range | Volumes retaining organ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bladder | 7,253,501 | 906,818 | 0.60415% | 0.60423% | 0.9997 | 0.9760–1.0173 | 143/143 |
| Rectum | 1,743,644 | 218,236 | 0.14523% | 0.14542% | 1.0021 | 0.9721–1.0280 | 143/143 |
| Prostate | 1,295,814 | 161,722 | 0.10793% | 0.10776% | 0.9981 | 0.9383–1.0423 | 143/143 |

Physical-volume ratio uses labelled voxel count multiplied by the respective
voxel volume; downsampled voxel counts are expected to be near one eighth of
raw counts after halving all three axes. No small organ vanished in this
cohort. The largest Prostate volume reduction was 6.17% in `D031_Week0`;
142/143 Prostate volumes were within ±5% of raw labelled physical volume.
These aggregate measurements do not establish preservation of thin structures
or contour accuracy. Raw axial Prostate coverage was 3,852/18,320 slices
(21.03%); after downsampling it was 1,919/9,160 (20.95%). The corresponding
Rectum rates were 41.92% and 42.01%, and Bladder rates 29.02% and 28.98%.
These are anatomical slice frequencies, not training-patch hit rates.

The model adapter added 131,072 Background voxels only to the 144-slice
`K019_Week1` case, padding its downsampled depth from 72 to 80. It added no
foreground labels. All model inputs were finite, all labels stayed legal,
and per-case nonzero-intensity normalized means and standard deviations were
within 8.1×10⁻⁹ of 0 and 5.9×10⁻⁹ of 1, respectively. This verifies the
implemented numerical path, not that the normalization is optimal for model
learning. The current restart still lacks ROI cropping, augmentation, a
Patch Sampler and a formal training loop; actual class exposure in model
training remains unmeasured.

### G3 candidate Patch exposure — 18 October 2026

Because the current restart has no training Patch Sampler, a separate,
Train-only *candidate simulation* compared two crop policies on the verified
downsampled masks. The frozen design used ten 64×64×32 XYZ patches per volume
per policy, seed 371017, across all 143 Train volumes and 26 patients. The
uniform policy placed all ten crops randomly. The organ-aware candidate
shared the first four uniform crops, then anchored two crops each on a
Bladder, Rectum, and Prostate voxel. This holds case frequency fixed while
changing crop placement; it is not a record of historical training or an
active sampler in the current model code.

The two-case smoke check passed on CPU job `671975`. The full CPU job
`671984` saved 2,860 patch records (1,430 per policy). The returned manifest
and CSV have SHA-256 values
`e142be157e692572a7fd96084ec80c241edc6f165498dec25ea13919abb1889b`
and `96ac1118b0109a21e3bc943df71094026dcf1852535d79124f200383d790dce2`.
Local verification matched these hashes, checked every case/arm/draw key and
crop bound, reconciled every patch count with its source Mask, and reproduced
the manifest totals. The [class and patient tables with a comparison figure](../data_analysis/g3_pipeline/README.md)
hold the derived results.

| Organ | Uniform patches containing organ | Organ-aware patches containing organ | Uniform sampled-voxel share | Organ-aware sampled-voxel share | Uniform patches containing entire organ | Organ-aware patches containing entire organ |
|---|---:|---:|---:|---:|---:|---:|
| Bladder | 1,211/1,430 (84.69%) | 1,300/1,430 (90.91%) | 2.198% | 2.584% | 190/1,430 (13.29%) | 249/1,430 (17.41%) |
| Rectum | 1,307/1,430 (91.40%) | 1,344/1,430 (93.99%) | 0.637% | 0.750% | 172/1,430 (12.03%) | 193/1,430 (13.50%) |
| Prostate | 1,377/1,430 (96.29%) | 1,405/1,430 (98.25%) | 0.643% | 0.697% | 546/1,430 (38.18%) | 649/1,430 (45.38%) |

For these fixed crops and one deterministic seed, uniformly placed patches
rarely missed Prostate altogether. Yet Prostate still occupied under 0.7%
of sampled voxels, and fewer than half of patches contained its entire
label. Its 0.643% uniform-crop share is about six times its 0.10776%
downsampled full-volume share, showing that the crop already concentrates
central anatomy while leaving Prostate numerically rare. Thus Patch Hit Rate
alone overstates how much complete small-organ
context a crop supplies. The organ-aware candidate increased cohort-wide
hits and voxel share modestly; per-case hits did not improve uniformly
(Prostate hits fell in 2 of 143 cases). This simulation does not prove that
organ-aware sampling improves Dice or recall. Ten patches per volume produce
10–80 patches per patient because patients contribute 1–8 scans; the study
still needs a patient-aware exposure decision. Actual training exposure
remains unknown until a sampler and training loop are implemented and logged.

## Problems Identified

Every Train mask has an unspecified spatial unit in its NIfTI header. This is
a metadata warning, not proof of image–mask misalignment. The audit reported
no mechanical case errors. The intensity spread and tiny Bladder components
were flagged for review, not established corruption. Affine agreement alone
cannot certify that labels match the anatomy. The visual screens below add
section-level evidence without claiming expert verification of every boundary.

| Stage 1 hypothesis | Current evidence status |
|---|---|
| Gross raw-data corruption or pairing error | Not supported by the 143-case mechanical audit; subtle annotation errors remain possible |
| Severe global minority-class imbalance | Supported by raw Train voxel counts |
| Patient and visit-level organ variation | Supported by per-patient shares and physical-volume ranges |
| Small organs occupy limited axial sections | Supported by the full Train slice audit |
| Existing downsampling deletes a small organ | Excluded for these 143 Train volumes; all three organs survive, although physical-volume and boundary changes remain |
| Uniform candidate crops frequently miss Prostate entirely | Not supported for the audited 64×64×32 crops and seed: 96.29% hit rate; complete-organ inclusion and voxel share remain limited |
| Insufficient exposure in actual training patches | Unresolved; the current restart code has no Patch Sampler |
| Boundary contrast explains segmentation failure | Unresolved; no current model-error or boundary-contrast test isolates it |

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

The verified factor-two downsampling and nonzero-intensity Z-score remain the
current preprocessing starting point. G3 compared candidate crop placements
without changing source files. For the next controlled 3D experiments, the
provisional sampling reference is one scan per Train patient per epoch plus
uniform 64×64×32 crops for both model arms. Organ-aware placement remains a
targeted alternative pending actual training exposure and Validation errors.
No loss or architecture change has been selected in G4.

### G4 patient-balanced sampling preparation — 18 October 2026

The code now provides a Train-only sampler interface that selects one scan per
patient per epoch, cycles through that patient's visits over successive
epochs, and prepares paired 64×64×32 XYZ image/Mask patches as model-ready
`1×32×64×64` image and `32×64×64` label tensors. It supports a uniform arm
and a four-uniform plus two-per-organ candidate arm. A recorder measures class
hits and voxel counts on the exact post-transform labels supplied to the
model. The full Rangpur CPU dry run in job `672092` produced 520 patch records:
260 per policy, ten per patient, one selected visit from each of 26 Train
patients. The Rangpur and local copies of the manifest and CSV have matching
SHA-256 values `cafd3593efa5880a4a61b6281bac5d177f58cd713e734cf8dad422d191f0b90c`
and `04a6822f15940b658f0c3eb062528aa4fa17721552030dc5dc6d45ce5c441968`.
The [G4 class, patient and figure summary](../data_analysis/g4_patient_sampling/README.md)
reconciles all 520 rows to the manifest, patient visit selection and G3
processed grid bounds. Eight successive scheduled epochs cover all 143 Train
visits while preserving 26 equal patient selections per epoch; this schedule
check used G3 metadata and did not train a model.

| Small organ | Patient-uniform patch hits | Patient-organ-aware patch hits | Uniform sampled-voxel share | Organ-aware sampled-voxel share |
|---|---:|---:|---:|---:|
| Bladder | 232/260 (89.23%) | 245/260 (94.23%) | 2.921% | 3.144% |
| Rectum | 238/260 (91.54%) | 248/260 (95.38%) | 0.747% | 0.843% |
| Prostate | 247/260 (95.00%) | 252/260 (96.92%) | 0.628% | 0.679% |

The organ-aware policy raises pooled exposure modestly in this one fixed
patient-balanced epoch. Patient-level voxel share nevertheless falls in 10/26
Bladder, 4/26 Rectum and 7/26 Prostate cases; one patient's Rectum patch-hit
count also falls. These paired candidate streams do not establish a Dice or
recall gain. The crop and recorder are ready for a training loop, but no
training loop, spatial augmentation or GPU run has used them yet. Exact
post-augmentation class exposure still needs to be logged during training.

## Experimental Results

No restarted model has been trained. G1–G4 are data and pipeline measurements,
not segmentation performance comparisons.

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
G2 measured Train-only class, patient, volume, slice, intensity, spacing,
organ-size, normalized position and visit-level variation. These results
support global class imbalance and limited raw axial coverage for Prostate and
Rectum. G3 then verified that the implemented factor-two downsampling and
normalization path processes all 143 Train cases without case errors or
eliminating a small organ. The lowest Prostate labelled physical-volume ratio
is 0.9383, so spatial-detail and boundary effects still warrant review.
Neither G2 nor G3 establishes the cause of old model failures. G4 now supplies
a patient-balanced Train patch interface and verified one-epoch CPU tensor
dry run; it is still not actual exposure during optimization. The current
restart has no formal training loop or spatial augmentation.
For one fixed 64×64×32 candidate policy and seed, uniform random crops already
contained Prostate in 96.29% of patches, but Prostate filled only 0.643% of
their voxels and was fully contained in 38.18%. Organ-aware candidate crops
raised these measures modestly. These findings narrow the sampling hypothesis:
mere absence from patches is not prominent under this simulated crop size,
while weak voxel exposure and incomplete organ context remain plausible
training concerns. Patient exposure is also unequal under per-volume draws.
G4 equalized patient draw counts in a new candidate stream; it measured 95.00%
uniform Prostate patch hits and 0.628% Prostate voxel share. Its organ-aware
alternative raised these to 96.92% and 0.679%. The limited gain favors a
uniform-crop reference for the first controlled training comparison, with
organ-aware placement held for a targeted follow-up if actual batch logs and
Validation failures support it. Validation must establish whether any
intervention improves segmentation without increasing false positives.
