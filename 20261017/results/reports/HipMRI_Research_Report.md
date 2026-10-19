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
learning. At this G3 audit point, the restart had no ROI cropping,
augmentation, Patch Sampler or formal training loop; actual class exposure
was measured later in G5.

### G3 candidate Patch exposure — 18 October 2026

Because there was no training Patch Sampler at this G3 decision point, a separate,
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
| Insufficient exposure in actual training patches | Unresolved at G3; measured later from G5 optimizer input |
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

At this G1–G4 review point, no formal cohort model had been trained. G1–G4
were data and pipeline measurements, not segmentation performance comparisons.

The G5 real-Train-patch GPU probe ran as Rangpur A100 job `672173` on
`B006_Week0`, batch size two, input `2×1×32×64×64`, with the same provisional
half unweighted-CE plus half foreground-Dice objective for both models. The
returned JSON SHA-256 matched Rangpur:
`f8d2a059d31f5d50302399dd8e92d1218c6159352775545f5f02b38ceee651eb`.
Both outputs were `2×6×32×64×64`; all parameter gradients were present,
finite and nonzero. Standard 3D U-Net used 0.157 GiB peak allocated GPU memory
and Improved 3D U-Net used 0.255 GiB, against a 40 GiB A100. Their one-off
forward/backward times were 9.27 and 0.90 seconds. The first run may include
CUDA/cuDNN initialization, so these are capacity timings, not a fair latency
comparison. No optimizer step or checkpoint was produced. A fixed Train-patch
learning test is still needed before full training; the probe loss is not a
Validation score or a selected training objective.

### G5 two-patch learning check — 18 October 2026

Rangpur A100 job `672194` made 100 temporary Adam steps for each 3D model
on the same two fixed `B006_Week0` Train patches, anchored on Rectum and
Prostate. The returned JSON matched Rangpur SHA-256
`097b047bd9d2a7c7f6c713cbbb98ba42558a2577f661aed744cfc8feba43c5ad`.
The paired batch had 262,144 labelled voxels, including 2,450 Bladder,
2,723 Rectum and 2,171 Prostate voxels. Both models used the same provisional
half unweighted-CE plus half foreground-Dice objective, Adam learning rate
`1e-3`, seed, inputs and step count. No checkpoint was saved.

| Model at step 100 | Fixed-batch loss | Bladder predicted voxels / Dice | Rectum predicted voxels / Dice | Prostate predicted voxels / Dice |
|---|---:|---:|---:|---:|
| Standard 3D U-Net | 0.71108 | 0 / 0 | 0 / 0 | 0 / 0 |
| Improved 3D U-Net | 0.05524 | 2,417 / 0.9706 | 2,658 / 0.9586 | 2,126 / 0.9393 |

Standard predicted all 262,144 voxels as Body from step 10 through step 100,
even while its total loss decreased from 0.97062 to 0.71108 over those
checkpoints. Thus this run directly illustrates aggregate-loss improvement
without small-organ recognition. Improved fit these exact patches well; its
final Rectum and Prostate recall were 0.9471 and 0.9295. Neither outcome
establishes Validation generalization or an isolated architecture effect:
the networks differ in multiple implementation details. Before full training,
hold the data, provisional objective, optimizer and seed fixed and extend only
the Standard model's tiny-run step budget to test delayed learning.

### G5 Standard step-budget follow-up — 18 October 2026

Rangpur A100 job `672228` repeated the Standard two-patch learning check for
500 steps without changing the case, crop origins, labels, seed, optimizer,
learning rate, objective or model implementation. Its returned JSON SHA-256
matched Rangpur:
`d10c454a24f1bdb1876fec034784a53fbb4d0a5c0da6fcfdcdc0dee75a72c53f`.
At step 100 it again predicted all 262,144 voxels as Body. At step 500 its
loss fell to 0.53212 and it predicted 7,864 Background, 210,257 Body and
44,023 Bones voxels, but **zero Bladder, Rectum and Prostate voxels**. Thus
longer training restored some large classes without resolving the small-organ
collapse on this fixed batch. The result does not prove a specific cause; the
next controlled diagnostic will change only the cross-entropy class weights
while keeping the two Train patches and 500-step budget fixed. The weights
will be derived from Train-only patient-balanced uniform Patch exposure and
reported as a diagnostic intervention, not a validated final loss.

### G5 Standard weighted-CE diagnostic — 18 October 2026

Rangpur A100 job `672730` completed 500 Standard steps on the same two fixed
Train patches. The downloaded JSON matched Rangpur SHA-256
`f9f53f6eaaabb922f29e04d4905d5529ad24c668b3b2733620c9de340008b7c7`.
Compared with unweighted job `672228`, the case and MRI/Mask hashes, Patch
origins and labels, model code, seed, Adam settings, 500-step budget, and
foreground-Dice term match. The changed factor is the CE class-weight vector:
Background 1.453871, Body 1, Bones 2.617199, Bladder 4.498586, Rectum
8.893215, and Prostate 9.699950. Its capped square-root inverse-frequency
rule was calculated from **Train-only G4 patient-uniform Patch exposure**;
see `code/config/g5_weighted_ce_diagnostic.json` for the source hash. The
comparison is a two-patch optimization diagnostic, not a cohort experiment.

| Class, step 500 | True voxels | Unweighted predicted / Dice / Recall | Weighted predicted / Dice / Recall |
|---|---:|---:|---:|
| Bladder | 2,450 | 0 / 0 / 0 | 2,518 / 0.9843 / 0.9980 |
| Rectum | 2,723 | 0 / 0 / 0 | 2,710 / 0.9796 / 0.9772 |
| Prostate | 2,171 | 0 / 0 / 0 | 2,285 / 0.9722 / 0.9977 |

The weighted run's final small-organ precision was 0.9710, 0.9819 and
0.9479, respectively, on these same patches. Weighting therefore removed
the observed small-organ prediction collapse under this controlled fixed-batch
check. Its aggregate loss fell to 0.14638, but weighted and unweighted loss
values have different definitions and should not be compared as a common
performance metric. This result supports inadequate *effective training
signal under the provisional unweighted objective* as a cause of this tiny-run
failure. It does not show that raw class imbalance alone caused historical
failures, that the chosen weights are optimal, or that weighted CE improves
unseen patients. In particular, false positives and generalization require
Validation evaluation under a frozen cohort training protocol.

### G5 complete-volume inference capacity — 18 October 2026

Rangpur A100 job `672782` used the largest-Z Train case in the G1 inventory,
`K019_Week1` (raw 256×256×144). After factor-two preprocessing, the MRI was
128×128×72 XYZ; the model adapter padded its depth to 80. Both random-weight
3D models returned legal six-class predictions on the exact 128×128×72
processed grid. The downloaded JSON matched Rangpur SHA-256
`263c2309e00d21666e69418a8c82d436585e3037285c0e02f17f959488b21c53`.

| Random-weight model | Peak allocated / reserved GiB | Median measured inference, two repeats |
|---|---:|---:|
| Standard 3D U-Net | 0.306 / 0.434 | 0.100 s |
| Improved 3D U-Net | 0.430 / 0.582 | 0.028 s |

The measured call includes tensor transfer and output conversion, but excludes
NIfTI loading, downsampling and normalization. One warmup preceded two timed
repeats per model; this is a capacity and shape check, not a robust speed
comparison. Random weights make its class predictions clinically meaningless.
It supports whole-volume Validation inference on this A100 without resolving
training duration, model quality, or original-resolution reconstruction.

### G5 paired cohort training candidate

The next controlled Standard/Improved comparison is specified in
`code/config/g5_controlled_training_candidate.json`. It keeps the G3 geometry
and normalization, G4 patient-balanced uniform crops, and the same bounded
Train-derived weighted CE plus foreground Dice objective for both networks.
The candidate budget is 40 epochs: one selected visit from each of 26 Train
patients, ten 64×64×32 crops per visit, batch size two, or 130 updates per
epoch. No augmentation is introduced in this first paired comparison, so
actual delivered-label exposure remains directly measurable. Every epoch
will record loss, class predictions/recall, exposure, gradients, runtime and
memory. Validation on complete processed volumes is planned every five
epochs; checkpoint selection gives each of the six Validation patients equal
weight and focuses on Bladder, Rectum and Prostate Dice. Test remains sealed.
This is a *candidate protocol pending a bounded training smoke run and runtime
check*, not a claim that 40 epochs or the current weights are optimal.

### G5 paired training-candidate smoke — 18 October 2026

Rangpur A100 job `672807` used the candidate configuration SHA-256
`aafdcc4e695de24fec9c6748db3448287b9e5040d036bd7eb4422776521f0ee7`.
Its returned JSON matched Rangpur SHA-256
`4161b30138243dbf3f52ba44cb097aead6046e50d60e37b60d64329bf6a336d5`.
The G4 patient selector chose `O025_Week6` and `R039_Week2`, with one patch
from each patient in each of two batches. Each Patch contained 131,072
labelled voxels. Both networks received identical four-Patch label exposure:
Bladder appeared in four patches, Rectum and Prostate in three each. Their
respective pooled voxel shares were 2.136%, 0.665% and 0.369%.

Both networks completed two weighted-CE plus foreground-Dice optimizer
updates with finite positive gradient norms. Standard's two recorded losses
were 1.42226 and 1.36789 (peak allocated 0.169 GiB); Improved's were
1.49115 and 1.30671 (0.274 GiB). The measured model-loop times were 0.74
and 0.17 seconds, excluding NIfTI loading and preprocessing. These two
updates verify basic wiring and recording only. Their losses, predictions
and times do not establish relative model quality, full-epoch runtime or
Validation performance. The proposed 40-epoch budget remains a candidate.
The formal runner has been staged separately. It cannot start with the
candidate configuration, and its first epoch will save a resumable checkpoint
once the protocol is frozen; the one-epoch budget probe remains the final
disposable run in this sequence.

### G5 controlled 3D Validation results — 19 October 2026

The Standard run (`672954`) and Improved run (`672955`) each contain 40
complete epoch JSON records. Every epoch records 130 optimizer batches and
260 delivered patches across 26 Train patients. Both manifests cite the
same frozen config SHA-256
`e5bd98a99213f4fe6cf6e3e8c6db3b95d5ef334a143a91444426d8ea019923ce`.
The local review independently recomputed every saved Validation confusion
matrix, per-class overlap, patient score, eight five-epoch selection points,
and the selected epoch. Both `best.pt` files exist on Rangpur; checkpoint
bytes have not been copied to the local repository.

Across the 40 completed epochs, the two 3D arms received exactly the same
selected Train visits and the same recorded class exposure in every epoch:
10,400 patches per arm. Bladder, Rectum and Prostate appeared in 8,962
(86.2%), 9,557 (91.9%) and 9,987 (96.0%) patches, respectively. Their
delivered voxel shares were 2.582%, 0.756% and 0.646%. This directly checks
the actual optimizer input and makes frequent complete absence of Prostate
from 3D patches an unlikely explanation for remaining errors. Its small voxel
share and imperfect spatial context remain relevant; exposure alone cannot
prove why a model succeeds or fails.

| Validation measure at selected epoch 30 | Standard 3D | Improved 3D |
| --- | ---: | ---: |
| Patient-balanced small-organ selection score | **0.8413** | 0.8325 |
| Pooled Background Dice | 0.9961 | 0.9638 |
| Pooled Body Dice | 0.9720 | 0.9266 |
| Pooled Bones Dice | 0.8268 | **0.8490** |
| Pooled Bladder Dice | 0.9297 | **0.9379** |
| Pooled Rectum Dice | **0.7865** | 0.7404 |
| Pooled Prostate Dice | **0.8321** | 0.8146 |

The selection score weights six Validation patients equally and averages
their Bladder, Rectum and Prostate Dice. The pooled Dice rows instead combine
all 38 Validation volume confusion matrices before calculating overlap;
they are different aggregation rules. Standard leads the predeclared 3D
selection criterion by 0.0089. This is a modest difference on six patients,
not evidence of a statistically established architecture benefit. The
[current Train and Validation trajectories](../figures/comparisons/g5_three_arm_training_trajectories.png)
show all 50 completed epochs. The values in this paragraph describe the
earlier 40-epoch checkpoint review.

| Organ, pooled Validation | Standard precision / recall | Improved precision / recall |
| --- | ---: | ---: |
| Bladder | 0.8973 / 0.9646 | 0.9187 / 0.9580 |
| Rectum | 0.7192 / 0.8677 | 0.6151 / 0.9296 |
| Prostate | 0.7921 / 0.8763 | 0.7310 / 0.9198 |

Improved has greater pooled Rectum and Prostate recall, with lower precision.
Its predicted-to-GT voxel ratio is 1.511 for Rectum and 1.258 for Prostate,
compared with 1.207 and 1.106 for Standard. These counts suggest more
over-segmentation; spatial overlays are needed to locate the extra voxels.
No Test metrics were consulted.

### Initial epoch-30 Validation failure review

At selected epoch 30, neither 3D model entirely omitted Rectum or Prostate
in any of the 38 Validation volumes. That narrows the old missing-organ
hypothesis under the current combined preprocessing, patient sampling and
weighted objective; it does not isolate which change caused the improvement.

Improved Rectum Dice was 0.331 on `K018_Week3` and 0.334 on `K018_Week2`.
For `K018_Week3`, it predicted 3,850 Rectum voxels against 1,134 GT voxels;
2,820 predicted Rectum voxels belonged to GT Background. The patient's
visit-pooled Rectum Dice was 0.538 for Improved versus 0.746 for Standard.
These are candidates for Validation MRI/GT/prediction overlays and spatial
boundary review. Standard's lowest Prostate case Dice was 0.643 on
`S022_Week6`, followed by 0.693 on `S022_Week2`. The confusion summaries
cannot distinguish displacement from boundary errors or yield HD95; those
require full-resolution predictions and physical-grid evaluation.

### Interim assessment after G1–G5 checks

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
dry run; at that interim point it was not actual exposure during optimization.
Formal training and its exposure records are reported below. Spatial
augmentation was not part of the frozen run.
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
G5 now shows that the Standard model can fit two fixed Train patches with
moderate Train-derived CE class weights, whereas it predicted none of the
three small organs after 500 steps with unweighted CE. This is evidence that
the loss weighting matters for that narrow learning check, not a Validation
result or a final choice of class weights.

### G5 formal-training transition — 18 October 2026

The two-patient paired GPU optimizer smoke passed, and random-weight
whole-volume inference fitted on the A100. The first paired cohort protocol
has been frozen in `code/config/g5_controlled_training_frozen.json` from
Train-only evidence. Both 3D networks will receive the same patient-balanced
uniform patches and weighted CE plus foreground Dice objective. Forty epochs
is a fixed comparison budget, not a demonstrated optimum. The first formal
epoch will save resumable model and optimizer weights; Validation first runs
at epoch five. The separate no-checkpoint epoch-budget probe (job 672837)
was still pending at the decision point and is to be cancelled. No full-cohort
checkpoint or Validation quality result exists yet.

### Optional 2D arm: sampling prerequisite

The user requested a supplementary 2D U-Net run in parallel with the primary
3D comparison. G2 raw Train coverage shows Prostate on 3,852 of 18,320 axial
slices (21.0%), Bladder on 5,317 (29.0%) and Rectum on 7,680 (41.9%). These
raw percentages do not measure delivered 2D training crops. A Train-only CPU
audit has been staged in `code/tools/audit_g5_2d_sampling.py`: one visit per
patient, ten 64×64 axial crops, comparing uniform slice/crop draws with four
uniform plus two targeted draws for each small organ. The audit must run on
Rangpur CPU and its measured exposure be reviewed before freezing a 2D
training policy or spending GPU time. The 2D arm will remain a supplementary
comparison because its receptive context differs from the 3D arms.

### G5 recovery and storage revision

At the user's request, pending one-epoch Standard/Improved jobs 672872 and
672886 were cancelled before allocation. No cohort model weights resulted.
The frozen 40-epoch protocol now requests an atomic recovery checkpoint after
every completed optimizer batch and keeps separate weights and detailed
metrics for each completed epoch. It also verifies the config, split, source
code, software versions and dataset root before resuming; an explicit future
epoch-horizon extension leaves the other training settings unchanged. The
per-batch checkpoint frequency increases shared-storage writes and may lengthen
training. `latest.pt` retains the most recent completed batch and is replaced
on each update; all completed epochs keep immutable `.pt` snapshots and
per-batch metrics in their `.json` records. Archiving every one of the 5,200
batch weight states for both models would exceed the current home-directory
quota. These are implementation provisions, not yet evidence that a 40-epoch
job has completed or that Validation quality improved.

The submission script `code/tools/submit_g5_40.sh` passed SHA-256 and Bash
syntax checks on Rangpur and submitted recovery test 672953, Standard job
672954, and Improved job 672955 on 18 October 2026. The recovery dependency
cleared; both A100 jobs were still PENDING for priority when last checked.
Slurm predicted starts of 19 October 00:45:37 and 00:58:00, respectively;
these are queue estimates, not observed training starts. No new full-cohort
checkpoint, Validation result, or 40-epoch completion is claimed here.

The optional 2D arm proceeded through its Train-only exposure gate while the
3D jobs ran. Its different slice context prevents a simple architecture-only
interpretation against the 3D arms.

### G5 completed 3D runs and 2D sampling decision — 19 October 2026

Rangpur Slurm reports `COMPLETED 0:0` for Standard job 672954 (01:15:52)
and Improved job 672955 (01:42:19). Both logs reached epoch 40/40. Their
printed best patient-balanced small-organ Validation selection scores were
0.8413 and 0.8325, both at epoch 30. These are selection scores, not a
per-class final comparison or held-out Test result. The job logs name their
checkpoint directories, but checkpoint contents and per-class Validation
records still require inspection before final model selection is recorded.

The optional 2D Train-only CPU audit completed as job 673872 (`COMPLETED 0:0`)
across 26 patients. Both candidate policies delivered ten 64×64 axial crops
per patient, 260 crops each. The uniform policy contained Prostate in 45/260
crops (17.3%), Rectum in 84/260 (32.3%), and Bladder in 75/260 (28.8%).
The organ-aware policy contained them in 100/260 (38.5%), 165/260 (63.5%),
and 122/260 (46.9%), respectively. Prostate voxel share increased from
0.337% to 0.765%; Rectum from 0.411% to 1.051%; Bladder from 2.171% to
3.496%. The returned CSV has 520 records, ten crops per policy and patient,
4096 labelled voxels per crop, and no targeted crop missing its class.
The remote and local SHA-256 values match for both files. This supports an
organ-aware 2D sampling policy, but it does not prove improved Validation
quality; oversampling could raise false positives. The separate frozen 2D
config derives its weighted CE coefficients from these Train-only delivered
voxel shares and retains whole-volume patient-balanced Validation selection.
Supplementary 2D training job 673908 later started on A100 after its CPU
check passed. Its first epoch completed, and it entered epoch two. The
initial Prostate Train patch Dice of 0.0016 is an early optimization signal,
not a Validation outcome; model selection waits for the planned five-epoch
evaluations. Among the two completed 3D arms, Standard currently leads the
frozen patient-balanced Validation criterion and is the provisional 3D
checkpoint choice. Qualitative Validation review and held-out Test remain
pending.

### Bounded 3D continuation decision — 19 October 2026

At epoch 30, the Standard and Improved patient-balanced Validation scores
reached 0.8413 and 0.8325. At epoch 40 they were 0.8308 and 0.8267, while
Train loss continued to fall. This gives no present evidence that a longer
run will improve Validation. A requested continuation from the durable epoch-40
states to epoch 50 is therefore a bounded check, using the unchanged frozen
training config, patient split, sampling, objective and five-epoch Validation
interval. Both runs retain their previous best checkpoints unless a later
Validation score exceeds them. The prepared Slurm submission also schedules
paired high-overlap and failure-case Validation overlays after both continuations
finish. This paragraph records the decision before those jobs ran; their
results follow below.

### Supplementary 2D completion and Validation review — 19 October 2026

Job 673908 completed all 40 epochs (`COMPLETED 0:0`). The 41 local JSON files
(40 epochs and one manifest) were checked against the frozen 2D config and
the per-case Validation confusion matrices. Its selected checkpoint is epoch
40 with a patient-balanced small-organ Validation score of **0.7367**. The
score rose from 0.7032 at epoch 30, dipped to 0.6414 at epoch 35, then
reached the highest observed value at epoch 40; the trajectory is noisy.

At the selected epoch, pooled Validation Dice / precision / recall were
Bladder **0.8813 / 0.8256 / 0.9449**, Rectum **0.7052 / 0.5833 / 0.8915**,
and Prostate **0.6591 / 0.5164 / 0.9105**. High Prostate recall alongside
low precision suggests substantial overprediction in this 2D scheme; spatial
review is needed to locate those false positives. The corresponding Train
patch Prostate Dice was 0.7303 at epoch 40, which is a different measurement
from whole-volume Validation Dice. The supplementary 2D arm uses axial crops,
organ-aware sampling and separate Train-derived class weights, so its quality
gap from the paired 3D arms cannot be attributed to architecture alone. The
current 50-epoch learning curves are in the
[three-arm trajectory figure](../figures/comparisons/g5_three_arm_training_trajectories.png);
no 2D Test result has been generated. The figures above describe the earlier
epoch-40 checkpoint and are retained here as a historical step.

The actual 2D optimizer input across 40 epochs contained 10,400 crops.
Bladder, Rectum and Prostate appeared in 5,144 (49.5%), 6,830 (65.7%) and
4,374 (42.1%) crops; their delivered voxel shares were 3.514%, 1.095% and
0.895%. This confirms substantial Prostate exposure beyond the original raw
slice prevalence, but also shows that most 2D crops did not contain it. The
2D crop and 3D patch voxel shares use different spatial units and contexts.

### Three-arm Validation comparison and 2D lessons — 19 October 2026

After all three arms finished 50 epochs, their selected checkpoints were
compared on the same 38 Validation volumes. This is a comparison of complete
training recipes: the 2D arm also has slice-wise context, organ-aware crops,
16 base channels and its own Train-derived class weights, while both 3D arms
use uniform 3D patches and 8 base channels. It cannot measure an isolated
architecture or sampling effect.

| Selected Validation measure | 2D | Standard 3D | Improved 3D |
| --- | ---: | ---: | ---: |
| Patient-balanced mean small-organ Dice | 0.7757 | 0.8502 | 0.8420 |
| Pooled Background Dice | 0.9950 | 0.9969 | 0.9636 |
| Pooled Body Dice | 0.9718 | 0.9769 | 0.9274 |
| Pooled Bones Dice | 0.8367 | 0.8596 | 0.8800 |
| Pooled Bladder Dice | 0.8983 | 0.9436 | 0.9509 |
| Pooled Rectum Dice | 0.6897 | 0.8038 | 0.7835 |
| Pooled Prostate Dice | 0.7672 | 0.8179 | 0.7878 |

2D has no pooled Dice advantage over both 3D models, and its small-organ
patient score is lower for each of the six Validation patients. It does have
higher pooled Body and Background Dice than Improved 3D, and its Prostate
precision (0.7377) exceeds Standard (0.7250) and Improved (0.6745), while
its Prostate recall (0.7990) is lower than both 3D values (0.9383 and
0.9467). Its Rectum recall (0.9207) exceeds Standard's 0.8807 but remains
below Improved's 0.9330; its Rectum precision is only 0.5513. These are
precision–recall tradeoffs, not a general 2D performance win. The
[six-class figure](../figures/comparisons/g5_three_arm_validation_all_classes.png)
and [shared learning curves](../figures/comparisons/g5_three_arm_training_trajectories.png)
show the current data rather than the retired three-organ plots.

The 2D organ-aware rule demonstrably increased Train crop hits relative to
uniform 2D sampling, but the current 3D uniform patches already exposed
Prostate in 96.0% of the first 40 epochs' patches. Transferring the 2D
sampler directly to 3D is therefore not the first supported intervention.
The potentially transferable observation is **specificity control**:
Improved 3D's lower Body/Background Dice and lower Prostate precision warrant
checking distant foreground predictions and organ false positives on
Validation. Any threshold, postprocessing or loss change must be proposed
and assessed in a new Train/Validation controlled experiment. No Test result
was used to select this direction.

### G6 Test evaluation gate

The prepared one-time Test evaluator requires both 3D continuations to finish
epoch 50 before it reads any Test MRI or label. It checks each run's frozen
config, source hashes, final Validation record and immutable selected-epoch
checkpoint. The existing patient-balanced Validation rule chooses `best.pt`;
Test values do not choose or alter the weights. The planned output is a single
derived JSON with per-case and pooled overlap, patient-level small-organ
scores, physical HD95, prostate apex/base measurements and inference cost.
The following sections report the completed outputs from this gate.

### G5 final 3D checkpoint selection and Validation images — 19 October 2026

Standard job 674102 and Improved job 674103 resumed their respective frozen
epoch-40 states and completed epoch 50 (`COMPLETED 0:0`). All 50 epoch JSON
records and both updated manifests were copied locally and checked against
the fixed patient split, per-case confusion matrices and predeclared checkpoint
rule. The highest patient-balanced Validation scores occurred at **epoch 50**:
Standard **0.8502**, Improved **0.8420**. Both are higher than their prior
epoch-30 peaks of 0.8413 and 0.8325. This bounded extension helped the
selection score; it does not establish that still more epochs would help.

| Pooled Validation at selected epoch 50 | Standard 3D | Improved 3D |
| --- | ---: | ---: |
| Bladder Dice | 0.9436 | 0.9509 |
| Rectum Dice | 0.8038 | 0.7835 |
| Prostate Dice | 0.8180 | 0.7878 |

The patient-balanced selection score and pooled organ Dice aggregate the
same cases differently. The updated [G5 trajectories](../figures/comparisons/g5_three_arm_training_trajectories.png)
show all 50 epochs. Job 674104 reproduced the selected Validation case
confusion matrices before generating [four paired MRI/GT/prediction panels](../case_studies/comparisons/README.md).
`J026_Week1` Rectum and `B040_Week1` Prostate show high overlap; `K018_Week3`
shows extra Rectum prediction. `S022_Week6` was originally chosen for missed
Prostate tissue using an earlier checkpoint. At epoch 50, both models have
full-volume Prostate recall above 0.97 on this case but precision below 0.50;
its current image shows additional and misplaced predicted tissue. A single
slice should be read alongside the full-volume metrics in the companion JSON.

The supplementary 2D arm later resumed from its complete epoch-40 state and
finished epoch 50 as job 674592 (`COMPLETED 0:0`). The original 40 checkpoints
and records were verified before submission, and the training code rejected
changes to the frozen configuration, sources and software versions on resume.
The local 50-record manifest SHA-256 is
`f34deddbf59b4b688464033e4301e99b605637e5fae6e564fc0ba89c09fb5ff9`;
the epoch-50 record SHA-256 is
`364c40ebb8c5e460ca99787e242653bc8a39e4868af439cf266be004ce2d6587`.
The Validation-selected epoch moved from 40 (score 0.7367) to 50 (score
0.7757). At epoch 50, pooled Validation Dice is 0.9950 Background, 0.9718
Body, 0.8367 Bones, 0.8983 Bladder, 0.6897 Rectum and 0.7672 Prostate.
These are Validation findings; no 2D Test evaluation was run. The 2D
[learning and six-class figures](../figures/baseline_2d/README.md) were
regenerated from the 50 verified records.

### G6 one-time held-out 3D Test — 19 October 2026

Job 674125 completed once (`COMPLETED 0:0`) on the fixed 30 volumes from six
unseen patients, after both epoch-50 continuations. It used the two
Validation-selected `best.pt` checkpoints and did not select weights or tune
preprocessing on Test. The local result SHA-256 is
`21ce5e2eb53e3bf758478fd7c9fd95b887d82031b7a66ed77c7ef3168adc3e2f`.
The [one-time result](../tables/comparisons/g6_test_3d.json) was independently
reconciled against all 30 per-case confusion matrices for each arm, the six
patient scores, selected epoch records, HD95 medians and resource summaries.

| Test measure | Standard 3D | Improved 3D |
| --- | ---: | ---: |
| Patient-balanced mean small-organ Dice | 0.7851 | 0.8078 |
| Pooled Bladder Dice / IoU | 0.5192 / 0.3507 | 0.7316 / 0.5768 |
| Pooled Rectum Dice / IoU | 0.8063 / 0.6755 | 0.8218 / 0.6975 |
| Pooled Prostate Dice / IoU | 0.7180 / 0.5601 | 0.7530 / 0.6039 |
| Bladder median HD95, mm | 3.36 | 3.44 |
| Rectum median HD95, mm | 7.06 | 9.17 |
| Prostate median HD95, mm | 7.98 | 6.72 |
| Parameters | 1,412,550 | 2,366,946 |
| Selected checkpoint, MiB | 16.28 | 27.21 |
| Maximum measured allocated VRAM, GiB | 0.255 | 0.351 |
| Median measured inference, ms/volume | 147.9 | 74.3 |

Improved 3D has higher Test pooled Dice for all three small organs and a
patient-balanced score higher by 0.0227. It scores higher on three of the six
Test patients and lower on the other three; the largest positive difference is
for `M015` (+0.1695 patient small-organ Dice). The checkpoint is 1.67 times the
parameter count and uses more measured allocated GPU memory. Timing is from
one run on one A100 with sequential model evaluation and should not establish
an inherent speed advantage. Other classes trade off: Standard has higher
pooled Test Body Dice (0.9591 vs 0.9312) and Background Dice (0.9875 vs
0.9619), while Improved has slightly higher Bones Dice (0.8662 vs 0.8569).
The median Body HD95 is also much larger for Improved (51.81 mm vs 4.86 mm),
including large distances on several patients. This requires a separate
spatial review before describing its overall boundaries as better.
Both models use the same frozen loss, sampling, optimizer, split and
Validation selection rule. Their Test difference is therefore evidence for
the compared architecture implementations under this protocol. The two
architectures change several components together, so it does not isolate any
single residual, upsampling or deep-supervision component.

Patient-level analysis exposes a large Bladder failure concentrated in `M015`:
the pooled patient Bladder Dice is 0.1056 for Standard and 0.3713 for
Improved, versus at least 0.90 for each other Test patient under Standard.
`W029` remains difficult for Prostate (0.4867 Standard, 0.4579 Improved).
The median per-volume prostate axial-region Dice was lower at the base than
in the centre (Standard 0.3924 vs 0.7852; Improved 0.6202 vs 0.8323).
Boundary summaries come from factor-two downsampled masks at approximately
3.36 × 3.36 × 3.12 mm; fine edges below that resolution cannot be assessed.
The [six-class Test comparison](../figures/comparisons/g6_test_all_classes.png)
shows pooled Dice; the [case-class table](../tables/comparisons/g6_test_case_class_metrics.csv)
supports per-patient review. Only six independent Test patients are
available, and the case breakdown should accompany any aggregate claim.

## Failure Analysis

The [all-class Validation and Test figures](../figures/comparisons/README.md)
were regenerated from the current selected epoch records and the same
one-time Test JSON. They include Background, Body and Bones alongside the
three small organs. On Test, pooled Bones Dice is 0.8569 for Standard and
0.8662 for Improved. Standard retains higher Background and Body Dice. The
[360-row Test case/class table](../tables/comparisons/g6_test_case_class_metrics.csv)
exposes per-volume confusion and boundary measures for review without
re-running inference.

| Pooled Test Dice, all classes | Standard 3D | Improved 3D |
| --- | ---: | ---: |
| Background | 0.9875 | 0.9619 |
| Body | 0.9591 | 0.9312 |
| Bones | 0.8569 | 0.8662 |
| Bladder | 0.5192 | 0.7316 |
| Rectum | 0.8063 | 0.8218 |
| Prostate | 0.7180 | 0.7530 |

M015's Bladder error persists across all eight Test visits: Standard's
per-volume Bladder Dice ranges from 0.068 to 0.326; Improved ranges from
0.293 to 0.640. Aggregating its eight confusion matrices, Standard predicts
142,258 Bladder voxels against 48,316 labelled voxels; 131,891 of its
Bladder predictions belong to GT Body. Improved predicts 60,953 Bladder
voxels; 39,700 belong to GT Body. The existing Week7 overlay shows spatial
misplacement, but these counts establish that the problem is longitudinal,
not confined to that one image.

| M015 visit | Standard Bladder Dice | Improved Bladder Dice | Standard Body→Bladder FP | Improved Body→Bladder FP |
| --- | ---: | ---: | ---: | ---: |
| Week0 | 0.110 | 0.394 | 21,121 | 5,073 |
| Week1 | 0.083 | 0.333 | 20,114 | 4,552 |
| Week2 | 0.131 | 0.338 | 13,742 | 7,296 |
| Week3 | 0.115 | 0.358 | 11,112 | 4,255 |
| Week4 | 0.091 | 0.365 | 17,936 | 3,875 |
| Week5 | 0.326 | 0.640 | 3,870 | 1,070 |
| Week6 | 0.084 | 0.293 | 19,663 | 9,126 |
| Week7 | 0.068 | 0.406 | 24,333 | 4,453 |

These are descriptive Test counts from the fixed one-time confusion matrices;
they are not a basis for a new training choice.

Improved Body HD95 is elevated across several Test patients, not just M015.
Its patient medians include L011 54.49 mm and M015 67.37 mm, versus Standard
4.58 mm and 18.72 mm; single-volume Improved values reach 93.44 mm on
M015 Week3 and 80.76 mm on L011 Week5. This can coexist with high Body Dice
because HD95 is sensitive to distant surface errors. The JSON identifies
which volumes warrant a spatial overlay; it cannot alone establish whether
the error is a remote foreground island, boundary displacement or another
geometric pattern.

Job 674598 (`COMPLETED 0:0`) reproduced the fixed Test confusion matrices
before drawing three-plane [Body overlays](../case_studies/comparisons/README.md).
The selected slices show Improved Body predictions outside the anatomical
outline and near image margins in all three reviewed volumes. The distant
false-positive voxel is 135.3 mm from GT Body in M015 Week3, 135.7 mm in
L011 Week5 and 137.3 mm in W041 Week0. These maxima are **not** HD95 values;
the fixed Test HD95 values are 93.44, 80.76 and 79.78 mm, respectively.
Improved has 140,809, 49,756 and 25,656 Background-to-Body false-positive
voxels in those cases, versus Standard's 36,893, 1,235 and 2,531. The spatial
evidence supports a remote-background prediction pattern, but its cause still
needs to be tested on Train/Validation, not selected from Test.

For any later model work, first inspect these patterns on **Train and
Validation**: patient-level Bladder confusion with Body, small-organ
precision/recall, Body boundary outliers, and patch exposure by class. Choose
one intervention only after that review and assess it on Validation. These
held-out Test findings describe the frozen models and must not become a
selection or tuning criterion.

### Train/Validation analysis sequence

1. The [G5 Validation case/class table](../tables/comparisons/g5_validation_case_class_metrics.csv)
   and [patient heatmap](../figures/comparisons/g5_validation_patient_class_dice.png)
   now expose every class's Dice, precision, recall, prediction volume and
   GT-to-prediction confusion. Rank patients and visits from these records,
   keeping patient weighting explicit.
2. Measure Body HD95 and inspect complete-volume overlays on **Validation**.
   The completed fixed-checkpoint audit and selected images are documented
   below.
3. Cross-reference those patients with Train-only spacing, intensity,
   organ size and actual patch exposure. Class exposure and spatial coverage
   are now measured; the mechanism behind the architecture gap still needs
   the auxiliary-output diagnostic below.
4. Predeclare one intervention and compare it with the frozen baseline using
   the same Train/Validation split and metrics. Report both gains and new false
   positives. A changed model would require a fresh independent Test cohort
   for an unbiased final estimate; the present Test cohort is already known.

### Completed Validation Body and Train spatial-coverage audit — 19 October 2026

Rangpur job 675319 evaluated both unchanged, Validation-selected epoch-50 3D
checkpoints on **all 38 Validation volumes**. Each new prediction reproduced
its saved G5 confusion matrix before surface distances were calculated. The
[complete derived record](../tables/comparisons/g5_validation_body_audit.json)
and [patient comparison](../figures/comparisons/g5_validation_body_boundaries.png)
show that the large Improved Body boundary error is not confined to the Test
examples or a single Validation patient.

| Validation Body measure, 38 volumes | Standard 3D | Improved 3D |
| --- | ---: | ---: |
| Median per-volume HD95 | 4.75 mm | 76.61 mm |
| Highest per-volume HD95 | 7.09 mm | 125.63 mm |
| Background → Body false-positive voxels | 59,766 | 1,675,691 |
| Of those, ≥20 mm from true Body | 1,007 | 1,358,516 |

Improved HD95 exceeds Standard in **all 38** Validation volumes. The three
selected [three-plane overlays](../case_studies/comparisons/README.md) show
Improved predictions far outside the anatomical Body outline in K018 Week4,
B040 Week4 and J026 Week4. They are selected illustrations; the table above
uses all Validation volumes. `K008` has a milder patient median Improved HD95
of 11.7 mm, so severity still varies by patient.

The paired 3D arms received identical case schedules and recorded label
exposure. A separate [Train-only geometry reconstruction](../data_analysis/g4_patient_sampling/g5_train_patch_geometry.json)
replayed all **13,000 uniform crop origins** from the frozen seed, raw shapes
and 50 epoch records. Background occurred in 12,988 patches, but only 2,936
(22.6%) crops touched any outer 2-voxel volume face and only **6 (0.046%)**
touched an outer 2-voxel corner. For outer 8-voxel slabs, 9,177 (70.6%)
touched any face and 423 (3.25%) touched a corner. Thus frequent Background
*class presence* does not imply broad sampling of remote image margins.
Every patch still has convolutional padding at its own boundary; this count
specifically measures coverage of the *volume's* outer anatomy/background.
That geometry gap is a plausible contributor, but **cannot alone explain the
architecture-specific difference**:
Standard saw the same crops and maintained low Body HD95. Padding, receptive
field and the Improved network's summed deep-supervision logits remain possible
mechanisms; none is established by the coverage count.

Job 675392 then compared the Improved model's usual sum of main and two
auxiliary logits with its main head alone on those same Validation volumes.
Every summed-output confusion matrix again reproduced the frozen G5 record.
The summed output reached a patient-balanced small-organ score of **0.8420**,
median Body HD95 **76.61 mm**, and **1,675,691** Background→Body errors.
The isolated main head fell to **0.2414**, **108.96 mm**, and **7,561,150**
errors, respectively. The main head was trained only as part of the sum, so
this does **not** prove that a separately retrained single-head architecture
would fail. It does reject a simple inference-time removal of auxiliary
outputs as the next fix.

A new controlled experiment is justified by the 38-case Body failure. The
first candidate is a **Train-only edge-aware patch mixture** for Improved 3D:
keep nine of ten patient patches uniform and reserve one for a volume-margin
Background/Body context. First dry-run it across the 143 Train volumes and
50 fixed patient schedules, measuring corner exposure, six-class patch hits
and voxel shares. Only proceed if this single sampling change genuinely adds
edge context without materially erasing small-organ exposure. Then retrain
Improved from scratch with the same architecture, loss, optimizer, seed,
patient visits, patch size, batch size, 50-epoch budget and Validation
selection rule. Compare with frozen Improved on patient-balanced small-organ
Dice **and** Body HD95, distant Background→Body errors, six-class Dice and
per-patient variance. The held-out Test split has already been inspected and
must not choose this intervention or be reported as a fresh unbiased test.
If the Train-only dry run cannot preserve small-organ exposure, redesign the
sampling proposal before training. If adequate edge exposure fails to fix the
Validation Body errors, the next separate controlled experiment should
retrain a single-head Improved architecture; do not combine both changes.

The final epoch-50 Validation panels and the one-time Test case matrices
identify different failure types. `K018_Week3` is a Validation Rectum
overprediction example. `S022_Week6` had been nominated from an earlier
checkpoint as missed Prostate tissue, but its final full-volume recall is high
and its precision is low; the selected image shows extra or displaced regions.
The frozen Test result shows a patient-specific Bladder collapse on `M015`,
while `W029` is difficult for Prostate in both models. Improved also has a
large Body HD95 despite its small-organ Dice gains. Job 674523 completed on
the short A100 test partition. It reproduced both original Test confusion
matrices on each of three cases and generated the
[local descriptive overlays](../case_studies/comparisons/README.md). `M020_Week1`
Prostate has high full-volume Dice (0.9046 Standard, 0.8866 Improved).
`M015_Week7` shows a severe Standard Bladder localization error: Dice 0.0676,
precision 0.0404, with much of the predicted region in other tissue. Improved
partly recovers it (Dice 0.4063) but still has false positives. On
`W029_Week6`, Prostate precision is 0.3046 Standard and 0.2045 Improved;
the displayed slice alone looks closer to GT than the full-volume metrics.
These cases illustrate the errors without feeding Test information into
training or checkpoint selection.

These observations do not establish whether the remaining errors arise from
annotation boundaries, resolution loss, anatomy shifts, or optimization. The
small-organ 3D patches were encountered frequently during actual training;
therefore a simple claim that the models never saw those organs is ruled out
for the current pipeline. The small voxel share inside those patches remains
a plausible pressure on learning.

## Conclusions

The data-led sequence established a mechanically sound Train pairing, measured
strong raw class imbalance and patient variation, checked that current
downsampling preserves every small-organ label, and recorded the exact class
exposure delivered to optimization. The two frozen 3D training recipes reached
their best Validation score at epoch 50. On the independent six-patient Test
set, Improved increased the patient-balanced small-organ Dice from 0.7851 to
0.8078 and improved pooled Bladder, Rectum and Prostate Dice, with a larger
model and a notable Body-boundary disadvantage. Its patient score advantage is
concentrated in `M015`; it does not win every patient.

The Standard and Improved arms share the frozen data and loss protocol, so
the result supports the Improved **architecture implementation** under these
conditions. Its residual, upsampling and deep-supervision components were not
ablated separately. The supplementary 2D arm has a separate Validation result
but no held-out Test result. Any later loss, sampling or architecture revision
must start as a new controlled Train/Validation experiment; this Test set must not
be used to tune one. The descriptive Test overlays have been reviewed locally.
Any submission should include the fixed split, code/config receipts, result
JSON, figures and this report, with the required HipMRI data acknowledgment.

The completed 38-volume Validation boundary audit now establishes the main
development problem: Improved has markedly worse Body boundaries than Standard
under identical 3D training exposure. The next Improved 3D direction is to
increase remote Background/Body context through a **single-factor, edge-aware
Train patch sampling experiment**, subject to a full Train-only exposure dry
run. The isolated main output performed much worse on Validation, so removing
auxiliary logits at inference is unsupported. If the sampling experiment does
not improve Validation Body errors, retrain a single-head architecture as a
separate controlled ablation. Directly importing the 2D organ-aware sampler
or simply increasing class weights is not justified by the present 3D exposure
and failure evidence.
