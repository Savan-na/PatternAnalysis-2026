# HipMRI 3D Improved U-Net Segmentation

## Project Status

Production filename discovery, MRI/segmentation pairing, and frozen patient-level split checks passed on the real Rangpur HipMRI dataset. One-pair NIfTI loading, spatial validation, 3D physical resampling, MRI intensity normalisation, and an on-demand 3D Patch Dataset are implemented. The student-run Batch 2.4 Rangpur regression suite passed 87/87 tests; one real training and one real validation DataLoader patch passed acceptance checks. Full-dataset processing, model training, and final performance evaluation remain incomplete.

## Problem

This project investigates volumetric MRI segmentation for prostate radiotherapy planning using HipMRI.

## Dataset and Labels

The dataset source is the HipMRI Study on Rangpur at `/home/groups/comp3710/HipMRI_Study_open`. MRI files are discovered under `semantic_MRs/*_LFOV.nii.gz`; segmentation files are discovered under `semantic_labels_only/*_SEMANTIC.nii.gz`. Files are paired by patient ID and Week number. The dataset root can be passed to `discover_dataset()`; its default is the Rangpur path.

Production discovery verified 38 unique patients and 211 matched MRI/segmentation pairs on Rangpur, with no missing or unmatched files or unexpected patient IDs. A student-run, read-only audit of all 211 MRI/mask headers found LPS orientation for both files in every pair and zero shape, affine, spacing, or orientation grid mismatches. All 211 MRIs declare spatial units of `mm`; all 211 masks declare `unknown`. This header audit did not decode every image array or assess clinical annotation accuracy. The authoritative CSIRO semantic label mapping is:

| ID | Structure |
| --- | --- |
| 0 | Background |
| 1 | Body |
| 2 | Bones |
| 3 | Bladder |
| 4 | Rectum |
| 5 | Prostate |

## Approved Patient Split

The approved patient assignments are stored explicitly in `dataset.py`; seed **3710** records provenance and is not used to redraw the split at runtime.

| Split | Verified patients | Verified volumes |
| --- | ---: | ---: |
| Train | 26 | 143 |
| Validation | 6 | 38 |
| Test | 6 | 30 |
| Total | 38 | 211 |

Real-data validation confirmed that all Weeks from a patient stay in one partition, no patients overlap between splits, and `K019_Week1` is in Train. Discovery rejects missing MRI/mask partners, unknown or missing approved patients, duplicate patient/Week pairs, and deviations from the approved volume counts.

## NIfTI Loading and Spatial Validation

`load_volume_pair()` in `volume_io.py` accepts one approved `VolumePair` from `discover_dataset()`. It decodes one 3D MRI and segmentation without changing voxel order, shape, affine, spacing, orientation, or source files. The returned metadata includes both effective affines, voxel spacing, axis codes, stored data types, and labels present. MRI values are returned as finite `float32` without intensity normalisation; validated integer labels 0–5 are returned as `uint8`, with a subset of labels permitted in an individual case.

Matching declared physical spatial units are accepted only with matching geometry. For the audited HipMRI pattern, an MRI declaring `mm` and a mask declaring `unknown` may be paired only after 3D shape, valid affine, spacing, and orientation checks pass. The returned `spatial_unit` is the MRI-declared working unit; `segmentation_header_spatial_unit` retains the mask's actual `unknown` value, and `segmentation_unit_inferred=True` records the conditional inference. Explicit unit conflicts or geometric mismatches remain errors. The original mask header is not rewritten or described as explicitly declaring `mm`.

After this correction, the student ran 33/33 tests successfully on Rangpur and loaded the real `K019_Week1` pair. Both decoded arrays had shape `(256, 256, 144)`; MRI dtype was `float32`, segmentation dtype `uint8`, axis codes were `('L', 'P', 'S')`, and labels present were `(0, 1, 2, 3, 4, 5)`. The working unit was `mm`, while the mask header remained `unknown` and its unit inference flag was `True`.

Still pending: array-content verification and resampling for all 211 pairs; Standard and Improved 3D U-Net training; full-volume inference; and final Dice/IoU evaluation. The initial Patch Dataset retains integer mask IDs rather than one-hot encoding them.

## 3D Resampling and Spatial Alignment

`resample_volume_pair()` in `resampling.py` converts one validated `LoadedVolumePair` to a shared physical 3D output grid. The first target-spacing candidate is `(1.6796875, 1.6796875, 1.55999755859375)` mm, the median of patient-level median spacings across the **26 training patients and 143 training volumes**. It was derived without fitting a parameter on validation or test patients and is not claimed to be clinically optimal. Both planned 3D model arms must use the same resampling policy.

`make_target_grid()` preserves the source voxel-axis directions and covers the source voxel-edge field of view; it does not reorient LPS data. MRI and mask use the **same output shape and affine**. Each array is sampled with its own output-to-input mapping, `inverse(source_affine) @ output_affine`. MRI uses linear interpolation (`order=1`) with `nearest` boundary mode. The discrete mask uses nearest-neighbour interpolation (`order=0`) with `grid-constant` mode and background value `0`; no fractional label IDs are introduced. Both operations run on CPU, one pair at a time.

Source review found that SciPy `constant` mode erased some valid face-touching labels when output centers fell just outside the source voxel-center range but remained within its voxel-edge extent. The targeted correction uses `grid-constant` for the mask, retaining those edge labels while assigning background beyond the edge domain. MRI uses `nearest` to avoid a zero-intensity ramp near the boundary. Synthetic regression tests cover all six faces and samples beyond the source edge. This change did not alter the physical-grid equations.

The configured allocation guard is **512 MiB** (`536870912` bytes), estimating four times the combined source and output MRI/mask array bytes. It rejects a predicted excess before output allocation; it is **not** a guaranteed peak resident-memory limit. The result retains spatial-unit provenance, original and resulting labels, per-class voxel counts and physical volumes. Existing foreground labels that disappear during resampling cause an error. These checks do not establish clinical segmentation accuracy.

`resampling_config.json` records the spacing candidate and memory cap, which callers **pass explicitly** to `resample_volume_pair()`; the function has no defaults for them, so editing JSON alone does not change a call that passes the old values. The JSON also declares MRI boundary mode `nearest`, mask boundary mode `grid-constant`, and boundary value `0`; `resampling.py` reads and validates those declarations and rejects unsupported changes. Interpolation orders `1`/`0`, the 4× planning multiplier, grid-rounding tolerance, and orthogonality tolerance are fixed in source. The configuration records the approved initial policy rather than a trained or clinically optimised parameter set.

The student-run **updated Rangpur suite passed 53/53 tests**. Real acceptance examples after the boundary fix were:

| Pair | Input → output shape | Prostate voxels, input → output | Peak RSS | Exit status |
| --- | --- | ---: | ---: | ---: |
| `K019_Week1` | `(256, 256, 144)` → `(256, 256, 144)` | 17,245 → 17,245 | 182,120 KiB | 0 |
| `S035_Week0` | `(256, 256, 128)` → `(286, 286, 128)` | 3,478 → 4,448 | 185,740 KiB | 0 |

Only these **two real MRI arrays** were resampled as acceptance examples. The complete 211-volume dataset has not been fully resampled or clinically validated; the voxel-count change for S035 is not a segmentation-performance result.

## MRI Intensity Normalisation

`normalize_volume_pair()` in `intensity_normalization.py` processes one validated `ResampledVolumePair` on CPU. It selects voxels where the **original MRI intensity is nonzero**, using MRI intensities alone and never the ground-truth segmentation. It computes the 0.5th and 99.5th percentiles of those values, clips them to the resulting bounds, then computes the clipped values' `float64` mean and population standard deviation (`ddof=0`). It Z-scores the selected values and writes a new finite `float32` MRI of the same shape. Original zero-intensity locations remain zero. The segmentation is retained by reference without in-place modification; the affine is copied, and pair identity, spatial geometry, label metadata, and spatial-unit provenance are preserved.

An entirely zero MRI, non-finite values, invalid 3D shape or dtype, and zero or numerically degenerate clipped nonzero standard deviation are rejected. The result records the selected voxel count, percentile bounds, mean and standard deviation used, original zero fraction, and output statistics on the originally selected voxels. `normalization_config.json` freezes the original-MRI-nonzero selection, percentile values `0.5`/`99.5`, selected-voxel clipping, `ddof=0`, zero preservation, `float64` statistics, and `float32` output. The module reads and validates this policy. These settings are an initial engineering choice for both future model arms, not a clinically optimised formula.

The focused local synthetic suite passed **18/18** tests in WSL with an import shim because NiBabel was unavailable locally. The student-run combined Rangpur suite passed **71/71** tests in **0.214 s**. The student then reported these two real acceptance examples:

| Case | Selected voxels | Clipping bounds | Clipped mean / population std | Output selected mean / std | Peak process RSS | Wall time |
| --- | ---: | --- | --- | --- | ---: | ---: |
| `K019_Week1`, `(256, 256, 144)` | 7,199,017 | 1.0 / 298.0 | 62.71922972261352 / 70.6617517907729 | -2.0519089452645352e-09 / 0.9999999993619909 | 341780 KiB | 5.66 s |
| `S035_Week0`, `(286, 286, 128)` | 7,136,647 | 0.012912326492369175 / 292.36941619872994 | 64.3451434267759 / 73.6351732255404 | -1.364474822943357e-09 / 0.9999999989009823 | 345368 KiB | 5.54 s |

For both cases, unchanged mask and affine checks passed and the process exited with status `0`. Peak RSS is for the **whole verification process**, not memory exclusively allocated by normalisation. These are two real examples only: full 211-volume normalisation, clinical accuracy, and training remain unverified.

## On-Demand 3D Patch Dataset

`HipMRIPatchDataset` in `patch_dataset.py` accepts only `train` and `validation`, using the frozen patient-disjoint assignments from `discover_dataset()`. Each `__getitem__` loads, resamples, and normalises **one** MRI/mask pair through the approved functions; no processed full-volume cache is retained. The source NumPy convention `(X,Y,Z)` is transposed to tensor `(D,H,W)=(Z,Y,X)`, then MRI and mask receive identical crop coordinates. This axis permutation is not another physical resampling operation. Each sample contains a `float32` MRI tensor `[1,D,H,W]`, a `torch.long` mask `[D,H,W]` with integer IDs 0–5, and patient ID, Week, crop origin in DHW order, and sampling mode.

`patch_dataset_config.json` sets an initial `(64,64,64)` patch, one training patch per volume, foreground-sampling probability `0.5`, prostate foreground label `5`, seed `3710`, and center-crop validation. Patch dimensions can be configured but must be positive, divisible by 16, and fit the processed volume without padding. Training crop choices are reproducible from seed, epoch, and sample index; `set_epoch()` allows them to vary reproducibly between epochs. If foreground sampling selects label 5 and it exists, the sampled patch contains a prostate voxel; an absent label safely falls back to a random crop. Validation uses a deterministic center crop without label-guided selection. The initial DataLoader uses `num_workers=0` and CPU preprocessing.

Codex reported **16/16 focused local synthetic tests passed** in WSL. They used constructed arrays, mocked preprocessing interfaces, and an in-memory NiBabel import shim; they were not real-data tests. The student ran the five-module Rangpur regression suite: `Ran 87 tests in 8.646s`, then `OK` (71 previously verified tests plus 16 Patch Dataset tests).

The student then ran a CPU-only, read-only real-data check with `HipMRIPatchDataset`, `Subset`, and `DataLoader` (`batch_size=1`, `num_workers=0`). MRI shape/dtype and finiteness, mask shape/dtype and labels, prostate-aware training sampling with label 5 present, and deterministic validation center sampling passed:

| Split and case | MRI batch | Mask batch | Labels | Crop origin DHW | Mode | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Train `K019_Week1` | `(1,1,64,64,64)` float32 | `(1,64,64,64)` int64 | `[0,1,2,3,5]` | `[40,60,64]` | `prostate` | PASS |
| Validation `B040_Week0` | `(1,1,64,64,64)` float32 | `(1,64,64,64)` int64 | `[1,2,3,4,5]` | `[32,87,87]` | `center` | PASS |

The final output was `PASS: REAL PATCH DATASET VERIFICATION`; process wall time was **11.72 s**, maximum resident set size **768352 KiB**, and exit status **0**. RSS covers the whole Python verification process, not GPU memory or an isolated Patch Dataset allocation. Only two real patches were accepted. All 211 MRI arrays have not been processed through the full pipeline; center crops may miss the prostate in other validation cases, and repeated on-demand preprocessing may limit training throughput. Full-volume inference, clinical accuracy, model training, and Dice/IoU evaluation remain pending.

## Engineering Question

Compared with a Standard 3D U-Net (Normal Difficulty baseline), does a 3D Improved U-Net (Hard Difficulty) with volumetric residual units and deep supervision reduce clinically important prostate boundary and slice-transition errors, improve segmentation performance, and justify its additional computational cost?

## Planned Model Comparison

- Standard 3D U-Net — Normal Difficulty baseline
- 3D Improved U-Net — Hard Difficulty model with volumetric residual units and deep supervision

## Planned Evaluation

- Per-class Dice
- Per-class IoU
- Prostate base/apex analysis
- Failure-case analysis
- Peak GPU VRAM
- Inference time per case

## Repository Structure

- `modules.py`: Model components for the planned comparison.
- `dataset.py`: Approved patient split, filename discovery, pairing, and coverage checks.
- `test_dataset.py`: Synthetic tests for discovery and split validation.
- `volume_io.py`: One-pair 3D NIfTI decoding and spatial/content validation.
- `test_volume_io.py`: Synthetic NIfTI tests for loading, geometry, and invalid values.
- `resampling.py`: One-pair shared-grid 3D physical resampling and integrity checks.
- `resampling_config.json`: Approved initial spacing, memory limit, and boundary policy.
- `test_resampling.py`: Synthetic spatial, label, boundary, and memory-guard tests.
- `intensity_normalization.py`: One-pair MRI-only clipping and nonzero Z-score normalisation.
- `normalization_config.json`: Frozen initial percentile, selection, precision, and zero policies.
- `test_intensity_normalization.py`: Synthetic normalisation and input-integrity tests.
- `patch_dataset.py`: On-demand, aligned 3D training and validation patches.
- `patch_dataset_config.json`: Initial patch shape and sampling policy.
- `test_patch_dataset.py`: Synthetic tensor-axis, crop, sampling, split, and DataLoader tests.
- `train.py`: Training workflow for the planned models.
- `predict.py`: Inference workflow for the planned models.

## Artificial Intelligence Usage Disclosure

AI assistance was used to draft the discovery code, split validation, NIfTI loader, resampling, normalisation, and Patch Dataset implementations, synthetic tests, and documentation, and to review reported failures and propose targeted corrections. The approved patient assignments, expected counts, CSIRO labels, training-derived spacing evidence, and real intensity-audit results were provided as project facts; they were not generated by the assistant. The student executed the Rangpur tests, 211-pair header audit, and real-pair loading, resampling, normalisation, and DataLoader checks. The detailed AI contribution and command record is maintained in `ai.md`.
