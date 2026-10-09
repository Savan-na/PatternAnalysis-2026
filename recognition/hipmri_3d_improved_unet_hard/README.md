# HipMRI 3D Improved U-Net Segmentation

## Project Status

Production filename discovery, MRI/segmentation pairing, and frozen patient-level split checks passed on the real Rangpur HipMRI dataset. One-pair NIfTI loading, spatial validation, and 3D physical resampling are implemented. After the Batch 2.2 boundary correction, the student-run Rangpur regression suite passed 53/53 tests, and two real pairs passed resampling checks. Full-dataset array-content verification and resampling, later preprocessing, model training, and final performance evaluation remain incomplete.

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

Still pending: array-content verification and resampling for all 211 pairs; intensity normalisation; one-hot mask encoding; PyTorch training dataset preparation; Standard and Improved 3D U-Net training; and final Dice/IoU evaluation.

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
- `train.py`: Training workflow for the planned models.
- `predict.py`: Inference workflow for the planned models.

## Artificial Intelligence Usage Disclosure

AI assistance was used to draft the discovery code, split validation, NIfTI loader, synthetic tests, resampling implementation, and documentation, and to review reported failures and propose the targeted spatial-unit and resampling-boundary corrections. The approved patient assignments, expected counts, CSIRO labels, and training-derived spacing evidence were provided as project facts; they were not generated by the assistant. The student executed the Rangpur tests, 211-pair header audit, and real-pair loading and resampling checks. The detailed AI contribution and command record is maintained in `ai.md`.
