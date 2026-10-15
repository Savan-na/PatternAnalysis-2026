# Class Prediction Collapse Diagnosis

## V4 E20 status update - 14 October 2026

The frozen paired E20 Validation completed. Treatment reduced pooled Prostate false positives by 32.30%, but failed three mandatory acceptance gates: Prostate patient-macro Dice gain, pooled Recall-loss limit, and other-class patient-macro Dice safeguard. The frozen decision rejected Treatment, retained V3 E12 as the protected candidate and did not automatically select Control E20. No formal freeze or Held-out Test evaluation has occurred. See the independent audit (local-only: `research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md`) and decision (local-only: `research_archive/10_scientific_documentation/COMP3710_V4_FINAL_DOCS_20261014/V4_ACCEPTANCE_DECISION.md`).

The experiment motivation and prospective design below are preserved as historical context; they do not establish a causal explanation for the earlier class-collapse observations.

### Original experiment motivation (13 October 2026)

The selected V3 E12 parent still exhibits a Prostate
false-positive problem in the recorded Validation evidence.

The next controlled experiment therefore tests whether stronger
false-positive penalisation can improve Prostate segmentation
without unacceptable recall loss or regression in other classes.

### Original predeclared intervention (13 October 2026)

- Control: continue the selected V3 E12 loss unchanged.
- Treatment: change the Prostate Tversky alpha from 0.5 to 0.7.
- Changed anatomical class: Prostate, index 5.
- Continuation: E13-E20, 143 optimiser updates per epoch.
- Scheduled full-volume Validation: E14, E16, E18 and E20.
- Primary comparison: E20.

Both arms are designed to share the selected parent checkpoint,
model architecture, data split, sampling and optimisation policy.

### Evidence available at the 13 October 2026 planning snapshot

CPU numerical, training-loop, paired-integrity, checkpoint,
journal and synthetic failure-recovery checks had passed
within their respective test scopes. At that snapshot, V4 GPU
execution and full-volume Validation had not yet occurred.
The completed E20 Validation decision is summarised in the
14 October status update above.

### Scientific interpretation boundary

At that planning snapshot, the six Validation patients had already
informed development and checkpoint selection; any subsequent V4
Validation comparison would therefore be exploratory. The later
E20 patient-level comparison is reported in the 14 October update
above. It is not Held-out Test evidence, and no formal model freeze
or Held-out Test result has been established.

See the [report-to-code traceability register](../evidence/traceability.md) and
`hipmri_3d_improved_unet_hard/README.md` for the current
experiment design and verification record.


Internal investigation document; no new training result or submission milestone is claimed. Section 5 integration is preserved. The initial diagnosis batch added this report and isolated [diagnostic utility](../../hipmri_3d_improved_unet_hard/diagnostics/class_collapse.py) / [CPU tests](../../hipmri_3d_improved_unet_hard/diagnostics/test_class_collapse.py). Existing production files, configurations, figures and checkpoint protections are unchanged.

## Current optimisation decision - 12 October 2026

The later student-approved V2 weighted-CE experiment supersedes the historical gradient-attribution recommendation below. No attribution utility was implemented or executed. V2 training job 652649 completed ten epochs/1,430 updates and Validation job 652948 completed all 38 volumes. These are student-supplied Rangpur records, preserved with hashes in [ai.md](../../hipmri_3d_improved_unet_hard/ai.md). The V2 log/checkpoint/per-volume evidence remains remote; the downloaded original E10 logs do not substitute for V2 records. Original sections below retain their historical scope.

| V2 class | Pooled Dice | Precision | Recall | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bones | 0.362280 | 0.528495 | 0.275602 | 2,494,303 | 7,348,494 |
| Bladder | 0.826771 | 0.788611 | 0.868812 | 405,702 | 228,536 |
| Rectum | 0.461378 | 0.597848 | 0.375633 | 126,354 | 312,224 |
| Prostate | 0.445695 | 0.340148 | 0.646212 | 314,665 | 88,805 |

Precision/recall/FP/FN are arithmetic from the supplied pooled confusion totals, not a new evaluation. All four classes have TP-positive coverage in all 38 volumes. V2 is no longer an all-zero minority predictor, but Bones/Rectum remain substantially under-recalled and Prostate over-prediction is material. Patient-macro Dice is Bones 0.361053, Rectum 0.439592 and Prostate 0.453977. These differ from pooled metrics and cannot reveal individual patients' errors. No V2 panels or per-patient confusion matrices were available locally for this implementation task.

| Priority / hypothesis | Supporting evidence | Counterevidence and unresolved mechanism | Action |
| --- | --- | --- | --- |
| 1. Insufficient or uneven optimisation within 1,430 updates | Weighted CE recovered all classes without changing architecture; only one V2 Validation endpoint exists. | No measured Validation trajectory or demonstrated plateau. The final training loss is not evidence of class convergence; 126,976 leading NUL bytes and missing early log records remain unresolved. | Isolated V3 continuation: restore V2 state, add two epochs, full Validation, then decide. |
| 2. Weighted-objective class competition / precision-recall trade-off | Minority overlap improved while Body/Background Dice fell. Prostate predicted volume 476,872 exceeds truth 251,012, with precision 0.340148. | Confusion totals do not establish gradient dominance or identify which class should receive a different coefficient. Existing Dice and CE have different absent-class behaviour; the earlier probability probe tested original E10 models, not V2. | Hold the loss fixed while testing duration. Consider one later loss change only after V3 results. |
| 3. Context, resolution, anatomy or boundary limitations | 64-cube patches and ~1.68/1.68/1.56 mm grid may limit anatomical context/detail; earlier paired E10 panels showed local failures. | No demonstrated alignment defect. Bones exposure was 86.01% with 9.099% sampled voxel share; the sampler already includes prostate-aware crops. Existing panels cannot establish V2 boundary causes, and region Dice does not measure surface error. | Preserve sampling, spacing and aggregation for V3. No automatic augmentation, threshold or architecture change. |

**Implemented first experiment:** [V3 source](../../hipmri_3d_improved_unet_hard/train_improved_v3.py), [configuration](../../hipmri_3d_improved_unet_hard/improved_v3_config.json), [CPU tests](../../hipmri_3d_improved_unet_hard/test_improved_v3.py), and [exact procedure/decision rules](../experiments.md). It restores model/AdamW/all RNG state through the protected V2 resume helper, reuses the unchanged weighted-CE/Dice/deep-supervision objective and sampling, and uses full 38-volume Validation at E12. Selection prioritises patient-macro Dice for Bones/Rectum/Prostate with per-class regression guards and all-volume TP coverage. V2 remains the fallback. Validation preserves training RNG, and separate immutable epoch files plus exclusive append/fsync/hash-chain logs protect provenance and recovery.

**Local evidence:** final 19 focused tests passed; 126 combined tests (107 prior + 19 new) passed in 10.655 seconds on WSL CPU/PyTorch 2.12.1+cu130. These include genuine small synthetic updates, serialized optimiser/RNG continuation and mocked remote orchestration; local NiBabel is absent. No real V3 MRI/GPU result, improvement or Test result is claimed. The requested first GPU budget is 19 minutes diagnostic plus 45 minutes for E11-E12/full Validation, pending approval and current resource checks. Further stages and V4/V5 depend on actual results. A failed candidate will be retained as evidence, not relabelled as success.

## A. Problem statement and evidence

**Question:** Why do the current E10 models predict Background, Body and some Bladder, while omitting meaningful Bones, Rectum and Prostate segmentation?

| Evidence | Finding | Provenance and limits |
| --- | --- | --- |
| Standard E10 full Validation, job 649669 | 38 successful volumes; no predicted Bones, Rectum or Prostate; pooled Bladder DSC 0.6220039283221238 | Student-supplied Rangpur records, already reconciled in [Section 5](../../FINAL-REPORT.md#51-controlled-validation-comparison). Its original downloaded JSONL is now inspected locally, with all 38 per-volume confusion matrices and pooled metrics verified; no inference was rerun. |
| Improved E10 full Validation, job 650393 | 38 successful volumes; only three Bones voxels across two volumes; no Rectum or Prostate; pooled Bladder DSC 0.768230 | Original downloaded JSONL now verifies two Bones TP and one FP in K018_Week1/Week4: pooled DSC=3.943109603265999e-7, not exact zero or useful segmentation. Cancelled job 649912 is not the successful evaluation. |
| Train exposure reconstruction, job 650736 | 143 volumes / 26 patients / 1,430 patches, PASS; verified distributions below | Original JSONL read locally and independently recomputed; Standard-checkpoint authenticated replay, not proof of unchanged historical image bytes or a new training run. Slurm completion/time supplied by the student. |
| Checkpoint probability probe, job 650924 | PASS, three fixed patches / six main-head forwards; omitted-class TP zero even on selected Train crops | Original downloaded JSONL authenticated below; probabilities/ranks and source/configuration parity checked locally. No new forward or backward run; scheduler completion/time student-supplied. |
| E10 training identity | Both models completed 10 epochs / 1,430 updates | Downloaded original training logs verify Standard E4–10 and Improved E2–10, with resume/final counters ending at 10/1,430; all recorded source hashes match local files. Earlier step records and checkpoint binaries remain unavailable. |
| Paired M030_Week5, S022_Week5, K008_Week0 | Both models predict zero voxels of classes 2, 4 and 5 despite their presence in truth | Six local `results.json` files in `%USERPROFILE%\Downloads\failure_cases_650490`; authenticated paths/hashes are preserved in [the evidence manifest](../evidence/failure-cases.md). Their metadata records E10 / 1,430 updates. |
| B040_Week0 | Both omit Prostate, Rectum and Bones; 7,105 true Prostate voxels | Original full-validation confusion matrices now verify these counts; external PNGs were previously inspected, but B040 prediction-driver JSONs remain unavailable. Standard assigns 7,093 Prostate voxels to Body and 12 to Bladder; Improved assigns 7,042 to Body and 63 to Bladder. |

The four selected cases show Bladder improvement for B040, M030 and K008, but regression for S022. K008 has 11,114 true Prostate voxels and no Prostate prediction. These purposive Validation cases substantiate failure, not training class frequencies or a causal explanation. They do not establish clinical accuracy or Test performance.

Evidence levels used here: **source-confirmed**, **direct local synthetic test**, **locally inspected experimental artefact**, **student-reported remote result**, and **proposed / unexecuted measurement**. They are not interchangeable.

The local assessment PDF (local-only: `../COMP3710_Report_2026_draft_v2.2.pdf`) was inspected through its compressed text content streams, without package installation; conventional local PDF extractors were unavailable. Section 2.2, printed pp. 5-6, calls for patient separation, spatial alignment, normalisation, one-hot multi-channel encoding, a baseline comparison, per-class held-out Test DSC/IoU and failure-case analysis; the Hard model description includes volumetric residual units and deep supervision. Appendix A, p. 17, identifies HipMRI's 38 patients / 211 volumes and directory names. These requirements motivate careful diagnosis but do not prescribe a new loss formula or prove the training distribution. The internal Version 2.0 Draft / filename v2.2 discrepancy and encoding clarification remain existing compliance questions; this batch does not repeat that audit or claim a new requirement is satisfied.

## B. Pipeline audit

| Path | Confirmed implementation | Diagnostic interpretation |
| --- | --- | --- |
| [dataset.py](../../hipmri_3d_improved_unet_hard/dataset.py): `discover_dataset()`, `discover_pairs()`, `assign_approved_split()` | Pair by patient/Week; freeze 26/143 Train, 6/38 Validation, 6/30 Test | No split redraw. Diagnostic discovery checks filenames in all partitions, but only training arrays are decoded. |
| [volume_io.py](../../hipmri_3d_improved_unet_hard/volume_io.py): `load_volume_pair()` | Validate 3D matching shape, affine, spacing/orientation; integer IDs 0-5; output uint8 mask | No observed label remapping. MRI mm / mask unknown accepted only after geometric checks, with inference provenance. |
| [resampling.py](../../hipmri_3d_improved_unet_hard/resampling.py): `resample_volume_pair()` | Shared physical grid; MRI order 1 / nearest boundary; mask order 0 / grid-constant background 0; validate IDs and reject complete foreground disappearance | Discrete label preservation is explicit. Changes in structure voxel counts or partial-volume representation remain possible; survival is not anatomical fidelity. |
| [intensity_normalization.py](../../hipmri_3d_improved_unet_hard/intensity_normalization.py): `normalize_volume_pair()` | MRI-only nonzero clipping and Z-score; segmentation preserved | Ground truth does not select intensity statistics. |
| [patch_dataset.py](../../hipmri_3d_improved_unet_hard/patch_dataset.py): `__getitem__()`, lines 97-159 | Identical XYZ-to-DHW transpose `(2,1,0)` and crop; mask `torch.long`, MRI float32 | A valid aligned crop can still contain little foreground. Geometry correctness is distinct from adequate exposure. |
| [train.py](../../hipmri_3d_improved_unet_hard/train.py): `combined_loss()`, line 46 | CE class-index target, softmax on channel dimension 1; Dice compares channel c with `target == c`, c=1..5 | No off-by-one channel mapping found. Background channel 0 participates in CE. |
| [train_improved.py](../../hipmri_3d_improved_unet_hard/train_improved.py): `combined_loss()`, line 29 | Same base objective for main and two auxiliary heads; resize logits only, then normalise weights 1, 0.5, 0.25 by 1.75 | Integer masks are not interpolated. The production training function, rather than an unused helper, was tested. |
| [modules.py](../../hipmri_3d_improved_unet_hard/modules.py): `StandardUNet3D.forward()`; [improved_modules.py](../../hipmri_3d_improved_unet_hard/improved_modules.py): `ImprovedUNet3D.forward()` | Six raw channels; Improved defaults to the main head for inference | No internal argmax or softmax changes training targets. |
| [validate.py](../../hipmri_3d_improved_unet_hard/validate.py): `sliding_window_predict()`, `confusion_matrix()`, `metrics_from_confusion()`; [validate_improved.py](../../hipmri_3d_improved_unet_hard/validate_improved.py) | Uniform mean of overlapping six-channel softmax probabilities, then argmax; aligned XYZ truth; confusion rows=true, columns=prediction | Evaluation uses the same label/channel indexing and resampled truth. Both-absent metrics are null, not inflated to 1. |
| [predict.py](../../hipmri_3d_improved_unet_hard/predict.py): `load_model()`, `prepare_volume()`, `process_pair()` | Delegates protected loading, preprocessing, inference and metrics | No separate segmentation algorithm or class permutation was found. |

The implemented channel order is **0 Background, 1 Body, 2 Bones, 3 Bladder, 4 Rectum, 5 Prostate**, consistent with the supplied CSIRO mapping and evaluator names. This verifies internal correspondence; the exact authoritative CSIRO citation remains unavailable locally. No new blocking production defect was demonstrated. Existing 181-volume preprocessing acceptance supports executability, not clinical correctness or sufficient training supervision.

## C. Ground-truth class-frequency analysis

### Original full-Train audit authentication

The student executed Train-only reconstruction in Rangpur job **650736** and supplied Slurm **COMPLETED, ExitCode 0:0, elapsed 00:18:46**. Those scheduler facts are student-supplied; they are not fields independently verified from the JSONL. The original `C:\Users\ruiji\Downloads\hipmri_e10_evidence\patch_full_650736.jsonl` was read locally: **742,570 bytes**, SHA-256 **`028a495c41e14bec4855bad5fd55c1dcea51472c890285d7c499f5f4d93b22a1`**.

All **1,575 nonempty lines** parse as JSON: one setup, **143 full-volume records**, **1,430 patch records**, one final summary. No malformed/duplicate JSON records or error events were found. Full-volume indices cover 0-142, the frozen 26 Train patients and 143 unique patient/Week identities. Patch keys `(dataset_index, completed_epoch)` uniquely cover every volume in epochs 1-10, with sampler epoch=completed epoch minus one. Shape/count totals, six label IDs, legal DHW origins, affine determinants, target spacing and unit provenance pass **8,326 local assertions**. The summary reports **PASS**, `full_train_completed=true`; independently calculated pooled, distribution, patient, epoch and branch fields match it. The existing `summarize_exposure()` also reproduces its fields exactly, without decoding images or running a model.

Setup identifies the Standard E10 snapshot `standard_unet_epoch10.pt`, 10 epochs / 1,430 updates, checkpoint SHA-256 `f2a87f9c73e068719e2f1d878291116d26320be7c5176827f78537fc4fb4f9cf`, matching the recorded paired-case snapshot identity. Its saved source hashes match current files and both original training configurations; training pair order, preprocessing/sampling policies and library versions match the E10 training logs. The actual checkpoint binary was not reopened locally. This is one Standard-checkpoint-authenticated reconstruction of the common target policy, not a separately executed Improved reconstruction.

There are **1,420 distinct recorded mask hashes**: eleven all-Background masks share one hash, accounting for ten repeated hashes. Their case/epoch keys remain distinct, so this is not duplicate-record failure. Hash syntax was checked; mask arrays were not available for independent hash recomputation. **Neither these new patch hashes nor matching source/configuration hashes prove unchanged historical MRI/mask file bytes.** No historical source-image content hashes were recorded. Findings describe authenticated configuration replay on the audit inputs, not newly observed historical optimiser inputs.

### Raw, resampled and sampled distributions


All six classes occur in **143/143 raw and resampled volumes, 26/26 patients**. Below, class fractions use all six classes in each denominator, including Background. Counts pool longitudinal scans rather than giving each patient equal weight.

| Class | Raw voxels / fraction (%) | Resampled voxels / fraction (%) | Patch voxels / fraction (%) |
| --- | ---: | ---: | ---: |
| Background | 720,230,975 / 59.988278 | 716,170,300 / 60.172845 | 69,380,520 / 18.508090 |
| Body | 429,188,773 / 35.747276 | 423,821,834 / 35.609638 | 246,261,463 / 65.693212 |
| Bones | 40,906,813 / 3.407142 | 40,203,232 / 3.377888 | 34,110,362 / 9.099350 |
| Bladder | 7,253,501 / 0.604147 | 7,015,311 / 0.589429 | 15,115,893 / 4.032347 |
| Rectum | 1,743,644 / 0.145229 | 1,707,089 / 0.143430 | 4,618,511 / 1.232043 |
| Prostate | 1,295,814 / 0.107929 | 1,270,769 / 0.106770 | 5,379,171 / 1.434959 |

**Denominators:** raw=**1,200,619,520 voxels**, resampled=**1,190,188,535 voxels**, sampled=**374,865,920 voxels = 1,430 x 262,144**. Raw voxel physical volumes vary **3.084971196949482-5.4843664169311515 mm3** (median 4.401299264281989). Every resampled/patch voxel is **4.401299264281989 mm3**, from spacing `(1.6796875,1.6796875,1.55999755859375)` mm. Resampled affine determinants and spacings match those values. Source voxel volumes are logged scalars, not newly read source affines.

To account for variable raw voxel sizes, `physical_fraction[c] = sum_i(count[i,c] * voxel_volume[i]) / sum_i(total_voxels[i] * voxel_volume[i])`. Raw physical fractions differ from raw voxel-count fractions:

| Class | Raw physical fraction (%) | Resampled physical fraction (%) |
| --- | ---: | ---: |
| Background | 60.040519 | 60.172845 |
| Body | 35.727871 | 35.609638 |
| Bones | 3.389040 | 3.377888 |
| Bladder | 0.591280 | 0.589429 |
| Rectum | 0.144024 | 0.143430 |
| Prostate | 0.107265 | 0.106770 |

The physical denominators are **5,221,552,082.115291 mm3** (raw) and **5,238,375,923.452358 mm3** (resampled), summed across scans. Repeated/overlapping patches revisit anatomy; their 374,865,920-voxel denominator is sampled supervision, not unique anatomical volume. Label survival and close physical fractions do not establish annotation or boundary accuracy. The sampler enriches Bones, Bladder, Rectum and Prostate relative to their whole-volume fractions; whole-volume rarity must not be substituted for sampled-patch exposure.

The utility records raw-mask counts after decoding, resampled-mask counts, shapes, affine, voxel physical volumes and spatial-unit provenance. Its summaries give six counts and fractions, volumes/patients containing each class, and per-volume fraction min/Q25/median/Q75/max. Differences in presence before/after resampling are directly comparable. Unexpected IDs, non-finite inputs and total foreground disappearance are rejected by production validation; exceptions stop the diagnostic with a nonzero exit and no successful summary.

Required interpretations:

- Whole-volume pooled fraction: `sum class voxels / sum all volume voxels`, separately on raw and resampled grids; multiply by 100 for percentages. All six labels, including Background, are in the denominator.
- Presence rates use the selected volume and unique-patient denominators. Full scope requires 143 volumes / 26 patients. A subset PASS is not full training coverage.
- Raw scans have different spacings, so raw voxel counts do not measure anatomical volume equally. Use per-case `count * abs(det(affine[:3,:3]))` for mm³; do not compare raw counts directly with common-grid patch counts.
- Per-volume and per-patient summaries expose longitudinal weighting. The training objective visits volumes, not equally weighted patients; patients with more Weeks contribute more patches.
- Count survival does not prove boundary accuracy. Label fragments and relative physical volumes can change without a class disappearing.

No MRI, mask or prediction array is written by the utility. Metadata output goes to stdout, to be retained externally only after authorisation.

## D. Patch-exposure analysis

### Confirmed sampling mechanism

The frozen config uses seed 3710, one 64³ patch per volume and prostate-branch probability 0.5. Training calls `set_epoch(epoch)` with **0..9** for completed human epochs **1..10**, including resumed runs. Crop RNG is `default_rng(SeedSequence((seed, epoch, index)))`.

The prostate branch selects a true class-5 voxel and a legal crop containing it; it guarantees at least one Prostate voxel when class 5 exists. It does **not** guarantee complete prostate coverage, substantial voxel fraction, boundary context, or Bones/Rectum exposure. Random crops can also contain Prostate; a missing Prostate triggers `random_fallback`. Thus 0.5 is neither the observed positive-patch rate nor a guaranteed half of the 1,430 patches.

With one patch per volume and every index visited once, DataLoader shuffling changes optimisation order but not the `(seed,epoch,index)` crop. Local synthetic replay was compared against the actual shuffled DataLoader across all ten epoch configurations and matched crop origins, mask counts and patch hashes.

### Reproducible measurement, not a historical execution claim

`patch-exposure` requires an E10 checkpoint. It verifies weights-only CPU loading, the approved model/hyperparameters, all saved training-source hashes, three JSON policies, NumPy version, 143-volume order/paths and 26 patient identities. NiBabel/SciPy versions must match saved preprocessing versions. It also calls the existing protected strict model loader without running the model or restoring optimiser/RNG.

For each selected **original Dataset index**, production loading/resampling/normalisation runs once. Scoped mocks loan that processed pair to ten actual `__getitem__()` calls; production RNG, transpose and crop execute unchanged. No alternate sampler is implemented, no random patch is cached, and no production cache is introduced. Full arrays are released before the next case; only counts/metadata remain. This reduces repeated decoding without changing the deterministic target reconstruction.

Exact historical interpretation additionally requires unchanged source data and the saved source/environment/configuration. Original checkpoints do not contain MRI file-content hashes or patch hashes, so matching newly reconstructed targets cannot independently certify that historical source files were unchanged. The appropriate label is **authenticated configuration replay / reconstruction**, not a newly observed historical training run. Hash, policy, path or version mismatches must be investigated using the original environment; never disable checks or edit production files to force a match. CUDA training itself was best-effort deterministic, not bitwise deterministic; that limitation does not change this NumPy sampler's index-keyed target selection.

### Required output and denominators

Full-scope success requires **143 full-volume records, 1,430 unique patch records, 26 patients**, exactly ten patches per volume, and **374,865,920 patch voxels = 1,430 × 64³**. These previously specified accounting totals are now met by the authentic full-Train log. Each patch includes case, original index, human/sampler epoch, DHW origin, mode, six voxel counts, shape and mask SHA-256.

The summaries report:

- Per-class positive-patch count/fraction: `patches with count[c] > 0 / patches examined`.
- Pooled patch voxel fraction: `sum patch count[c] / sum all patch voxels`.
- Positive-patch voxel-count and voxel-fraction distributions (min/Q25/median/Q75/max); absent samples produce null, not invented zero distributions.
- Per-epoch, per-patient and prostate/random/random-fallback branch statistics.
- Largest-class patch counts (argmax of six counts, lower-ID tie break) and patches where Background + Body exceed half the patch voxels. The latter is a descriptive definition, not a proven harmful threshold.
- Source/full-volume coverage separately from sampled-patch coverage. Raw records allow `patch count[c] / full resampled count[c]` and inspection of crop boundaries to identify tiny fragments. Count ratios alone cannot determine adequate anatomical context.

**Real reconstruction rates are now measured**, distinct from the prior synthetic tool tests. Both saved training logs match the ordered pairs, preprocessing/sampling policies and NumPy versions used in this replay. The sampler is model-independent, so these support a common configured target schedule, conditional on the same input bytes; they are not an independent historical patch transcript for either training run.

### Verified real patch exposure (Job 650736)

Presence means `count[class] > 0`. Positive rates divide by **1,430 patches**; pooled fractions divide by **374,865,920 voxels**. Positive-patch distributions exclude absent patches and give **min / Q25 / median / Q75 / max** of `count/262,144`, in percent, using linear percentiles. Rounded display values retain their exact record-derived denominators.

| Class | Positive patches / rate (%) | Pooled voxel fraction (%) | Positive-patch fraction min / Q25 / median / Q75 / max (%) |
| --- | ---: | ---: | --- |
| Bones | 1,230/1,430 / 86.013986 | 9.099350 | 0.001144 / 5.746555 / 10.078812 / 14.882374 / 31.694794 |
| Bladder | 869/1,430 / 60.769231 | 4.032347 | 0.000381 / 1.634216 / 5.116653 / 9.353638 / 45.227814 |
| Rectum | 777/1,430 / 54.335664 | 1.232043 | 0.000763 / 0.840378 / 1.907349 / 3.124237 / 12.720490 |
| Prostate | 880/1,430 / 61.538462 | 1.434959 | 0.002289 / 1.237106 / 1.903343 / 3.044128 / 8.418655 |

### Small-count and partial-target thresholds

These are descriptive count thresholds chosen for this audit, **not clinical cut-offs, hyperparameters or proof of fragmented morphology**. Absent patches are excluded and reported separately. Occupancy categories are nested: tiny=`0 < count < 0.001 x 262,144` (**1-262 voxels**, below 0.1% of a patch); small=`0 < count < 0.01 x 262,144` (**1-2,621 voxels**, below 1%). A second denominator measures captured target fraction: `patch_count / matching_full_resampled_class_count`; below 1% or 10% means little of that volume's class is captured, not necessarily little patch occupancy or loss of a complete anatomical component.

| Class | Absent patches | Occupancy <0.1% | Occupancy <1% (includes tiny) | Capture <1% of full-volume class | Capture <10% (includes <1%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bones | 200 | 36 | 108 | 106 | 662 |
| Bladder | 561 | 52 | 164 | 73 | 207 |
| Rectum | 653 | 51 | 219 | 39 | 106 |
| Prostate | 550 | 27 | 170 | 19 | 51 |

### Sampling branches

**754 prostate-aware / 676 random** patches; no random-fallback records. The 52.727273% realised prostate-branch rate is a seed-specific outcome of configured probability 0.5, not a changed policy. Branch rates divide by their own patch counts; branch pooled voxel fractions divide by **197,656,576 / 177,209,344 voxels**, respectively. Medians are conditional on class-positive patches within that branch.

| Class | Branch | Positive count / branch rate (%) | Pooled voxel fraction (%) | Positive median fraction (%) | Tiny <0.1% / small <1% counts |
| --- | --- | ---: | ---: | ---: | ---: |
| Bones | prostate | 754/754 / 100.000000 | 12.051167 | 11.319351 | 0 / 1 |
| Bones | random | 476/676 / 70.414201 | 5.806939 | 6.193161 | 36 / 107 |
| Bladder | prostate | 677/754 / 89.787798 | 6.415364 | 5.995178 | 23 / 100 |
| Bladder | random | 192/676 / 28.402367 | 1.374365 | 2.572441 | 29 / 64 |
| Rectum | prostate | 663/754 / 87.931034 | 2.056125 | 2.002716 | 34 / 172 |
| Rectum | random | 114/676 / 16.863905 | 0.312876 | 1.323318 | 17 / 47 |
| Prostate | prostate | 754/754 / 100.000000 | 2.491564 | 2.084923 | 2 / 107 |
| Prostate | random | 126/676 / 18.639053 | 0.256437 | 0.998878 | 25 / 63 |

All prostate-aware patches contain Bones and Prostate. Random crops add 126 positive Prostate patches, so **61.538462%** overall positivity is not simply the branch frequency. Only **2/754** prostate-aware patches have tiny Prostate occupancy versus **25/676** random patches. Bones tiny/small occupancy occurs in **0/1** prostate-aware versus **36/107** random patches. Ground-truth exposure improves with targeting; it does not establish learned predictions.

### Epoch variability

Every epoch has **143 patches / 37,486,592 voxels**. Each class cell lists **positive-patch count / pooled voxel percentage**; rates are that count divided by 143. Prostate/random branch counts are given explicitly.

| Epoch | Prostate / random branches | Bones | Bladder | Rectum | Prostate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 76 / 67 | 133 / 10.103159% | 89 / 4.543414% | 82 / 1.339471% | 95 / 1.442361% |
| 2 | 72 / 71 | 124 / 9.797255% | 90 / 4.300946% | 79 / 1.224643% | 86 / 1.396763% |
| 3 | 68 / 75 | 118 / 7.912045% | 80 / 3.772928% | 71 / 1.178763% | 81 / 1.278294% |
| 4 | 72 / 71 | 116 / 8.688901% | 84 / 4.201193% | 73 / 1.123943% | 82 / 1.363645% |
| 5 | 74 / 69 | 113 / 8.086843% | 72 / 3.024057% | 71 / 1.099566% | 80 / 1.315132% |
| 6 | 75 / 68 | 131 / 9.093902% | 85 / 3.346906% | 80 / 1.499445% | 89 / 1.615113% |
| 7 | 80 / 63 | 127 / 8.794958% | 98 / 3.742258% | 81 / 1.129569% | 93 / 1.502262% |
| 8 | 76 / 67 | 119 / 9.336517% | 88 / 4.236312% | 75 / 1.160484% | 86 / 1.332532% |
| 9 | 89 / 54 | 128 / 9.978413% | 99 / 5.164527% | 85 / 1.252170% | 99 / 1.663912% |
| 10 | 72 / 71 | 121 / 9.201506% | 84 / 3.990926% | 80 / 1.312379% | 89 / 1.439571% |

Bones positive rates range **79.020979-93.006993%**, Bladder **50.349650-69.230769%**, Rectum **49.650350-59.440559%**, Prostate **55.944056-69.230769%**. Every epoch contains substantial Bones exposure and positive examples of all four classes. Changing reconstructed crops across epochs can affect losses; the audit does not recover their optimiser ordering or per-class gradients.

### Patient variability

Every patient has positive patches for all four classes. Patients contribute **10-80 patches** (ten per longitudinal volume), so pooled rates weight patients differently. Cells give positive count and patient-specific rate; samples from the same patient/volume are correlated, and no confidence interval or population generalisation is claimed.

| Patient | Patch denominator | Bones positive / rate (%) | Bladder positive / rate (%) | Rectum positive / rate (%) | Prostate positive / rate (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B006 | 20 | 19 / 95.000 | 11 / 55.000 | 12 / 60.000 | 13 / 65.000 |
| B038 | 10 | 10 / 100.000 | 8 / 80.000 | 4 / 40.000 | 4 / 40.000 |
| C032 | 80 | 65 / 81.250 | 45 / 56.250 | 36 / 45.000 | 45 / 56.250 |
| D031 | 80 | 66 / 82.500 | 48 / 60.000 | 48 / 60.000 | 50 / 62.500 |
| G021 | 80 | 71 / 88.750 | 58 / 72.500 | 51 / 63.750 | 50 / 62.500 |
| H007 | 10 | 9 / 90.000 | 8 / 80.000 | 8 / 80.000 | 7 / 70.000 |
| H017 | 80 | 70 / 87.500 | 50 / 62.500 | 48 / 60.000 | 55 / 68.750 |
| J005 | 50 | 44 / 88.000 | 29 / 58.000 | 30 / 60.000 | 35 / 70.000 |
| K019 | 80 | 69 / 86.250 | 40 / 50.000 | 36 / 45.000 | 48 / 60.000 |
| K042 | 10 | 8 / 80.000 | 6 / 60.000 | 3 / 30.000 | 4 / 40.000 |
| M004 | 80 | 71 / 88.750 | 53 / 66.250 | 48 / 60.000 | 50 / 62.500 |
| M013 | 80 | 64 / 80.000 | 42 / 52.500 | 36 / 45.000 | 39 / 48.750 |
| M023 | 10 | 10 / 100.000 | 9 / 90.000 | 9 / 90.000 | 8 / 80.000 |
| M036 | 10 | 9 / 90.000 | 6 / 60.000 | 6 / 60.000 | 5 / 50.000 |
| N010 | 80 | 68 / 85.000 | 55 / 68.750 | 41 / 51.250 | 49 / 61.250 |
| O025 | 80 | 71 / 88.750 | 48 / 60.000 | 48 / 60.000 | 51 / 63.750 |
| R016 | 80 | 65 / 81.250 | 47 / 58.750 | 42 / 52.500 | 48 / 60.000 |
| R024 | 80 | 70 / 87.500 | 52 / 65.000 | 47 / 58.750 | 52 / 65.000 |
| R039 | 80 | 69 / 86.250 | 43 / 53.750 | 35 / 43.750 | 44 / 55.000 |
| S028 | 10 | 9 / 90.000 | 7 / 70.000 | 3 / 30.000 | 6 / 60.000 |
| S033 | 80 | 69 / 86.250 | 54 / 67.500 | 45 / 56.250 | 54 / 67.500 |
| S035 | 10 | 8 / 80.000 | 4 / 40.000 | 4 / 40.000 | 5 / 50.000 |
| T009 | 10 | 10 / 100.000 | 7 / 70.000 | 7 / 70.000 | 7 / 70.000 |
| T014 | 80 | 69 / 86.250 | 43 / 53.750 | 50 / 62.500 | 55 / 68.750 |
| V027 | 80 | 61 / 76.250 | 44 / 55.000 | 33 / 41.250 | 43 / 53.750 |
| W012 | 80 | 76 / 95.000 | 52 / 65.000 | 47 / 58.750 | 53 / 66.250 |

| Class | Patient-positive rate range (%) | Range of patient-specific positive-patch medians (%) | Equal-patient mean positive rate (%) |
| --- | ---: | ---: | ---: |
| Bones | 76.250000-100.000000 | 5.207062-15.923691 | 87.711538 |
| Bladder | 40.000000-90.000000 | 2.245712-11.019325 | 62.711538 |
| Rectum | 30.000000-90.000000 | 1.308250-4.293823 | 54.759615 |
| Prostate | 40.000000-80.000000 | 0.431061-6.096840 | 60.673077 |

Bones' lowest positivity is V027 (**61/80, 76.25%**), still frequent; B038/M023/T009 reach 100%. Prostate positivity is lowest for B038/K042 (**4/10, 40%**) and highest for M023 (**8/10, 80%**). Its positive-patch median occupancy spans **0.431061% (B038)** to **6.096840% (M004)**. Small single-volume denominators should not be treated as precise patient-level performance estimates.

### Exposure versus prediction collapse

**Simple missing-target exposure is contradicted for Bones.** It occupies **9.099350%** of sampled voxels, is present in **1,230/1,430 patches**, and occupies a positive-patch median **10.078812%**. Bladder has lower presence (**869**) and pooled share (**4.032347%**) but achieves appreciable Validation overlap. Thus class frequency or tiny fragments alone cannot explain near-total Bones failure. The two Improved Bones true positives are negligible against 10,144,275 Validation reference voxels.

Prostate is present in **880/1,430 patches**, including all 754 targeted patches, with **5,379,171 sampled voxels**. However its pooled share is **1.434959%**, positive median **1.903343%**, and 170 positive patches contain <1% patch occupancy. These quantify residual imbalance and partial exposure, not proof that the sampler is the root cause. Rectum similarly has 777 positive patches and 1.232043% pooled share.

Background + Body occupy **84.201301%** of sampled voxels and exceed half of the patch in **1,423/1,430 (99.510490%)** samples; one of those two is the largest class in every patch. This is measured target imbalance, **not measured CE dominance or gradient weakness**: CE contributions also depend on prediction errors, and foreground Dice weights classes equally. Counts alone could not establish target probabilities or argmax competition. The subsequent Job 650924 probe (Section F) measured both on two selected Train crops and one Validation center crop; it does not establish causes or population-wide/full-volume behaviour. Parameter-gradient attribution remains unmeasured, and no retraining is recommended.

## E. Loss and gradient analysis

### Actual objective

For N batch/spatial voxels, target `y_v` and six-channel probability `p_cv = softmax(logits, dim=1)`:

`CE = -(1/N) sum_v log(p[y_v,v])`

`D_c = (2 sum_v p_cv * 1[y_v=c] + epsilon) / (sum_v p_cv + sum_v 1[y_v=c] + epsilon)`, c=1..5, epsilon=1e-5.

`L_Dice = 1 - (1/5) sum_c D_c`

`L_base = 0.5 CE + 0.5 L_Dice`

`L_Improved = (L_main + 0.5 L_half + 0.25 L_quarter) / 1.75`.

CE uses unreduced voxel losses followed by `.mean()`, equivalent to unweighted mean CE here. Dice sums batch/spatial axes, then equally weights five foreground classes. Consequently it is inaccurate to describe the entire objective as simply voxel-frequency weighted. CE contribution per true class is `n_c/N * mean NLL on class-c voxels`; frequency alone does not establish dominance, since rare categories may have large errors. Actual component/gradient magnitudes are unmeasured.

Targets are `torch.long`; Dice's `target == c` indicators are mathematically equivalent to channels of a six-class one-hot mask. This does not resolve the separate course requirement about the stored mask representation. Auxiliary logits are trilinearly interpolated to the full target grid with `align_corners=False`; class-index masks remain unchanged.

### Important target-absent Dice behaviour

When class c is absent, `T_c=0`, `P_c=sum_v p_cv`:

`D_c = epsilon / (P_c + epsilon)` and the derivative of its contribution to `L_Dice` with respect to each `p_cv` is `epsilon / [5 (P_c + epsilon)^2]`.

For `P_c >> epsilon`, its score is near zero and its contribution to Dice loss is near 0.2, despite a very small Dice probability gradient. Four absent foreground classes can contribute approximately 0.4 to the **combined** loss after its 0.5 Dice weighting. This is a conditional limiting calculation, not a measured floor of the real training run; predictions, smoothing and present classes still contribute.

A genuine synthetic 16³ Body-only example with uniform six-class probabilities verified a derivative of approximately **4.2915e-12** per probability voxel for an absent class's Dice contribution. CE still gives a false-positive penalty. This establishes an implemented numerical property, **not** vanishing total gradients or the root cause of real class omission. Softmax coupling also means this probability-space derivative is not a network parameter-gradient measurement.

### Direct local test evidence

Final restored utility: **32 new tests + 16 unchanged Patch Dataset tests = 48 tests in 3.292s, OK**, CPU-only local WSL. PyTorch **2.12.1+cu130**, NumPy **2.4.4**, SciPy **1.17.1**; CUDA hidden. Local NiBabel is missing, so an in-memory import-only shim and mocked preprocessing were used. No real NIfTI or checkpoint binary was tested; protected-loader tests verify delegation using mocks.

The independent log-softmax CE / vectorised one-hot Dice reference matched production scalar components (`rtol=1e-6`, `atol=1e-7`) and logit gradients (`rtol=2e-5`, `atol=2e-8`). Correct channels beat a cyclic permutation for all six IDs. Body-biased synthetic predictions received finite corrective gradients for true Bones, Rectum and Prostate. Main, half and quarter outputs received finite nonzero gradients under the weighted production loss. Integer targets remained unchanged. Class metrics/confusion indices agreed. No optimiser update or model training occurred in these checks.

Additional tests cover raw/patch counting, distributions/dominance, replay reproducibility and DataLoader parity, bounded original-index selection, foreground fragments/fallback, invalid masks/identities, production preprocessing delegation, invalid CLI bounds rejected before discovery, checkpoint policy/hash rejection, and missing/duplicate/partial/malformed historical logs. Windows `compile()` syntax checks passed without bytecode writes. A first Windows-to-WSL inline runner failed due to quoting before tests began; the standard-input runner resolved this. Earlier interrupted-turn results are not substituted for the final restored-file test result.

### Authentic available training-loss trajectory

The four downloaded original E10 logs are now locally authenticated; [file hashes, events, epoch/component/runtime values and loss plot](../evidence/failure-cases.md#original-e10-log-authentication) preserve the evidence. Standard records **epochs 4–10**, 1,001 new updates, after `resume_completed_epochs=3`; Improved records **epochs 2–10**, 1,287 new updates, after `resume_completed_epochs=1`. Each epoch has 143 sequential step records matching all reported component means; final counters are ten epochs / 1,430 updates. There are no malformed/duplicate/error records or missing required summaries. All seven Standard and nine Improved training source hashes match local files, and both use identical ordered 143 training pairs, shared hyperparameters and preprocessing policies. This strengthens source provenance without loading checkpoint binaries or proving bitwise reproducibility.

Standard total loss changes from **0.7917109148 (E4) to 0.6992273720 (E10)**; Improved from **0.8104826800 (E2) to 0.6111517446 (E10)**. Both trajectories are non-monotonic. Standard Soft Dice loss rises **0.7609311405 → 0.7699025190** from E9 to E10; Improved rises **0.7058774162 → 0.7147218781**. These observations do not establish a plateau, rare-class learning or that extending training will solve collapse. Improved components include the weighted main/auxiliary objectives; Standard components are main-only. Their absolute magnitudes are not equivalent model-quality measures, and soft training Dice is not hard full-volume DSC.

**Missing:** original Standard E1–3 and Improved E1 stdout segments, including setup, step and completed-epoch events. The existing `summarize_loss_logs()` was run locally on the genuine files and returns `INCOMPLETE` with exactly these missing epochs; its available step/component checks pass. No values are reconstructed. The previously student-reported Standard E1 total **0.9565964858** remains historical conversation evidence, not a plotted original-log event. The **1.92833924** single-step CE/SGD smoke loss is not comparable to epoch CE+Dice/AdamW.

Both original E10 full-validation files are now audited: matching six patients / 38 identities, affine/shape/spacing, 64³ windows, stride 32, mean-softmax aggregation and absent-class policy. Per-volume/patient/pooled/macro arithmetic agrees, and all six true classes occur in every Validation volume. Rectum/Prostate still have exactly zero predicted voxels; Improved's three Bones predictions give two TP and one FP. **Validation truth counts do not measure training-volume or sampled-patch frequencies.** No per-class training components, sampled targets, probability margins or gradient attribution are recorded in these training logs. The Train exposure measurement is complete as configuration replay: its substantial Bones/Prostate exposure weakens simple absence explanations. These original training/Validation logs contain no per-class probability or parameter-gradient records; the later Job 650924 probe supplies selected-patch probabilities in Section F, while parameter-gradient strength remains unmeasured.

Standard full validation names `standard_unet_last.pt`, whereas paired-case/replay references name `standard_unet_epoch10.pt`. Both record E10 / 1,430 updates; identical binary weights are not established by the logs. Do not bypass the protected checkpoint/source checks. Earlier E3 validation aggregation/setup evidence also remains incomplete. These are evidence limitations, not claims that earlier experiments do not exist.

## F. Prediction competition and probability analysis

**The student-run probability probe is now authenticated and analysed below (job 650924).** It measures selected-patch suppression and competition; parameter gradients and causes remain unresolved. Original preparation tests and proposed commands are preserved as historical records, followed by the new real evidence. No new inference was executed in this documentation analysis.

### Original first-probe plan: three fixed patches, six main-head forwards

Use both protected E10 loaders (`validate.load_checkpoint_model()` and `validate_improved.load_checkpoint_model()`), unchanged preprocessing, six channel IDs and float32, with `eval()` / `torch.inference_mode()`. Read `weights_only=True`, strict state/source/policy checks and explicit snapshot identities; do not restore optimiser state or change thresholds. Standard snapshot identity is recorded above; Improved paired-case metadata records `improved_unet_epoch10.pt`, SHA-256 `4861c5d65a780af7c8ab67e59715ec37f70ca81dee314f3b80a6a26ad031c0d6`. Locally unavailable binaries must be checked at execution; compatibility or hash mismatches are errors, never waived. Processing on Rangpur or any real-data inference requires separate authorisation; this was a **NOT EXECUTED plan when drafted**; the student subsequently executed job 650924, authenticated below.

The two Train crops below were selected from this authentic reconstruction because **all four investigated foreground classes are present**, rather than selecting an empty patch. The earlier proposed K019 E1 crop is now known to contain none of classes 2-5; E10 omits Rectum. They would not answer the intended GT-conditional question. Reproduce the two recorded crops using the production Dataset and their original sampler epochs/indices; require the listed target counts and mask hash to match before inference. XYZ-to-DHW remains `(2,1,0)`; this is tensor/crop ordering, not new physical resampling.

| Split / case | Human / sampler epoch | Original Dataset index | Crop origin DHW | Bones / Bladder / Rectum / Prostate counts | Mask SHA-256 |
| --- | --- | ---: | --- | --- | --- |
| Train / K019_Week1 | 4 / 3 | 42 | `[39, 92, 93]` | 22665 / 5207 / 6149 / 17149 | `103f068d01591b1f66498210b7f35663fce150880edbf5de49ec4f48e913df92` |
| Train / S035_Week0 | 6 / 5 | 117 | `[43, 122, 107]` | 22864 / 9266 / 11924 / 4448 | `6ec4bea2d93d6f9b508c238cfd4fd57237477d7769b9fad0f5e2c71ff86e5b63` |

**Validation case:** B040_Week0, the existing deterministic center crop: processed XYZ `(239,239,128)`, origin DHW `[32,87,87]`, patch 64³. Earlier student DataLoader evidence records labels `[1,2,3,4,5]` in that crop; verify current geometry and positive counts anew rather than assuming unchanged data. Use the original validation center-selection policy, not a tuned crop. Record the new mask hash and exact denominators. If geometry or presence differs, stop and document the difference before replacing the case.

Run each patch through each architecture's **main head**, six patch forwards total, one CPU-preprocessed pair/patch at a time. CPU-only is possible; real GPU inference requires an allocated compute node. Freeze patch inputs across architectures and retain output statistics externally, not MRI/probability arrays in Git. Record original MRI/mask file SHA-256 at the probe to bind that new input evidence; such hashes do not retroactively prove the training inputs were unchanged. Capture checkpoint/source/configuration hashes, versions, case/split/index/epoch/origin/shape/affine, runtime and device metadata.

For each patch and each model, collect exactly:

1. Finite logits `[1,6,64,64,64]`, integer GT counts `n_c`, predicted counts, a truth-row/prediction-column 6x6 confusion matrix and GT-conditional recall for each class. These are selected-patch diagnostics, **not unbiased volume Validation scores**.
2. On true class-c voxels, target probability `p_c`: mean, min, Q10, median, Q90, Q99 and max. Include voxel denominators; use null for empty conditional groups. Also record target-class rank frequencies (1-6), winning competing class counts and ties using the existing argmax convention.
3. On those same voxels, logit margin `z_c - max_{k!=c} z_k` and probability margin `p_c - max_{k!=c} p_k`, with the same quantiles and fraction strictly greater than zero. Negative margins establish competition loss at those voxels, not its cause.
4. Per-class mean NLL from stable log-softmax and contribution `n_c/N * mean_NLL_c` to unweighted CE; absent-class contribution is zero, conditional NLL is null. Record `sum(p_c)`, GT counts, intersections and the existing per-class Soft Dice formula/smoothing, plus main-head CE/Dice/total. Do not equate probability mass, loss contribution or target fraction with parameter-gradient strength. Improved main-head values here are distinct from its historically weighted auxiliary totals.

No backward pass, optimiser update, auxiliary training, full Validation sweep or Test processing belongs to this first probe. If auxiliary loss attribution later becomes necessary, it is a separately reviewed extension using the unchanged logit-upsampled loss.

### Interpretation and strictly conditional extensions

- Collapse already on both foreground-rich **Train** patches strengthens a shared optimisation/objective hypothesis; it does not prove small gradients or that more epochs are the solution.
- Correct Train predictions but failure on the same classes in **Validation** makes generalisation/context effects a higher priority. Both windows may still be unrepresentative; no population inference follows from three crops.
- Appreciable `p_c` with consistently negative margins shows target channels lose to competing classes; very low `p_c` shows stronger suppression. Both are measured manifestations, not proven training root causes.
- Useful predictions in the Validation crop despite zero full-volume class predictions require a separately authorised **one-case B040 full-volume probability-aggregation parity check**, using 64³/stride 32/mean-softmax. Account for each final voxel once; compare the final argmax with the unchanged evaluator before interpreting probabilities. Do not sum overlapping window voxels as unique volume observations.

If a second Validation example is needed, use **K008_Week0**, whose original full-volume record contains 11,114 Prostate voxels and no predictions. Predeclare a deterministic center window; if a class is absent, report null rather than silently picking a better crop. A separately agreed GT-informed window may be used only as a clearly labelled diagnostic, not as a new unbiased metric. No extra case is required before reviewing the first six forwards. Any component-gradient probe (no optimiser) needs a separate scope and authorisation; inference-only measurements cannot establish gradients. No Test tuning or retraining is recommended.

### Phase 4 implementation and local verification (historical preparation)

The isolated [checkpoint_probabilities.py](../../hipmri_3d_improved_unet_hard/diagnostics/checkpoint_probabilities.py) and [test_checkpoint_probabilities.py](../../hipmri_3d_improved_unet_hard/diagnostics/test_checkpoint_probabilities.py) implement this fixed scope. Production Python/configuration files are not edited. The original exposure log is pinned by its SHA-256, parsed and independently summarised again before its fixed records are used. This new reader was executed locally against the actual downloaded log: record counts **1 setup / 143 full volumes / 1,430 patches / 1 summary**, PASS, without image decoding or model inference.

The initial expanded CPU suite passed **33/33 tests** in the existing WSL environment (PyTorch **2.12.1+cu130**). The final suite, also exercising both actual protected loaders with compatible synthetic state dictionaries, passed **34 tests in 4.935s, OK**. NiBabel is unavailable there: the tests explicitly install an **in-memory, test-only import shim**, and mock preprocessing/real checkpoint inputs. They test arithmetic, real Dataset center-crop logic on synthetic arrays, fixed-origin/count/hash rejection, full training-source/configuration protections, strict-loader delegation/rejection, checkpoint tensor comparison, one-forward inference semantics, three-patch/six-call orchestration, no backward/optimizer/save, and zero-forward preflight. These are synthetic results, not acceptance of remote E10 checkpoint files or real MRI replay. CLI `--help` and Python syntax were checked. Windows/WSL quoting errors in two inspection commands were corrected; no experiment was run by those failed commands.

Documentation checks passed: **20 relative links**, balanced fences, UTF-8/newline/whitespace checks (including these untracked files), **two PowerShell command blocks** parsed without execution, and **six Bash blocks** parsed with `bash -n` without execution. An initial Bash syntax-only check was affected by CRLF in the Windows pipeline; checking each block with Linux-normalised newlines passed. `git diff --check` passed for tracked changes; it does not inspect the new untracked files. Hash comparison preserved **42 pre-existing files**, including **28 Python files**, and all **five original external JSONL logs**. Only this diagnosis document and the two new isolated files were changed/created in Phase 4; earlier pending Git changes were retained.

**Actual interfaces reused:** `HipMRIPatchDataset`, its original `set_epoch()` and `__getitem__()`, `class_collapse.production_preprocess()` and `validate_replay_provenance()`, both unchanged protected `load_checkpoint_model()` functions, `train.configure_reproducibility()` and `combined_loss()`, and existing confusion/metric functions. Each real pair is preprocessed once. Scoped interception of preprocessing return values then runs the production crop algorithm twice to check exact repeatability, as in the approved exposure replay. No replacement sampler or persistent cache is implemented. Both models receive the same resulting patch; Improved forward is called with its default main-only output.

The Training indices **42 / 117**, epochs, branch **prostate**, origins, all six target counts, mask hashes and full resampled geometry/counts must match the authenticated records before inference. Validation B040 is original Dataset index **0**, center mode, processed XYZ `(239,239,128)`, origin `[32,87,87]`, and positive class IDs `[1,2,3,4,5]`. Its reference is earlier acceptance evidence, **not a Train exposure record**; a new mask hash is recorded. Real reconstruction was pending at preparation; all three selections are now authenticated below. Current source MRI/mask and patch hashes are recorded at execution; they do not prove unchanged historical training image bytes.

#### Checkpoint authentication and filename distinction

The probe uses the following student-recorded snapshots, never a silently substituted `last` file:

| Role | Recorded remote file | Expected SHA-256 |
| --- | --- | --- |
| Standard probe / paired predictions / exposure replay | `/home/Student/s4906923/hipmri_training_runs/standard_unet_epoch10.pt` | `f2a87f9c73e068719e2f1d878291116d26320be7c5176827f78537fc4fb4f9cf` |
| Standard original full Validation | `/home/Student/s4906923/hipmri_training_runs/standard_unet_last.pt` | Historical file hash **not recorded in the available Validation JSONL**; compute its current hash |
| Improved probe / original full Validation / paired predictions | `/home/Student/s4906923/hipmri_training_runs/improved_unet_epoch10.pt` | `4861c5d65a780af7c8ab67e59715ec37f70ca81dee314f3b80a6a26ad031c0d6` |

These paths were read from authentic local execution records; the actual remote binaries have not been opened by Codex. Preflight pins snapshot hashes, loads with `weights_only=True` on CPU, requires E10/1,430 updates, exact architecture, all training-source hashes (including `train.py`/`patch_dataset.py`), frozen JSON policies, training-pair order/paths and exact recorded NumPy/NiBabel/SciPy/PyTorch/CUDA-build versions. It then delegates to the unchanged protected strict state loaders. A differing file or production hash is a failure, not grounds to replace protected source or weaken checks.

Both Standard files undergo these protections and an exact state-dictionary tensor comparison. **Different file bytes are reported explicitly; different weights block the probe.** Equal tensor values despite different serialised bytes permit a qualified comparison. Neither equal current bytes nor equal current tensors proves what `last` contained at the historical Validation run, because its historical hash is absent. No optimizer or RNG checkpoint state is restored. Existing training reproducibility settings remain best-effort/warn-only; repeated sampler/statistics calculations are checked, but the six-forward budget does not include repeat GPU forwards and cannot establish bitwise GPU replay.

#### Exported JSONL schema and statistical scope

Success in inference mode emits one `preflight`, one `setup`, three `patch`, six `probability`, and one final `summary` record. CPU-only preflight emits `preflight` plus `summary` with zero forwards/arrays. Failures emit `error`, `status=FAIL`, exception details and non-zero exit; no final PASS is emitted.

| Event / field | Meaning |
| --- | --- |
| `preflight.checkpoints`, `standard_snapshot_vs_validation` | Paths, file/config/source hashes, versions, E10 metadata and current Standard file-byte/tensor equality; no weights exported |
| `patch.metadata` | Split, identity, original index/epoch, DHW/XYZ origin, target counts/hash, repeated-sampler equality, full XYZ affine and crop-adjusted XYZ affine, units and source-file/patch hashes |
| `probability.statistics.predicted_class_counts`, `confusion_counts_truth_rows_prediction_columns` | All-patch counts and truth-row/prediction-column confusion; exactly one 64-cube patch |
| `classes[c].conditional` | Only GT class-c voxels: denominator, argmax recall, true-class probability mean/min/P10/median/P90/P99/max, ranks 1-6 and first/second/below-second fractions, winning/competing class counts, most frequent competing winner, ties, mean NLL and margins |
| Probability margins | Both nonnegative `winner_minus_true_probability` and signed `true_minus_best_other_probability`/logit margins; ranks resolve ties by lowest channel ID, matching argmax |
| `classes[c].soft_dice`, `objective_soft_dice_score` | All-patch `(2 sum(p_c t_c)+1e-5)/(sum(p_c)+sum(t_c)+1e-5)`; reported Soft Dice is null for absent GT, while the mathematical objective score is separately recorded for loss accounting |
| `ce_contribution_all_patch`, `main_head_loss` | `n_c/N * mean(-log p_c)` and unchanged main-head CE/foreground-Dice/total; absent conditional NLL is null and CE contribution zero; background is excluded from foreground-Dice loss |
| `forward_seconds`, `gpu_memory` | Synchronized one-forward time excluding CPU statistics; peak allocated/reserved CUDA bytes include this model/input/forward configuration, not an isolated layer or full-training memory |
| Final `summary` | PASS only for three patches/six forwards; zero optimizer updates, backward passes and Test-array access |

All probability/NLL calculations use CPU float32 softmax/log-softmax on the returned GPU logits; statistics quantiles use float64. Probability sums and finite outputs are checked. Recomputing statistics on the same logits must agree exactly. No probability/logit arrays, MRI panels or images are exported. B040 single-center-patch probabilities are **not equivalent to 64-cube/stride-32 mean-softmax full-volume predictions**, and patch recall is not full-volume Dice. No thresholds, auxiliary losses, backward passes or gradients are added.

#### Historical proposed manual execution commands

All commands below are prepared instructions, not historical transcripts or jobs run by Codex. Run transfer/authentication yourself and enter credentials only in the interactive terminal. Stop at any failed preflight, test, source hash or checkpoint check; do not overwrite production files to force compatibility.

**1. Windows PowerShell: verify local files, compare diagnostic hashes and transfer only the two new isolated files plus the original log.** The remote verification workspace and original `class_collapse.py` must already exist from the completed exposure audit. The first SSH command refuses to overwrite an existing evidence log; if it already exists, verify its recorded SHA-256 and skip the log SCP.

```powershell
$repo = 'D:\Desktop\UQ26s2-COPMP3710-Pattern-Recognition-and-Analysis\Final project'
$diag = Join-Path $repo 'recognition\hipmri_3d_improved_unet_hard\diagnostics'
$exposureLog = Join-Path $env:USERPROFILE 'Downloads\hipmri_e10_evidence\patch_full_650736.jsonl'
$login = 's4906923@rangpur.compute.eait.uq.edu.au'
$sshOptions = @('-o', 'PubkeyAuthentication=no', '-o', 'PreferredAuthentications=password,keyboard-interactive')
$files = @((Join-Path $diag 'checkpoint_probabilities.py'), (Join-Path $diag 'test_checkpoint_probabilities.py'), $exposureLog)
$files | ForEach-Object { if (!(Test-Path -LiteralPath $_ -PathType Leaf)) { throw "Missing file: $_" } }
Get-FileHash -Algorithm SHA256 -LiteralPath $files
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $diag 'class_collapse.py')
ssh @sshOptions $login 'test -d /home/Student/s4906923/comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard/diagnostics && mkdir -p /home/Student/s4906923/hipmri_probability_evidence && test ! -e /home/Student/s4906923/hipmri_probability_evidence/patch_full_650736.jsonl'
if ($LASTEXITCODE -ne 0) { throw 'Remote workspace check failed or evidence log already exists; inspect before transfer.' }
scp @sshOptions $files[0] $files[1] "${login}:/home/Student/s4906923/comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard/diagnostics/"
if ($LASTEXITCODE -ne 0) { throw 'Diagnostic transfer failed.' }
scp @sshOptions $exposureLog "${login}:/home/Student/s4906923/hipmri_probability_evidence/patch_full_650736.jsonl"
if ($LASTEXITCODE -ne 0) { throw 'Exposure log transfer failed.' }
ssh @sshOptions $login
```

**2. Rangpur CPU preflight and synthetic tests.** Purpose: establish source/config/runtime/epoch/model compatibility and current snapshot-versus-Validation tensor equality before GPU work. Expected: pinned hashes agree, 34 tests end `OK`, and preflight ends `summary PASS`, zero forwards/arrays. This does not establish real crop replay or probabilities. Compare remote diagnostic hashes with the PowerShell outputs; protected production hashes are checked against checkpoint metadata by the preflight itself.

```bash
cd "$HOME/comp3710_hipmri_verify"
conda activate torch
probe_run=$(mktemp -d "$HOME/hipmri_probability_phase4.XXXXXX")
printf 'Diagnostic output directory: %s\n' "$probe_run"
sha256sum "$HOME/hipmri_training_runs/standard_unet_epoch10.pt" \
  "$HOME/hipmri_training_runs/standard_unet_last.pt" \
  "$HOME/hipmri_training_runs/improved_unet_epoch10.pt" \
  "$HOME/hipmri_probability_evidence/patch_full_650736.jsonl"
sha256sum recognition/hipmri_3d_improved_unet_hard/diagnostics/{class_collapse,checkpoint_probabilities,test_checkpoint_probabilities}.py
CUDA_VISIBLE_DEVICES='' "$HOME/miniconda3/envs/torch/bin/python" -B -m unittest \
  recognition.hipmri_3d_improved_unet_hard.diagnostics.test_checkpoint_probabilities -v
test $? -eq 0 || { echo 'STOP: CPU tests failed'; return 1 2>/dev/null || exit 1; }
CUDA_VISIBLE_DEVICES='' "$HOME/miniconda3/envs/torch/bin/python" -B -m \
  recognition.hipmri_3d_improved_unet_hard.diagnostics.checkpoint_probabilities \
  --exposure-log "$HOME/hipmri_probability_evidence/patch_full_650736.jsonl" \
  --standard-checkpoint "$HOME/hipmri_training_runs/standard_unet_epoch10.pt" \
  --standard-validation-checkpoint "$HOME/hipmri_training_runs/standard_unet_last.pt" \
  --improved-checkpoint "$HOME/hipmri_training_runs/improved_unet_epoch10.pt" \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --preflight-only >"$probe_run/preflight.jsonl" 2>"$probe_run/preflight.err"
preflight_status=$?
cat "$probe_run/preflight.err"
tail -n 1 "$probe_run/preflight.jsonl"
test "$preflight_status" -eq 0 || { echo 'STOP: provenance preflight failed'; return 1 2>/dev/null || exit 1; }
```

The `--preflight-only` run may report unequal Standard file hashes with equal tensors. Retain that distinction and the missing historical Validation hash. A tensor mismatch or snapshot/source/version mismatch is a NO-GO; do not submit the next block.

**3. Submit exactly one bounded A100 job after successful CPU checks.** The job repeats all preflight protections, validates real crops before their model calls and performs exactly six main-head forwards if all checks succeed. No `--mem=4G`, training, Test access or extra forward is requested. Slurm sets the allocated GPU visibility; do not set `CUDA_VISIBLE_DEVICES=''` here.

```bash
probe_job=$(sbatch --parsable \
  --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:20:00 --job-name=hipmri-probability \
  --output="$probe_run/probability_%j.jsonl" --error="$probe_run/probability_%j.err" \
  --wrap='cd "$HOME/comp3710_hipmri_verify" && CUBLAS_WORKSPACE_CONFIG=:4096:8 "$HOME/miniconda3/envs/torch/bin/python" -B -m recognition.hipmri_3d_improved_unet_hard.diagnostics.checkpoint_probabilities --exposure-log "$HOME/hipmri_probability_evidence/patch_full_650736.jsonl" --standard-checkpoint "$HOME/hipmri_training_runs/standard_unet_epoch10.pt" --standard-validation-checkpoint "$HOME/hipmri_training_runs/standard_unet_last.pt" --improved-checkpoint "$HOME/hipmri_training_runs/improved_unet_epoch10.pt" --dataset-root /home/groups/comp3710/HipMRI_Study_open --run-six-forwards')
submission_status=$?
test "$submission_status" -eq 0 || { echo 'STOP: submission failed'; return 1 2>/dev/null || exit 1; }
probe_job=${probe_job%%;*}
printf 'Job: %s\nOutput directory: %s\n' "$probe_job" "$probe_run"
```

**4. After completion, inspect scheduler, errors and aggregated statistics.** Expected: main job `COMPLETED 0:0`, six probability records, three patch records, final PASS/6. Warnings must be retained and interpreted rather than assuming stderr must always be empty. On any exception or incomplete count the result is not accepted.

```bash
sacct -j "$probe_job" --format=JobID,State,ExitCode,Elapsed
cat "$probe_run/probability_${probe_job}.err"
"$HOME/miniconda3/envs/torch/bin/python" -B - "$probe_run/probability_${probe_job}.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding='utf-8')]
assert not any(r.get('status') == 'FAIL' for r in rows), 'Failure record present'
assert rows[-1]['event'] == 'summary' and rows[-1]['status'] == 'PASS'
assert rows[-1]['main_head_forward_passes'] == 6
patches = [r for r in rows if r['event'] == 'patch']
results = [r for r in rows if r['event'] == 'probability']
assert len(patches) == 3 and len(results) == 6
for r in results:
    print(r['model'], r['case'])
    for c in (2, 3, 4, 5):
        row = r['statistics']['classes'][c]
        print(json.dumps({'class': row['name'], 'gt': row['gt_voxels'],
                          'predicted_all_patch': row['predicted_voxels_all_patch'],
                          'conditional': row['conditional'], 'soft_dice': row['soft_dice']}))
print('PASS: six-forward diagnostic JSONL scope checked; interpretation still required')
PY
```

**5. Download aggregate records to a new Windows directory outside Git.** Set the remote path from the printed `probe_run`, not an invented historical path. This copies only the preflight/probability JSONL and error logs generated in that directory.

```powershell
$login = 's4906923@rangpur.compute.eait.uq.edu.au'
$sshOptions = @('-o', 'PubkeyAuthentication=no', '-o', 'PreferredAuthentications=password,keyboard-interactive')
$remoteRun = Read-Host 'Paste the absolute diagnostic output directory printed by Rangpur'
if ($remoteRun -notmatch '^/home/Student/s4906923/hipmri_probability_phase4\.[A-Za-z0-9]+$') { throw 'Unexpected output directory.' }
$download = Join-Path $env:USERPROFILE ('Downloads\hipmri_probability_phase4_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $download -ErrorAction Stop | Out-Null
scp @sshOptions -r "${login}:$remoteRun" $download
if ($LASTEXITCODE -ne 0) { throw 'Result download failed.' }
Get-ChildItem -LiteralPath $download -Recurse -File
```

#### Interpretation gate

- **A: low probabilities even on the two Train crops.** Low GT-conditional probability, low rank, negative margins and low recall support training-patch suppression under these checkpoints. They do not identify weak gradients, label defects or insufficient epochs as its cause.
- **B: meaningful probability loses competition.** Higher target probabilities or frequent second rank, but another class wins and margins are negative, support relative class competition as the observed manifestation. No new threshold or prediction rule is introduced.
- **C: Train recognition exceeds selected Validation recognition.** Better Train recall/ranks/probabilities with worse B040 results support a selected-patch generalisation/context hypothesis, not a population estimate. Useful B040 patch predictions with absent full-volume predictions trigger the separately reviewed aggregation-parity extension, not an immediate training change.

These explanations can coexist. This was the interpretation gate before student execution. Job 650924 and the aggregate audit below now supply real probability evidence; gradient attribution and causality remain unresolved. Original command examples are not claimed as verbatim submission transcripts unless separately supplied.

### Authenticated real probability results

**Original evidence:** `%USERPROFILE%\Downloads\hipmri_e10_evidence\prob_probe_650924.jsonl`, **133,971 bytes**, SHA-256 `d41d06f26e8fc4a765ec66fb1cb1dc8600a7e718d3234918b26f3b6a6859c200`. This file was read directly locally; no checkpoint or MRI was decoded in this analysis. Job **650924**, Slurm **COMPLETED 0:0 / 25 seconds**, is student-supplied scheduler evidence, not an independently queried scheduler result. The log records host `a100-a` and NVIDIA A100-PCIE-40GB.

**Authentication PASS:** exactly **1 preflight + 1 setup + 3 patch + 6 probability + 1 final PASS summary** (12 records), no duplicate records/model-case pairs, missing event or error. Summary records six main-head forwards, zero backward passes, zero optimiser updates and zero Test arrays. Independent local arithmetic/source checks passed **1,271 assertions**, plus five supplementary checks of all exported floating-point values, exact fixed selections and the six model/case identities.

Checks cover confusion row/column/diagonal totals; GT counts; ranks, recall and competing winners; hard Dice/IoU; CE contributions; Soft Dice reconstructed from GT-conditional mean probability, GT count and whole-patch probability mass; foreground reduction and combined main-head loss. Distribution bounds/order, NLL bounds, omitted-class margin signs, crop-affine equations, source/configuration hashes and prior-log parity also pass. Numeric comparisons use `atol=1e-6, rtol=1e-6`, with an additional 0.1-voxel absolute allowance for summed float32 probability mass. **No numerical discrepancy was found within these checks.**

This is aggregate verification, not an independent replay. Raw logits/probability arrays were intentionally not exported, so recorded means/medians/quantiles, NLL and margins cannot all be independently recalculated from per-voxel data. Their available aggregate relationships and bounds were checked. Recorded per-voxel softmax-sum errors below 1e-6 and repeated-statistics flags are remote execution evidence. Recalculation on the same logits does not demonstrate repeatable GPU forwards; best-effort/warn-only reproducibility remains.

#### Fixed-patch and checkpoint provenance

| Case | Split / index | Human / sampler epoch | Origin DHW / branch | GT counts 0,1,2,3,4,5 |
| --- | --- | --- | --- | --- |
| K019_Week1 | Train / 42 | 4 / 3 | [39,92,93] / prostate | [0,210974,22665,5207,6149,17149] |
| S035_Week0 | Train / 117 | 6 / 5 | [43,122,107] / prostate | [0,213642,22864,9266,11924,4448] |
| B040_Week0 | Validation / 0 | not applicable / 0 | [32,87,87] / center | [0,156929,29715,50902,17493,7105] |

Both Train selections exactly match the exposure audit's index, epoch, origin, branch, six counts and mask SHA-256; full geometry matches their exposure-volume records. B040 is not a Train exposure record: its source/resampled shapes, affine and unit provenance match both original Validation logs. Its origin independently matches the center-crop formula on processed XYZ `(239,239,128)`; new mask SHA-256 is `b20c7fa010b156dcdf2cd582006b018aa4e78eeb108e28c25408be8e49ff2b4a`. Each patch affine equals full-grid affine times the reversed DHW-to-XYZ crop translation. Each patch has **262,144 voxels**, with no true Background; Background-conditional metrics are null/not evaluable.

Preflight configurations equal the original Standard/Improved E10 training configurations; E10/1,430 metadata equals the original Validation metadata. Seven/nine training-source hashes, three frozen preprocessing/patch JSON policies, both configuration digests and two diagnostic-source hashes match local files. Common train-pair order/policies match exposure. Versions are PyTorch 2.13.0+cu130, CUDA build 13.0, NumPy 2.4.6, NiBabel 5.4.2, SciPy 1.17.1.

The probe uses the remote snapshots above: Standard SHA-256 `f2a87f9c73e068719e2f1d878291116d26320be7c5176827f78537fc4fb4f9cf`; Improved `4861c5d65a780af7c8ab67e59715ec37f70ca81dee314f3b80a6a26ad031c0d6`. Preflight records **identical current Standard snapshot and last-file SHA-256 and state tensors**. This resolves current-file equality in the supplied execution record, not historical `last.pt` bytes at job 649669, whose log has no file hash. Binaries remain unavailable locally. New source-image/patch hashes bind this probe's inputs, not unchanged historical training files.

#### Class-conditioned probability and overlap measurements

Probability/rank/recall/margin/NLL statistics below use **only true-class voxels**. The dominant *incorrect* winner is **Body (1) in every row**, excluding correct wins from its tally. Mean gap is `max(p)-p_true`, nonnegative and zero on correct wins; logit margin is signed `z_true-max(z_other)`.

Soft Dice uses **all patch voxels** in its denominator: `(2*sum_GT(p_c)+1e-5)/(sum_all(p_c)+n_c+1e-5)`. Each foreground score enters `1-mean(scores_1..5)` equally. This is not hard-label Validation Dice. Improved rows describe its main head, not its historical auxiliary-weighted training loss. Display values are rounded; the external JSONL retains precision.

**K019_Week1**

| Model / class | GT voxels | Probability mean / median | TP / recall (%) | Mean gap | Mean logit margin | Mean NLL | Soft Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard / Bones | 22,665 | 0.171074 / 0.178318 | 0 / 0.000 | 0.588690 | -1.561555 | 1.840775 | 0.152903 |
| Standard / Bladder | 5,207 | 0.574445 / 0.678739 | 3,504 / 67.294 | 0.148871 | +0.429442 | 0.839078 | 0.429297 |
| Standard / Rectum | 6,149 | 0.019271 / 0.018372 | 0 / 0.000 | 0.745867 | -3.741381 | 4.013479 | 0.025484 |
| Standard / Prostate | 17,149 | 0.040148 / 0.039619 | 0 / 0.000 | 0.654318 | -2.914097 | 3.284971 | 0.064295 |
| Improved / Bones | 22,665 | 0.199654 / 0.203580 | 0 / 0.000 | 0.527634 | -1.413989 | 1.748578 | 0.191608 |
| Improved / Bladder | 5,207 | 0.841052 / 0.938536 | 4,765 / 91.511 | 0.022547 | +2.314987 | 0.227597 | 0.634093 |
| Improved / Rectum | 6,149 | 0.018903 / 0.018230 | 0 / 0.000 | 0.797636 | -3.868204 | 4.074712 | 0.024979 |
| Improved / Prostate | 17,149 | 0.081817 / 0.070540 | 0 / 0.000 | 0.604466 | -2.596051 | 3.008258 | 0.131232 |

**S035_Week0**

| Model / class | GT voxels | Probability mean / median | TP / recall (%) | Mean gap | Mean logit margin | Mean NLL | Soft Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard / Bones | 22,864 | 0.150987 / 0.153083 | 0 / 0.000 | 0.639791 | -1.733658 | 1.972963 | 0.134062 |
| Standard / Bladder | 9,266 | 0.309678 / 0.245834 | 2,467 / 26.624 | 0.386872 | -1.007702 | 1.573136 | 0.310668 |
| Standard / Rectum | 11,924 | 0.013820 / 0.011712 | 0 / 0.000 | 0.821609 | -4.216272 | 4.400988 | 0.022016 |
| Standard / Prostate | 4,448 | 0.028092 / 0.026329 | 0 / 0.000 | 0.719003 | -3.459623 | 3.757857 | 0.028334 |
| Improved / Bones | 22,864 | 0.192282 / 0.190068 | 0 / 0.000 | 0.550374 | -1.474868 | 1.787200 | 0.168172 |
| Improved / Bladder | 9,266 | 0.795039 / 0.904833 | 8,194 / 88.431 | 0.045558 | +1.865624 | 0.333249 | 0.499921 |
| Improved / Rectum | 11,924 | 0.025023 / 0.019403 | 0 / 0.000 | 0.787885 | -3.773427 | 3.991099 | 0.039729 |
| Improved / Prostate | 4,448 | 0.077205 / 0.065924 | 0 / 0.000 | 0.554662 | -2.444359 | 2.930013 | 0.070924 |

**B040_Week0**

| Model / class | GT voxels | Probability mean / median | TP / recall (%) | Mean gap | Mean logit margin | Mean NLL | Soft Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard / Bones | 29,715 | 0.135745 / 0.128309 | 0 / 0.000 | 0.651965 | -1.844757 | 2.088462 | 0.148320 |
| Standard / Bladder | 50,902 | 0.581742 / 0.632619 | 33,977 / 66.750 | 0.109601 | +0.493299 | 0.696795 | 0.714702 |
| Standard / Rectum | 17,493 | 0.013411 / 0.011247 | 0 / 0.000 | 0.859137 | -4.236440 | 4.374562 | 0.022263 |
| Standard / Prostate | 7,105 | 0.033480 / 0.031156 | 0 / 0.000 | 0.662813 | -3.165748 | 3.533706 | 0.042422 |
| Improved / Bones | 29,715 | 0.180112 / 0.155292 | 0 / 0.000 | 0.523448 | -1.529738 | 1.913286 | 0.192711 |
| Improved / Bladder | 50,902 | 0.851994 / 0.960689 | 45,714 / 89.808 | 0.056470 | +2.657462 | 0.287651 | 0.872922 |
| Improved / Rectum | 17,493 | 0.028011 / 0.024731 | 0 / 0.000 | 0.726842 | -3.397856 | 3.695901 | 0.045056 |
| Improved / Prostate | 7,105 | 0.150163 / 0.165016 | 0 / 0.000 | 0.334081 | -1.251270 | 2.000960 | 0.173531 |

#### Rank distributions

Counts list ranks **1,2,3,4,5,6**; ties favour the lower channel ID as in argmax. Percentages use GT-class denominators. No top-probability ties are recorded.

| Case / model / class | Counts by rank 1..6 | Rank 1 / rank 2 / below rank 2 (%) |
| --- | --- | --- |
| K019_Week1 / Standard / Bones | [0,21630,1035,0,0,0] | 0.000 / 95.433 / 4.567 |
| K019_Week1 / Standard / Bladder | [3504,1141,373,77,40,72] | 67.294 / 21.913 / 10.793 |
| K019_Week1 / Standard / Rectum | [0,0,1241,4303,598,7] | 0.000 / 0.000 / 100.000 |
| K019_Week1 / Standard / Prostate | [0,0,7129,9930,90,0] | 0.000 / 0.000 / 100.000 |
| K019_Week1 / Improved / Bones | [0,21125,1106,426,8,0] | 0.000 / 93.205 / 6.795 |
| K019_Week1 / Improved / Bladder | [4765,404,18,11,9,0] | 91.511 / 7.759 / 0.730 |
| K019_Week1 / Improved / Rectum | [0,0,3618,2429,102,0] | 0.000 / 0.000 / 100.000 |
| K019_Week1 / Improved / Prostate | [0,8,10306,5703,774,358] | 0.000 / 0.047 / 99.953 |
| S035_Week0 / Standard / Bones | [0,22602,262,0,0,0] | 0.000 / 98.854 / 1.146 |
| S035_Week0 / Standard / Bladder | [2467,5310,1313,126,28,22] | 26.624 / 57.306 / 16.070 |
| S035_Week0 / Standard / Rectum | [0,0,1296,9330,1245,53] | 0.000 / 0.000 / 100.000 |
| S035_Week0 / Standard / Prostate | [0,0,1778,2220,450,0] | 0.000 / 0.000 / 100.000 |
| S035_Week0 / Improved / Bones | [0,21940,556,368,0,0] | 0.000 / 95.959 / 4.041 |
| S035_Week0 / Improved / Bladder | [8194,795,49,179,49,0] | 88.431 / 8.580 / 2.989 |
| S035_Week0 / Improved / Rectum | [0,0,5124,6061,737,2] | 0.000 / 0.000 / 100.000 |
| S035_Week0 / Improved / Prostate | [0,1,3922,462,40,23] | 0.000 / 0.022 / 99.978 |
| B040_Week0 / Standard / Bones | [0,27513,2202,0,0,0] | 0.000 / 92.590 / 7.410 |
| B040_Week0 / Standard / Bladder | [33977,14678,1659,340,167,81] | 66.750 / 28.836 / 4.414 |
| B040_Week0 / Standard / Rectum | [0,0,389,17008,96,0] | 0.000 / 0.000 / 100.000 |
| B040_Week0 / Standard / Prostate | [0,0,3623,601,2881,0] | 0.000 / 0.000 / 100.000 |
| B040_Week0 / Improved / Bones | [0,26148,2419,1012,136,0] | 0.000 / 87.996 / 12.004 |
| B040_Week0 / Improved / Bladder | [45714,4343,590,189,66,0] | 89.808 / 8.532 / 1.660 |
| B040_Week0 / Improved / Rectum | [0,0,6516,10846,131,0] | 0.000 / 0.000 / 100.000 |
| B040_Week0 / Improved / Prostate | [0,46,6745,240,59,15] | 0.000 / 0.647 / 99.353 |

#### Verified comparison and interpretations

- **Bones:** mean true-class probabilities increase in all pairs (0.171074->0.199654, 0.150987->0.192282, 0.135745->0.180112); NLL/Soft Dice improve, but TP/recall remain zero. Body wins **every true Bones voxel** in all six outputs. Bones ranks second in 87.996%-98.854% of its true voxels, but mean winning-probability gaps are 0.523448-0.651965: second rank is not a near-tie. Improved's second-rank share decreases in all three crops despite higher means. Improved K019 predicts one Bones voxel on true Body: **one FP, no TP**.
- **Rectum:** means remain low (0.013411-0.028011), never rank first/second, and Body wins every true Rectum voxel. Improved improves S035/B040 but **regresses K019**: mean 0.019271->0.018903, NLL 4.013479->4.074712, Soft Dice 0.025484->0.024979, gap 0.745867->0.797636. Better K019 ranks do not imply better absolute probability or recall.
- **Prostate:** Improved raises means 0.040148->0.081817 (K019), 0.028092->0.077205 (S035), 0.033480->0.150163 (B040), with better NLL/Soft Dice but still zero TP. Most rank third/fourth. Improved B040 median is 0.165016, mean gap 0.334081 and signed logit margin -1.251270. These are nonzero responses, not recovered contours or calibrated clinical confidence.
- **Bladder:** recall improves 67.294%->91.511%, 26.624%->88.431%, 66.750%->89.808%; probabilities/NLL/Soft Dice improve. **Precision decreases** 88.844%->79.337%, 66.604%->43.662%, 98.710%->93.140%. Improved S035 predicts 18,767 Bladder voxels versus 9,266 true voxels. This Train-crop improvement does not negate the earlier S022 Validation-volume regression.
- **Train versus Validation:** Bones/Rectum/Prostate have zero TP already on both Train crops, weakening a Validation-only explanation. Improved B040 Prostate probability is **higher** than both Train values. Two purposive Train crops and one Validation center crop cannot establish population generalisation. B040 single-window probabilities/recall/Soft Dice are not the overlapping full-volume scores and cannot replace the 38-volume pooled comparison.

Original full-Validation counts remain unchanged: Standard predicts no Bones/Rectum/Prostate; Improved Bones has two TP plus one FP, and Rectum/Prostate remain absent. The probe's K019 Bones FP is a separate Train-patch observation.

| Hypothesis | Assessment | Limit |
| --- | --- | --- |
| A. Extremely low true-class probabilities even on Train | **Partly supported:** Rectum is strongly suppressed; Standard Prostate is also low. A blanket claim for all omitted classes is contradicted by Bones scores/ranks and Improved Prostate responses | No arbitrary probability threshold or calibration claim |
| B. Nontrivial probabilities lose argmax | **Supported as a manifestation**, especially Bones; Body is the dominant competitor throughout | Does not explain why these relative probabilities arose or justify changing argmax |
| C. Better Train recognition than selected Validation | **Not supported for omitted classes:** zero Train TP; Improved Prostate confidence higher on B040 | Does not rule out broader generalisation/context effects |
| D. Insufficient evidence to identify the underlying mechanism | **Supported for causation:** objective-gradient response, optimisation dynamics and representations remain unmeasured | Six forwards distinguish manifestations, not causes |

#### Reconciliation with training exposure

Independent tally of the 1,430 reconstruction records confirms:

| Class | Positive patches / rate | Pooled sampled voxel share |
| --- | ---: | ---: |
| Bones | 1,230 / 86.014% | 9.099350% |
| Bladder | 869 / 60.769% | 4.032347% |
| Rectum | 777 / 54.336% | 1.232043% |
| Prostate | 880 / 61.538% | 1.434959% |

Shares use **374,865,920 patch voxels** at the already documented common physical spacing; they are not raw full-volume fractions or organ-size estimates. Bones has more sampled presence/voxels than Bladder and substantial second-ranked probability, so absence or tiny fragments alone is insufficient. Small Rectum/Prostate shares remain plausible contributors, but do not measure effective gradients or explain Bones on their own. NLL/Soft Dice quantify current errors, not parameter-gradient strength; declining historical loss does not establish recovered class segmentation.

### Recommended next measurement

**One primary recommendation: a bounded, read-only Train-only CE/Dice parameter-gradient attribution audit, requiring separate approval. No immediate retraining or loss/sampling change.** This measurement is **not implemented or executed** here. Use the same two fixed Train crops and both authenticated E10 checkpoints, with no Validation/Test backward pass, optimiser step, checkpoint write or input-policy change.

**Hypothesis:** omitted classes receive weaker corrective parameter gradients, or gradients opposed by dominant-class/objective components, despite exposure. **Isolated variable:** the existing class/loss/head component being differentiated on identical input; no model parameter or coefficient changes. Preserve eval state for controlled local sensitivity, not replay of a historical training step. Standard uses 0.5 CE + 0.5 foreground Dice; Improved must also include its actual main/half/quarter losses weighted 1/0.5/0.25 and divided by 1.75, with approved logit upsampling.

**Measurement requirements:** obtain per-class CE contributions averaged over all voxels (zero for absent GT) and each weighted `(1-Dice_c)/5` term, plus head-wise contributions. Use read-only `autograd.grad` on `model.head` and the final decoder block shared by all classes. Report L2 norms/RMS, head-channel norms, cosine similarity with Body/Bladder and between CE/Dice, and the complete objective gradient. Compare Bones/Rectum/Prostate with **Bladder on the identical crop/model**, and unweighted versus actual weighted contributions without changing the objective. Verify component gradients sum to the unchanged objective within float32 tolerances; require finiteness/provenance/zero updates. Undefined cosine for a zero vector must remain not evaluable. Export aggregates only. E10 sensitivity is not the historical optimisation trajectory.

**Baseline and Validation metrics:** retain unchanged E10 checkpoints and existing 38-volume, stride-32, mean-softmax `overall_pooled` Dice/IoU, TP/FP/FN and predicted-class coverage as the performance baseline. This read-only measurement yields **no new Validation performance**. Any later separately approved intervention must use those same metrics, including Body/Bladder regressions and prostate/rectum recall; never select it using Test or falling training loss alone.

**Expected interpretation:** small/opposed parameter gradients would support an objective-response explanation worth a later isolated intervention; substantial aligned corrective gradients would weaken that explanation and leave history/representation/context unresolved. Neither is causal proof. Do not infer weak gradients from probabilities, choose class weights yet or launch longer training. This is the **single active recommendation**; older alternatives below remain deferred.


## G. Updated hypothesis ranking

Ranking is investigation priority, **not proven causality**. Exposure/probabilities distinguish observed suppression from failure confined to Validation.

| Priority / hypothesis | Measured support | Counterevidence / limits | Next discriminator |
| --- | --- | --- | --- |
| 1. Unequal effective objective/parameter-gradient response or conflict | High omitted-class NLL, low Soft Dice and Body competition; zero Train TP despite substantial Bones exposure | Class frequency/probabilities do not measure parameter gradients; no demonstrated loss/channel defect | The single proposed Train-only component attribution above |
| 2. Limited/uneven optimisation or insufficient features under the fixed budget | Failure already on selected Train crops after 1,430 updates; higher probabilities do not cross argmax | E10 responses do not reconstruct historical gradient dynamics; total loss is not per-class convergence | Interpret attribution before any intervention; no automatic longer-training run |
| 3. Residual imbalance and target/context variability | Small Rectum/Prostate shares, strong Body prior, patient/branch variability | Bones has 86.01% patch presence / 9.10% share, exceeding Bladder; missing examples alone is contradicted | Exposure plus component attribution; no new sampler now |
| 4. Generalisation or patch-to-volume aggregation/context effects | Patches and volumes differ in context/target priors | Both Train crops already fail omitted classes; Improved B040 Prostate confidence exceeds Train | Not primary; broader/parity measurement needs a separate evidence trigger |
| 5. Shared semantic/geometry defect or historical input mismatch | Shared pipeline; historical image bytes/clinical annotations not certified | Synthetic/source checks, fixed replay counts/hashes, affine checks and protected preflight pass | Investigate concrete inconsistency only; preserve provenance limits |

**Observed manifestation:** Bones loses substantial second-ranked probability to Body; Rectum is strongly suppressed; Prostate has improved but insufficient scores. No weak-gradient mechanism, channel bug, isolated residual/deep-supervision effect or clinical annotation error has been established.

## H. Historical unexecuted proposals - deferred

**No retraining proposal is active.** The earlier sampling-only and CE-weighting designs below are retained as historical, unexecuted proposals; they are not this audit's next-step recommendation. Frequent Bones exposure invalidates a simple absence trigger. Probability findings are now reviewed, but objective-gradient response remains unresolved; neither historical design is selected or authorised. No weights, revised sampler or longer-training run are proposed here.

<details>
<summary>Earlier conditional designs, preserved for history only</summary>

| Item | Experiment 1: sampling only | Experiment 2: CE weighting only |
| --- | --- | --- |
| Trigger / hypothesis | Replay shows classes 2/4/5 frequently absent or represented as tiny fragments; deliberate coverage may improve learning | Adequate presence but very unequal CE/gradient contribution; fixed training-derived class weighting may improve rare-class learning |
| Single changed factor | Replace the sampling allocation rule with a frozen, reproducible multi-class-aware policy, maintaining one patch per volume per epoch | Change only CE class weights; keep current sampler, Dice formula and Improved deep-supervision weights unchanged |
| Fixed controls | Same patient split, six labels, preprocessing, 64³ patches, architecture, AdamW/lr, batch size, seed and float32 policy | Same controls; weights fitted from **training exposure only**, chosen/recorded before the run, never fitted on Validation/Test |
| Budget | 10 epochs / 1,430 updates per arm, with measured wall/GPU cost; no claim that more patches are free | Identical update budget and cost reporting; do not simultaneously change sampling or epoch count |
| Run identity / output | Proposed new identity `sampling_only_v1`; fresh seed-3710 initialisation, external new run directory/checkpoint | Proposed `ce_weight_only_v1`; fresh initialisation, separate external directory/checkpoint |
| Validation evidence | Same 38-volume evaluator/settings: pooled and patient-level Dice/IoU, per-class TP/FP/FN and predicted coverage, especially 2/4/5; compare both architectures if resources allow | Identical metrics/protocol, including Body/Bladder regressions and prostate recall/false positives |
| Decision rule | Promote only if omitted-class true overlap/coverage improves against the matched control; scattered false-positive voxels alone fail | Same; no promotion based on falling training loss alone |
| Trade-off / gate | Record changes in Body/Bladder false positives and overlap. Agree the numerical tolerated degradation **before** running; no retrospective threshold | Reweighting may harm frequent-class performance or increase small-class false positives; use the same predeclared degradation bound |
| Source/checkpoint implications | New reviewed adapter/config and hashes; no resume from old checkpoints under changed policy, no weakened safeguards | New objective/config identity and hashes; old E10 checkpoints and results remain an unchanged control |
| Deadline value | Worth one bounded comparison only if measured exposure supports the factor; reject broad sampling searches | Worth testing only if component measurements support weighting; do not invent weights from anatomy sizes |

These designs are not run-ready approvals: allocation of classes, actual weight formula, external output paths and numeric regression bounds require a measured diagnosis and student decision. Reusing existing control results requires authenticated source/setup equivalence; otherwise a fresh matched control must be budgeted. If time allows only one experiment, choose the factor supported by the measurements. Do not launch an unmeasured longer-training intervention merely because Epoch 10 omits classes.

</details>

## I. Decision and next steps

**GO:** exposure and job 650924 probability authentication/recomputation are complete; separately review the one Train-only component-gradient measurement above. **NO-GO:** retraining, revised loss/sampler/architecture, additional inference or Test evaluation in this batch. No gradient probe is authorised or executed here; missing early epoch logs remain a separate historical-evidence limitation.

### Historical audit CLI examples

The commands below were proposed in the original diagnosis batch. The student has now executed the full reconstruction as job 650736, but its exact submission/terminal command was not supplied in this evidence update, so these examples are **not claimed as its verbatim executed transcript**. They are retained for history and are not instructions to repeat the completed audit. Codex did not run any remote command. They assume the two new diagnosis files have been placed beside the original protected production source in the established external Rangpur workspace. Do not overwrite production source to match this local checkout. Run the data audit in an appropriate authorised CPU compute session; no Slurm submission is automated here. The script writes stdout/stderr only; any redirection must use an existing personal directory outside Git and the dataset, without overwriting original logs.

```bash
cd "$HOME/comp3710_hipmri_verify"

# Bounded preliminary reconstruction: two TRAIN cases, 20 target patches.
CUDA_VISIBLE_DEVICES='' "$HOME/miniconda3/envs/torch/bin/python" -B -m \
  recognition.hipmri_3d_improved_unet_hard.diagnostics.class_collapse patch-exposure \
  --checkpoint "$HOME/hipmri_training_runs/standard_unet_epoch10.pt" \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open \
  --cases K019_Week1 S035_Week0

# Separate explicit full TRAIN reconstruction: 143 volumes, 1,430 patches.
CUDA_VISIBLE_DEVICES='' "$HOME/miniconda3/envs/torch/bin/python" -B -m \
  recognition.hipmri_3d_improved_unet_hard.diagnostics.class_collapse patch-exposure \
  --checkpoint "$HOME/hipmri_training_runs/standard_unet_epoch10.pt" \
  --dataset-root /home/groups/comp3710/HipMRI_Study_open --full-train
```

Expected bounded result: `status=PASS`, `scope=selected training subset`, two volumes / 20 patches, `full_train_completed=false`. Expected full result: 143 / 1,430 / 26, `full_train_completed=true`, valid six-class counts and exit 0. PASS establishes completion of **this reconstruction**, not recovery of original logged crops or clinical quality. Any provenance/geometry/count failure must stop and be reported, not bypassed. `--max-cases N` is an alternative deterministic bound retaining original indices; it is not a new split.

Repeat the same authorised command with `improved_unet_epoch10.pt` only if a second checkpoint provenance check is required; compare replay counts/hashes rather than assuming equality. Both checkpoint paths above are supported by the existing case-JSON metadata, not newly inspected binaries.

For already supplied original logs, the implemented history CLI is:

```bash
CUDA_VISIBLE_DEVICES='' "$HOME/miniconda3/envs/torch/bin/python" -B -m \
  recognition.hipmri_3d_improved_unet_hard.diagnostics.class_collapse loss-history \
  --logs "$STANDARD_INITIAL_LOG" "$STANDARD_RESUMED_LOG"
```

`STANDARD_INITIAL_LOG` / `STANDARD_RESUMED_LOG` are placeholders requiring the student's **actual** original paths, not claimed files or executed commands. Supply all nonduplicated segments as needed; process Improved logs separately. Expected complete result: ten authentic epoch means, component consistency and `status=PASS`. Missing epochs return `INCOMPLETE` / nonzero; bad records raise a clear error. This checks recorded history, not independent execution. Both full-validation JSONL files have now been supplied and independently audited locally; this utility does not parse them as training-loss logs. The existing history parser was called locally on each available E10 training file separately and correctly reports the missing earlier epochs; the placeholder command above remains a proposed multi-segment Rangpur invocation, not an executed remote command.

With real NiBabel available, the proposed CPU test command is:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest \
  recognition.hipmri_3d_improved_unet_hard.diagnostics.test_class_collapse \
  recognition.hipmri_3d_improved_unet_hard.test_patch_dataset -v
```

This exact command is proposed for Rangpur, not claimed executed there. The actual local run used a standard-input Python runner to add an import-only NiBabel shim, set two CPU threads and load those same two suites. `CUDA_VISIBLE_DEVICES=''` hides GPUs; `-B` prevents new bytecode; `-m unittest` resolves package tests; `-v` names individual outcomes.

### Completion boundaries

Full Train configuration-replay exposure is now authenticated and independently recomputed from the original job 650736 JSONL. At that historical exposure-integration stage, **missing early epoch components**, checkpoint probabilities and gradient attribution remained pending. The real probability log is now audited above; missing epochs and gradients remain unresolved. The original diagnosis batch supplied tested measurement methods and preserved all pre-existing files against its starting snapshot; its 48-test result above remains historical synthetic evidence. The preceding E10 log audit added a numeric loss plot. This subsequent patch-evidence integration modifies documentation only: original reconstruction counts, patient/epoch/branch distributions and summary arithmetic are directly inspected locally, without decoding MRI or running checkpoints. Production/diagnostic Python, configurations, original logs and checkpoints are unchanged. No MRI decoding, remote connection/job, inference/training, Test access or commit/push occurred. The student executed the original Rangpur experiments and full Train reconstruction and downloaded the logs; Codex performed local content/arithmetic audits and documentation updates. These roles do not establish a completed causal diagnosis.

**Subsequent Phase 4 preparation:** Codex added only the two isolated diagnostic files linked in Section F and this implementation/execution documentation. The 34 CPU tests include synthetic model forwards and mocked orchestration; they do not execute real checkpoint inference. At preparation, real reconstruction and forwards were pending. The student subsequently executed job 650924; its preflight, all three reconstructed patches and six forwards are now authenticated from the downloaded log. Binaries have not been opened locally. Production code/configuration and prior diagnostic files remain byte-identical to this batch's starting snapshot; pending Git changes are preserved. No remote command, training/backward/optimizer work, Test-array access or commit/push was performed.

**Current real-probability evidence analysis:** the student executed and downloaded job 650924. Codex read the six original external logs, reviewed the existing diagnostic/statistical contracts, independently checked aggregate arithmetic/provenance, and updated documentation only. No new MRI decoding, inference, backward pass, optimiser work, remote access or Test processing occurred. Raw probability quantiles and remote checkpoint-byte equality remain reported execution evidence, not independently replayed measurements. Existing source/configuration/checkpoints and pending Git changes are preserved. Only the separately authorised Train-only gradient-attribution measurement is recommended.
