# HipMRI Report-to-Code Traceability

## A. Purpose and scope

This internal reference connects the [Final Report](FINAL-REPORT.md), [task README](hipmri_3d_improved_unet_hard/README.md), implementation, JSON policies, tests and available experiment records. It is **not an additional teacher-mandated submission**. Source line numbers below refer to the local audit snapshot and should be refreshed if code changes.

The local report initially contained only Dataset and Preprocessing. Sections 2–4 were added from inspected code and student-supplied experiment records; unavailable original drafts were not reconstructed as though they had been inspected. Section 5 now integrates authentic full-Validation records and four paired case studies; numerical provenance and external artefacts are recorded in the [failure-case manifest](FAILURE_CASE_EVIDENCE.md).

### Authority and evidence levels

- **Verified from implementation:** a method/constant is present in inspected source; this alone does not prove a remote run used that source.
- **Verified from recorded experimental evidence:** identify whether original log/JSON bytes were inspected and recomputed locally, or only a student transcript/documentation was available. Both differ from independent experiment execution.
- **Partially verified:** source supports the mechanism but source logs, invocation settings or authority are missing.
- **Unverified / missing evidence:** no authentic result or authoritative clarification is available.
- **Contradicted and corrected:** documentation did not match current source or evidence and has been corrected.

The course document was found one directory above Git: [COMP3710_Report_2026_draft_v2.2.pdf](../../COMP3710_Report_2026_draft_v2.2.pdf). Its cover internally states **Version 2.0 Final Assessment Specification (Draft)**. Relevant inspected sections are 2.2 (printed pp. 5–6), 3 and 4.1 (pp. 11–13), 5.3 (p. 14), 6.1 (p. 15), 8 (p. 16), and Appendix A (p. 17). Text was read from the local PDF content streams because installed PDF extraction libraries/tools were unavailable; this was a content audit, not a PDF layout review. The filename/title discrepancy and whether a later official clarification exists remain unresolved. The parent-directory PDF link is for local use and is not included in the Git submission.

The [weekly continuity instructions](../../COMP3710_Project_Instructions_Weekly_Continuity.md) reinforce separation of observation, hypothesis and unknown evidence. They are workflow guidance, not an assessment rubric. At the earlier audit, no local E10 checkpoint or original full-validation JSONL/log was found in the repository, searched parent course evidence or relevant Downloads filenames. That statement describes the earlier documentation audit. The student subsequently downloaded four original E10 JSONL files to `C:\Users\ruiji\Downloads\hipmri_e10_evidence`; all four are now read and authenticated locally. [The manifest](FAILURE_CASE_EVIDENCE.md#original-e10-log-authentication) records filenames, byte counts, SHA-256, event counts, resumed epoch scope, exact pooled metrics and prediction coverage. This directly verifies recorded contents/configuration/arithmetic, not a rerun or independent certification of Rangpur execution. Checkpoint binaries and B040 prediction-driver JSONs remain unavailable; B040 numerical counts are now also verified from full-validation records.

## B. Report-to-Code Mapping

### Section 1 — Dataset and Preprocessing

| Report Section | Claim / Method | Source File | Function / Class / Configuration | Code Evidence | Verification Status |
| --- | --- | --- | --- | --- | --- |
| 1: dataset source | HipMRI root, patient/Week filename convention | [dataset.py](hipmri_3d_improved_unet_hard/dataset.py) | `DEFAULT_DATASET_ROOT`, `parse_volume_filename()`, `discover_pairs()` | Root at line 10; parser line 44; discovery line 106 pairs `semantic_MRs` and `semantic_labels_only` | Verified from implementation; course Appendix A independently describes 38 patients / 211 volumes |
| 1: labels | 0 Background, 1 Body, 2 Bones, 3 Bladder, 4 Rectum, 5 Prostate | [volume_io.py](hipmri_3d_improved_unet_hard/volume_io.py), [validate.py](hipmri_3d_improved_unet_hard/validate.py) | `ALLOWED_LABELS`, `CLASS_NAMES` | Loader line 19 allows IDs 0–5; metric naming maps channels to structures | Partially verified: mapping is student-provided CSIRO evidence; original CSIRO citation is unavailable locally |
| 1: frozen split | 26/6/6 patients, 143/38/30 volumes | [dataset.py](hipmri_3d_improved_unet_hard/dataset.py) | `APPROVED_PATIENTS`, `EXPECTED_*`, `validate_approved_split()` | Lists at line 14; counts lines 25–28; patient overlap validation line 58 | Verified from implementation and historical student Rangpur verification; remote files not newly recounted |
| 1: coverage | All Weeks stay with their patient; missing/duplicate/unexpected entries rejected | [dataset.py](hipmri_3d_improved_unet_hard/dataset.py), [test_dataset.py](hipmri_3d_improved_unet_hard/test_dataset.py) | `assign_approved_split()`, `discover_dataset()` | Lines 144 and 183; explicit K019 Week1 requirement and counts | Verified from implementation; tests cover positive and rejection paths, not a new real-data audit |
| 1: NIfTI grids | Finite arrays; matching shape/affine/spacing/orientation; conditional mm/unknown compatibility | [volume_io.py](hipmri_3d_improved_unet_hard/volume_io.py), [test_volume_io.py](hipmri_3d_improved_unet_hard/test_volume_io.py) | `load_volume_pair()` | Line 53; `get_fdata(float32)`, decoded-mask integer validation before uint8 conversion | Verified from implementation; 211-pair header results are historical student-supplied evidence |
| 1: spacing | `(1.6796875,1.6796875,1.55999755859375)` mm | [resampling_config.json](hipmri_3d_improved_unet_hard/resampling_config.json) | `target_spacing_mm`, `target_spacing_provenance` | Exact spacing and training-only patient-weighted provenance in JSON | Exact value verified; statistical derivation is recorded student audit evidence, not recomputed locally |
| 1: physical resampling | Shared shape/affine, preserved directions and voxel-edge field of view | [resampling.py](hipmri_3d_improved_unet_hard/resampling.py) | `make_target_grid()`, `resample_volume_pair()` | Lines 94 and 134; target-axis geometry and affine pull mapping | Verified from implementation; not necessarily downsampling for every volume |
| 1: interpolation | MRI order 1/nearest; mask order 0/grid-constant/cval 0 | [resampling.py](hipmri_3d_improved_unet_hard/resampling.py), [test_resampling.py](hipmri_3d_improved_unet_hard/test_resampling.py) | `_boundary_policy()`, nested `transform()` | Policy line 22; `affine_transform`, `prefilter=False`, finite/dtype/label checks | Verified from implementation; targeted boundary repair preserved in historical README |
| 1: label integrity | IDs 0–5; vanished foreground classes cause failure | [resampling.py](hipmri_3d_improved_unet_hard/resampling.py) | `_counts()`, `resample_volume_pair()` | Counts line 125; compares input/output counts and physical volumes | Verified from implementation; voxel-count preservation is not anatomical accuracy |
| 1: normalisation | Per-volume MRI-only nonzero clipping, float64 statistics, ddof 0 and zero preservation | [intensity_normalization.py](hipmri_3d_improved_unet_hard/intensity_normalization.py), [normalization_config.json](hipmri_3d_improved_unet_hard/normalization_config.json) | `normalize_volume_pair()`, `NormalizationStatistics` | Lines 79 and 40; selection uses the resampled MRI before normalisation; labels never select statistics | Verified from implementation; clarified the meaning of original/nonzero |
| 1: patch tensors | XYZ → ZYX; MRI `[1,D,H,W]`, mask `[D,H,W]` | [patch_dataset.py](hipmri_3d_improved_unet_hard/patch_dataset.py) | `HipMRIPatchDataset.__getitem__()` | Line 97; transposes lines 120–121 and identical crop slices | Verified from implementation and historical real patch records |
| 1: sampling | One 64³ patch/volume; 0.5 selects prostate branch, not exact positive-patch fraction | [patch_dataset.py](hipmri_3d_improved_unet_hard/patch_dataset.py), [patch_dataset_config.json](hipmri_3d_improved_unet_hard/patch_dataset_config.json) | `set_epoch()`, `SeedSequence((seed,epoch,index))` | Epoch setter line 92; sampling lines 133–148; fallback if class 5 absent | Verified from implementation; probability interpretation corrected |
| 1: augmentation/cache | No augmentation or integrated full-volume cache | [patch_dataset.py](hipmri_3d_improved_unet_hard/patch_dataset.py) | `__getitem__()` | Production preprocessing repeats for each request; no augmentation path | Verified from implementation; benchmark helper does not change Dataset behaviour |
| 1: test isolation | Train/validation arrays only in implemented entry points | [train.py](hipmri_3d_improved_unet_hard/train.py), [train_improved.py](hipmri_3d_improved_unet_hard/train_improved.py), [validate.py](hipmri_3d_improved_unet_hard/validate.py) | `main()`, `select_validation()` | Training explicitly selects `train`; validation selection line 212 requires six patients/38 volumes | Verified from implementation; untouched-test history is student-reported, not a remote access audit |

### Section 2 — Model Architecture

| Report Section | Claim / Method | Source File | Function / Class / Configuration | Code Evidence | Verification Status |
| --- | --- | --- | --- | --- | --- |
| 2: baseline | Four downsamplings, double 3D convolutions, GroupNorm(1), ReLU | [modules.py](hipmri_3d_improved_unet_hard/modules.py) | `_DoubleConv3D`, `StandardUNet3D` | Lines 9 and 23; four encoder blocks plus bottleneck and MaxPool3d | Verified from implementation |
| 2: decoder | Learnable transpose convolutions and concatenated skips, six raw logits | [modules.py](hipmri_3d_improved_unet_hard/modules.py), [test_standard_unet.py](hipmri_3d_improved_unet_hard/test_standard_unet.py) | `StandardUNet3D.forward()` | `upconvs`, `decoder`, `torch.cat`, `head`; no softmax in forward | Verified from implementation and earlier CPU tests |
| 2: residual units | Residual encoder, bottleneck, decoder; identity/channel projection | [improved_modules.py](hipmri_3d_improved_unet_hard/improved_modules.py) | `ResidualBlock3D`, `ImprovedUNet3D` | Lines 18 and 36; branch + shortcut then ReLU | Verified from implementation; not an exact paper reproduction claim |
| 2: deep supervision | Half/quarter decoder heads; optional tuple output | [improved_modules.py](hipmri_3d_improved_unet_hard/improved_modules.py), [test_improved_unet.py](hipmri_3d_improved_unet_hard/test_improved_unet.py) | `ImprovedUNet3D.forward()` | Decoder indices 2/1 return 32³/16³ logits for 64³ input; default main only | Verified from implementation and earlier local synthetic tests |
| 2: model sizes | Standard 5,646,470; Improved 5,734,114 trainable scalars | [modules.py](hipmri_3d_improved_unet_hard/modules.py), [improved_modules.py](hipmri_3d_improved_unet_hard/improved_modules.py) | Default constructors `(1,6,16)` | CPU construction and `sum(p.numel() for p in model.parameters() if p.requires_grad)` during this audit | Verified locally; no forward/training experiment required |

### Section 3 — Experimental Setup

| Report Section | Claim / Method | Source File | Function / Class / Configuration | Code Evidence | Verification Status |
| --- | --- | --- | --- | --- | --- |
| 3: optimiser/batch | AdamW .001, weight decay .00001, batch 1, 143 steps | [train.py](hipmri_3d_improved_unet_hard/train.py), [train_improved.py](hipmri_3d_improved_unet_hard/train_improved.py) | `HYPERPARAMETERS`, `main()` | Main lines 215 and 101; fixed policy guards, DataLoader and AdamW construction | Verified from implementation and matching saved configurations in both original E10 training logs |
| 3: base loss | .5 CE + .5 mean foreground Soft Dice, smoothing 1e-5 | [train.py](hipmri_3d_improved_unet_hard/train.py), [test_training.py](hipmri_3d_improved_unet_hard/test_training.py) | `combined_loss()` | Line 46; CE reduction none then mean, labels 1–5 all included | Verified from implementation; target-absent foreground terms are not excluded |
| 3: deep supervision | Weighted loss/CE/Dice components, weights 1/.5/.25 divided by 1.75 | [train_improved.py](hipmri_3d_improved_unet_hard/train_improved.py) | `combined_loss()` | Line 29; calls baseline objective three times after logit interpolation | Verified from implementation; auxiliary targets are not resized |
| 3: epoch execution | forward/loss/backward/update; partial diagnostic distinguished | [train.py](hipmri_3d_improved_unet_hard/train.py), [train_improved.py](hipmri_3d_improved_unet_hard/train_improved.py) | `run_epoch()` | Lines 92 and 54; actual step counts, finite gradients and completed flag | Verified from implementation and prior synthetic integration checks |
| 3: run budgets | 10 epochs / 1,430 updates each on A100 | Original training JSONL and downloaded prediction JSONs | Jobs 648790 / 649668; setup/resume/epoch/result and checkpoint metadata | 10 × 143 = 1,430; case JSONs identify completed E10 checkpoints | Original training logs inspected: Standard E4–10 / Improved E2–10, final 10/1,430 counters and source hashes match; earlier segments/binaries absent |
| 3: precision/reproducibility | float32, no AMP, seed 3710, warn-only determinism | [train.py](hipmri_3d_improved_unet_hard/train.py) | `configure_reproducibility()` | Line 74; seeded generators, cuDNN/TF32 flags, CUBLAS default | Verified from implementation; CUDA bitwise determinism not guaranteed |
| 3: checkpoint/resume | Atomic epoch boundary; model/optimiser/RNG/config/source hashes | [train.py](hipmri_3d_improved_unet_hard/train.py) | `_rng_state()`, `atomic_checkpoint()`, `resume_checkpoint()` | Lines 139, 147 and 172; exact config equality and complete-epoch checks | Verified from implementation; no best-model selection or automatic validation |
| 3: Improved compatibility | Separate model identity, filename guard and source hashes | [train_improved.py](hipmri_3d_improved_unet_hard/train_improved.py), [validate_improved.py](hipmri_3d_improved_unet_hard/validate_improved.py) | `source_hashes()`, `improved_checkpoint_path()`, `load_checkpoint_model()` | Lines 24/44 and 17; safe weights-only loading, policies/architecture/hash checks | Verified from implementation; existing Standard checks unchanged |
| 3: full-volume inference | 64³ windows, stride 32 for paired cases, boundary coverage | [validate.py](hipmri_3d_improved_unet_hard/validate.py) | `window_starts()`, `sliding_window_predict()` | Lines 33 and 44; final edge starts, high-side padding, CPU accumulators | Source and both original full-validation setups verified: 64³ windows/stride 32, matching geometry and mean-softmax aggregation |
| 3: metric aggregation | Mean probabilities → argmax; Dice/IoU; null for both absent | [validate.py](hipmri_3d_improved_unet_hard/validate.py) | `confusion_matrix()`, `metrics_from_confusion()`, `macro_metrics()`, `summarize()` | Lines 107/122/140/151; ground-truth rows, prediction columns | Verified from implementation; same functions reused by Improved validation |

### Section 4 — Baseline Results and Failure Analysis

| Report Section | Claim / Method | Source File | Function / Class / Configuration | Code Evidence | Verification Status |
| --- | --- | --- | --- | --- | --- |
| 4: Standard E10 | Job 649669; 38 successes, zero failures; supplied Dice/IoU | [Final Report, Section 4](FINAL-REPORT.md#4-baseline-results-and-failure-analysis) | Student-supplied remote validation record | Original 649669 JSONL: 38 confusion matrices sum to the reported pooled values | Direct local content/arithmetic verification; inference not rerun |
| 4: aggregation label | Supplied E10 tables are `overall_pooled` | [validate.py](hipmri_3d_improved_unet_hard/validate.py) | `summarize()` | Line 151 pools confusion counts before overlap | Confirmed from original summary and independently pooled per-volume matrices; distinct macro fields also verified |
| 4: prediction coverage | No Bones/Rectum/Prostate predictions in any of 38 cases | Original E10 per-volume confusion matrices | `metrics_from_confusion()` exposes predicted counts | Column sums are predicted voxels; positive per-volume counts can be tallied across records | Direct local tally of positive column sums in all 38 matrices; source does not emit an automatic cases-predicted summary |
| 4: collapse interpretation | Partial class prediction collapse observed; cause not isolated | Supplied coverage + overlap records | Isolated loss/source checks, Train exposure and authenticated job 650924 probabilities; parameter-gradient diagnosis pending | Zero predicted voxels support observed class absence, not its origin | Partially verified interpretation; causal claims withheld |
| 4: boundary utility | No prostate voxels predicted; four paired autopsies now in Section 5 | [Final Report](FINAL-REPORT.md#53-paired-anatomical-failure-case-studies) | Case counts and original PNGs | Eight panels inspected; six local JSONs, B040 counts now also confirmed by original full-validation logs | Omission verified within evidence scope; no clinical base/apex designation or surface metric |
| 4: Improved E10 | Job 649912 CANCELLED; successful replacement 650393 | Student-provided full-validation record | 38 successful / zero failed, reported 8m04s | Full `overall_pooled` comparison in Section 5 | Recorded successful Validation; not independently rerun |

### Section 5 — Controlled Comparison and Paired Failure Evidence

| Report subsection | Claim / method | Code or exact evidence reference | Verification and limits |
| --- | --- | --- | --- |
| 5.1 | Same 38-volume split; E10 pooled DSC/IoU and prediction-class coverage | `validate.summarize()`, `metrics_from_confusion()`; student records 649669 / 650393; [full numerical provenance](FAILURE_CASE_EVIDENCE.md#full-validation-numerical-provenance) | Original full-run JSONL inspected and recomputed; matching 38 identities/settings, 2 Bones TP + 1 FP, positive DSC 3.943109603265999e-7 |
| 5.2 | Four distinct purposively selected Validation patients | `dataset.APPROVED_PATIENTS['validation']`; case JSON `setup`, `volumes`, `summary`; [manifest](FAILURE_CASE_EVIDENCE.md) | Three model pairs authenticated locally; B040 PNG model attribution uses filenames/supplied records; numerical metrics now have full-validation log support; not population sampling |
| 5.3, Figure 5.1 | B040 Z=51,62,73; Bladder gain, 7,105 Prostate voxels omitted | [B040 exact external PNG paths/hashes and missing JSON note](FAILURE_CASE_EVIDENCE.md#b040_week0); [original narrative](Failiure-analysis.md) | Both PNGs previously inspected; B040 counts/metrics now also verified from both original full-validation logs; driver JSON and PNG model attribution remain separate evidence |
| 5.3, Figure 5.2 | M030 Z=43,55,67; DSC 0.4279→0.8515 | `standard_M030_Week5/results.json`, `improved_M030_Week5/results.json` at [exact paths](FAILURE_CASE_EVIDENCE.md#m030_week5) | Model/case/checkpoint/geometry/Z/counts authenticated; original PNGs inspected |
| 5.3, Figure 5.3 | S022 Z=44,54,64; DSC regression 0.6243→0.2490 | `standard_S022_Week5/results.json`, `improved_S022_Week5/results.json` at [exact paths](FAILURE_CASE_EVIDENCE.md#s022_week5) | TP/GT/predicted counts reproduce DSC/IoU, precision and recall; genuine counterexample |
| 5.3, Figure 5.4 | K008 Z=56,70,84; 11,114 GT Prostate voxels, none predicted | `standard_K008_Week0/results.json`, `improved_K008_Week0/results.json` at [exact paths](FAILURE_CASE_EVIDENCE.md#k008_week0) | Largest in reported representative screening, not established dataset-wide maximum |
| 5.3–5.4 | Aligned displays and organ omission | `predict.aligned_slice_panels()`, `save_visualization()`; local PNGs; `volumes[].confusion_counts` | Same `[:,:,z].T`, error codes; source-axis display, not established anatomical base/apex; causes unproven |
| 5.5 | Counts, single-case inference time and CUDA peaks | Default models; six JSON `inference_seconds`, `peak_gpu_*_bytes` | Three paired single runs, no repeated matched full-run benchmark or training-cost inference |
| 5.6 | Research-only recommendation and outstanding requirements | Observed class omissions, case regression, [task assessment gaps](hipmri_3d_improved_unet_hard/README.md#assessment-documentation-gaps) | Test evaluation, clinical validity, check-off and submission unverified; figures remain external pending permission |

## C. Data Pipeline Trace

1. `discover_dataset(root)` indexes filenames, pairs patient/Week and validates all frozen partitions. It does not load array contents. Training selects only `train`; the existing validation entry points select only `validation`. The prediction driver defaults to Validation and can select Test only with explicit split/opt-in; no real Test evaluation is claimed.
2. `load_volume_pair(pair)` decodes finite float32 MRI and integer uint8 segmentation, validates shared grids and retains actual/inferred unit provenance. NiBabel intensity scaling is applied before validating mask IDs.
3. `resample_volume_pair(loaded, spacing, max_estimated_bytes=...)` generates one physical output grid, with linear MRI and nearest-neighbour mask interpolation. The 512 MiB limit is a four-times source-plus-output **planning guard**, not measured peak RSS.
4. `normalize_volume_pair(resampled)` computes per-volume statistics from pre-normalisation nonzero MRI values, clips and Z-scores in float64, then returns float32. Segmentation does not determine intensity statistics. Zero positions stay zero.
5. `HipMRIPatchDataset.__getitem__` permutes XYZ → ZYX and applies identical crops. Seed/epoch/index define stochastic foreground-aware or random sampling. A sample is MRI `[1,64,64,64]`, long mask `[64,64,64]`, patient, Week, DHW crop origin and sampling mode. DataLoader batch 1 adds a leading batch dimension.

No padding/augmentation is introduced during training. The validation Dataset's center crop is a limited patch interface; the standalone full-volume evaluators bypass it. The temporary K019 benchmark writes equivalent arrays for timing but is not integrated into training.

## D. Model Architecture Trace

Both default models use widths 16/32/64/128 at 64³/32³/16³/8³, with a 256-channel 4³ bottleneck. Four decoder stages restore 8³/16³/32³/64³ features, with channels 128/64/32/16. Encoder skips concatenate with upsampled features, doubling decoder-block input channels. Main heads produce six raw logits.

Standard blocks apply two Conv3d/GroupNorm/ReLU sequences. Improved blocks apply two volumetric convolutions with GroupNorm, add an identity or learned 1×1×1 shortcut, and apply ReLU. Improved decoder indices 2 and 1 feed the half- and quarter-resolution auxiliary heads, respectively; returned auxiliary order is half then quarter. Default Improved inference omits auxiliary predictions. Neither main forward applies softmax or argmax.

The course's architecture table supports the selected Normal/Hard bands. Having residual units and deep supervision does not prove paper reproduction, clinical validity or attainment of the test Dice target. A two-arm 3D comparison examines residual/deep-supervision changes jointly; it cannot isolate each change separately or estimate a 2D-versus-3D effect.

## E. Training and Loss Trace

`train.main` and `train_improved.main` construct the same training Dataset, seeded shuffled DataLoader and AdamW settings. Each calls `set_epoch(epoch)` before `run_epoch`. A full epoch has 143 steps; a diagnostic intentionally stops after one step without completing an epoch.

The baseline objective is:

```text
CE = mean over all voxels of unreduced six-class cross entropy
D_c = (2 sum(p_c t_c) + 1e-5) / (sum(p_c) + sum(t_c) + 1e-5), c=1..5
Lbase = 0.5 CE + 0.5 (1 - mean_c D_c)
Limproved = (Lbase(main) + 0.5 Lbase(upsampled half)
             + 0.25 Lbase(upsampled quarter)) / 1.75
```

Here `p_c` is softmax probability and `t_c=(target==c)` is a temporary binary indicator, summed over batch and spatial axes. Absent target classes still contribute. Trilinear interpolation changes **auxiliary logits**, with `align_corners=False`; masks remain class indices. Training Dice terms are soft scores, not the reported hard-label validation Dice.

`atomic_checkpoint` writes/fsyncs a temporary file and atomically replaces the last checkpoint after a completed epoch. The shared resume helper restores model, optimiser and RNG states and rejects changed configuration before loading state. Improved config adds model/deep-supervision identity and hashes the new trainer/model as well as shared production sources. Source-hash protection must remain intact; documentation-only edits do not change these Python hashes.

There is no integrated validation scheduler, best-checkpoint selection, augmentation, AMP or cache. Matching step budgets control exposure, not compute cost. The two logged total losses differ in head composition and should not be used as a direct ranking of segmentation quality.

Recorded epoch/step loss evidence is mapped to the [manifest table](FAILURE_CASE_EVIDENCE.md#recorded-training-loss-trajectory) and [Figure 3.1](FINAL-REPORT.md#recorded-training-loss-trajectory). Each original epoch mean matches its 143 recorded step values; Improved components include weighted auxiliary heads. Standard E1–3 and Improved E1 are absent from these resumed-job files. Local `summarize_loss_logs()` returns INCOMPLETE for those gaps without inventing values. Declining, non-monotonic objective values do not establish class-specific convergence.

## F. Evaluation and Metric Trace

`validate_improved` imports the same `_process_pair`, `select_validation` and `summarize` used by Standard evaluation; only checkpoint/model loading differs. `_process_pair` calls production preprocessing, sliding-window inference and confusion-count metrics. Its metadata distinguishes source `original_shape_xyz`, processed `resampled_shape_xyz` and `inference_input_shape_xyz`, preventing the earlier duplicate-key TypeError.

`sliding_window_predict` processes one 64³ patch at a time under `eval()`/`inference_mode()`, averages overlapping **softmax probabilities** uniformly on CPU, then argmaxes and converts back to uint8 XYZ. It preserves full spatial coverage including boundaries. Six float32 score arrays plus one count array use 28 bytes per working voxel, subject to a 512 MiB accumulator guard; this is not whole-process or GPU peak memory.

With confusion rows = truth and columns = prediction:

```text
TP_c = diagonal count
Ntrue_c = row sum; Npred_c = column sum
Dice_c = 2 TP_c / (Ntrue_c + Npred_c)
IoU_c = TP_c / (Ntrue_c + Npred_c - TP_c)
```

Both-absent classes produce null and are excluded from macro averaging. For each patient, counts are pooled across Weeks. `overall_pooled` pools every successful volume's voxel counts; `overall_volume_macro` averages evaluable volume metrics equally; `overall_patient_macro` averages evaluable patient metrics equally. More scans/larger volumes can affect pooled weighting. There is no implicit foreground-only overall Dice scalar or acceptance score in the evaluator.

The E10 prediction-coverage table is now directly recomputed from all 38 confusion matrices per model: column sums give predicted voxel counts and positive per-volume counts give cases predicted. Exact totals and the two Improved Bones case identifiers are retained in the manifest; they match the earlier student-supplied coverage. Zero coverage is failure evidence; nonzero coverage alone is not accuracy.

Fair comparison is supported by common split, preprocessing, sampling, optimiser policy, update count and evaluator. It remains limited by one recorded seed, different model initialisations/objectives, CUDA warn-only determinism and incomplete matched runtime/VRAM profiling. E10 table aggregation is verified as `overall_pooled` from original summaries and recomputation; both original full-run setups confirm stride 32 and common policies/geometry. Standard full validation names `standard_unet_last.pt`, paired predictions name `standard_unet_epoch10.pt`; full-validation binary hashes are absent, so weight equivalence across names is unverified. Final metrics are on the resampled grid, not source-grid predictions or clinical surface-distance measurements.

### Functional prediction entry point — local implementation evidence

The required [predict.py](hipmri_3d_improved_unet_hard/predict.py) is now functional. `parse_args()` requires model/checkpoint/output and explicit case or whole-split scope; `select_pairs()` validates frozen evaluation coverage and enforces Validation-by-default plus `--split test --allow-test` opt-in. It rejects training/cross-split cases before decoding. Discovery inspects all partitions' filenames but does not decode other splits.

`load_model()` delegates to `validate.load_checkpoint_model()` or `validate_improved.load_checkpoint_model()`. Existing weights-only, strict state, architecture, completed-epoch, policy and source-hash safeguards are unchanged. `predict.py` is not in their production hash lists. `prepare_volume()` calls the existing loader, resampler and normalizer sequentially. `process_pair()` uses the shared `sliding_window_predict()`, `confusion_matrix()` and `metrics_from_confusion()`; `main()` reuses `summarize()`. Split metadata reflects actual selection rather than the existing validation helper's hardcoded field. Both arms use 64³ windows/default stride 32 and identical aggregation/absent-class policies.

`aligned_slice_panels()` applies `[:,:,z].T` identically to MRI/truth/prediction and computes prostate FP/FN plus other-class errors. `save_visualization()` uses a common affine-derived XY aspect and fixed six-class colours. Slice choice spans ground-truth prostate Z extent, not an inferred anatomical diagnosis or preprocessing parameter. Outputs are external PNG panels and `results.json` containing setup, checkpoint identity/hash, per-case evidence/failures and unchanged metric aggregations; no source files or checkpoints are written. `output_path()` rejects existing directories, Git/source-workspace locations and source-dataset locations. CUDA execution requires a Slurm GPU allocation; CPU is explicit. GPU fields are null on CPU, and failure prevents a final PASS/full-coverage claim.

Directly observed local verification: **22 new [prediction tests](hipmri_3d_improved_unet_hard/test_prediction.py)** plus 70 existing selected regression tests passed: **92 tests in 10.643s, OK**, WSL PyTorch `2.12.1+cu130`, CUDA hidden. Genuine synthetic tests cover saved checkpoint loading/rejection for both models, real CPU sliding-window alignment/metrics, PNG/JSON generation, split/CLI safety and error reporting. NIfTI stages use mocks where required because local NiBabel is missing. This establishes software behaviour, not real MRI/GPU acceptance or Test performance. Final Report experimental values are unchanged; no remote execution is claimed. Proposed real commands and prerequisites are in the [task README prediction section](hipmri_3d_improved_unet_hard/README.md#prediction-driver).

**Subsequent real acceptance:** student records confirm Standard job **650442** and Improved job **650468**, both PASS for B040_Week0. Job **650490** completed 0:0 and generated six additional paired-case PNG/JSON outputs. All six JSONs were inspected locally: model/checkpoint E10 identity, Validation scope, PASS, shared grid/settings and arithmetic agree. Eight PNGs were visually inspected. This is successful real MRI/A100 driver evidence supplied by the student plus local artefact review; Codex did not execute those remote runs. No Test evaluation is established. Original images stay outside Git; figure captions/placeholders and exact paths are in the manifest.

## G. Experimental Evidence Register

| Experiment or Job ID | Model | Checkpoint / validation stage | Available result | Evidence location or provenance | Verification status | May be cited in report? |
| --- | --- | --- | --- | --- | --- | --- |
| M1-B4 Batch 1 | No model | Discovery and frozen split | 211 matched pairs, 38 patients; 26/143, 6/38, 6/30; no leakage | Historical [ai.md](hipmri_3d_improved_unet_hard/ai.md) and task README, student Rangpur records | Recorded verification; not newly run | Yes, with historical provenance |
| Batch 2.1 header audit | No model | All 211 pair headers | Zero grid mismatches; LPS/LPS, MRI mm / mask unknown | Historical task README / ai.md | Recorded header evidence, not complete arrays/annotation audit | Yes, within scope |
| 647039 | Standard | One real GPU smoke step | Loss 1.92833924; 64/64 finite-gradient tensors; 59/64 changed; peak allocated 441744384 bytes | Historical student-run A100 record in README / ai.md | Recorded execution; not E10/full training resource profile | Yes, only as a smoke milestone |
| 647115 | No model | Full preprocessing coverage | 181/181 Train/Validation PASS; whole-process RSS 345210880 bytes | Historical student CPU Slurm record | Recorded coverage evidence, no Test arrays | Yes, not a training-speed measurement |
| 648247 | No model | One-case K019 temporary cache benchmark | Student previously reported completed 0:0 and exact array/metadata equivalence | Project conversation; local benchmark source, no remote log | Recorded result; full-training speedup not measured | Only for limited cache-equivalence statement |
| Standard E3 | Standard | Completed 3 epochs / 429 updates, full validation | Recorded Background Dice 0.921977893998784, Body 0.8071822054161119; other four zero | Student conversation/current supplied audit request | Recorded results; exact aggregation/settings/log not local | Yes, provisional comparison with stated limits |
| 648790 | Standard | E10 training, 10 epochs / 1,430 updates | Final PASS; resume=3, seven completed epochs / 1,001 new steps | `standard_e10_648790.jsonl`; [authenticated log register](FAILURE_CASE_EVIDENCE.md#original-e10-log-authentication) | Original records/step means/current source hashes checked; E1–3 events and binary absent; Slurm COMPLETED is student-reported | Yes, with resumed scope |
| 649668 | Improved | E10 training, 10 epochs / 1,430 updates | Final PASS; resume=1, nine completed epochs / 1,287 new steps | `improved_e10_649668.jsonl`; authenticated log register | Original records/step means/current source hashes checked; E1 events and binary absent; Slurm COMPLETED is student-reported | Yes, with resumed scope |
| 649669 | Standard | E10 full Validation, 38 volumes | 38 successful, 0 failed; pooled metrics and exact coverage verified | `standard_val_e10_649669.jsonl`; [Section 5.1](FINAL-REPORT.md#51-controlled-validation-comparison) / manifest | Original setup/38 confusion matrices/summary inspected; E10 `standard_unet_last.pt` metadata, no binary hash | Yes, Validation only |
| 649912 | Improved | Original E10 Validation attempt | CANCELLED; replaced by 650393 | Student job history | Cancelled attempt, no successful result attributed to it | Only as troubleshooting history |
| 650393 | Improved | Replacement E10 full Validation | 38 successful, 0 failed; log loop 458.372391 s, student-reported Slurm 8m04s | `improved_val_fast_650393.jsonl`; manifest | Original setup/38 matrices/summary checked; `improved_unet_epoch10.pt`; Bones=2 TP/1 FP, negligible positive overlap | Yes, Validation only |
| 650442 | Standard | Real prediction driver, B040_Week0 E10 | PASS; original PNG exists | Student records and local B040 PNG | PNG visually inspected; JSON absent locally | Yes, supplied metrics with explicit provenance |
| 650468 | Improved | Real prediction driver, B040_Week0 E10 | PASS; original PNG exists | Student records and local B040 PNG | Same B040 evidence limits | Yes, no clinical/Test conclusion |
| 650490 | Both | Paired M030_Week5, S022_Week5, K008_Week0 predictions | COMPLETED 0:0; six PNGs and six PASS JSONs | [Exact local/remote paths and hashes](FAILURE_CASE_EVIDENCE.md) | JSON identity/geometry/metrics checked locally; PNGs inspected; remote execution student-supplied | Yes, paired case studies and measured single-run resources |
| Prediction CPU verification | Both | Driver and selected regressions | 92/92 in 10.643s, OK; 22 focused tests included | Prior directly observed local output / tests | Synthetic software evidence, not additional remote inference | Yes, with synthetic scope |
| Section 5 integration | Both | Documentation/evidence arithmetic | Six JSONs authenticated; per-class confusion arithmetic agrees; eight PNGs inspected | This local audit and manifest | No new training, MRI decoding or inference; figures external | Yes for evidence reconciliation |
| 650736 | No new model run | Standard E10-authenticated Train patch reconstruction | PASS, 143 volumes / 26 patients / 1,430 unique case-epoch patches; full Train=true | Original external `patch_full_650736.jsonl`; [authenticated counts and breakdowns](CLASS_COLLAPSE_DIAGNOSIS.md#original-full-train-audit-authentication) | Original JSONL parsed; 8,326 independent assertions plus production summary parity; Slurm COMPLETED 0:0 / 00:18:46 supplied by student; no independent proof of historical input bytes | Yes, replay exposure rather than new training or causal proof |
| 650924 | Both E10 snapshots | Three fixed patches / six main-head forwards | PASS; zero backward, optimiser updates and Test arrays | External `prob_probe_650924.jsonl`; [hash, provenance and full class statistics](CLASS_COLLAPSE_DIAGNOSIS.md#authenticated-real-probability-results) | 12 records; 1,271 aggregate/source assertions plus five finite/selection checks; no raw-probability recomputation. Slurm COMPLETED 0:0 / 25 s student-supplied | Yes for selected-patch suppression/competition, not causation or volume metrics |
| Original E10 log audit | Both | Local JSONL authentication/arithmetic, no inference | Four files parse; 5,491 structural/count/mean assertions; production metric parity; training-history auditor INCOMPLETE only for missing early epochs | [Log hashes/counts/metrics/loss table](FAILURE_CASE_EVIDENCE.md#original-e10-log-authentication) | Direct local content checks; numeric plot only; no production edits or remote run | Yes, evidence audit with missing segments explicit |
| Prior local CPU model/integration tests | Both | Synthetic architecture/training/checkpoint/inference | 17 model tests and later 49 combined integration/training/validation tests passed in prior Codex reports | Project conversation and [test_improved_integration.py](hipmri_3d_improved_unet_hard/test_improved_integration.py) | Prior local synthetic evidence, not rerun in this documentation batch | Yes for software checks, not clinical/GPU generalisation |
| This documentation audit | Both | Source inspection / CPU construction only | Frozen patient lists valid: 38; default trainable counts 5,646,470 / 5,734,114 | Actual local command outputs from this audit | Directly observed local checks; no array loading or training | Yes for code facts |

No original full JSONL logs, prediction arrays or checkpoints were added to Git. The later log audit adds only documentation and [a numeric recorded-loss plot](figures/e10_recorded_training_losses.png), not medical images. Local HEAD at inspection was `ed705bdc4ef72db282e9c53e599f150e34a303e3`; training/validation and Improved files were already pending changes. This HEAD alone does not establish the remote E10 source version; the newly inspected training configurations instead supply seven/nine source hashes matching current local files. They do not identify a remote Git commit or certify checkpoint binaries.

The Section 5 batch preserves all executable/configuration file bytes and external original evidence. It adds only documentation/metadata; local pending source changes predate this batch. No commit, push, remote job, Test evaluation or source-medical-data download occurred.

The Train exposure evidence maps `diagnostics.class_collapse.exposure_records()` / `summarize_exposure()` to [diagnosis sections C/D](CLASS_COLLAPSE_DIAGNOSIS.md#c-ground-truth-class-frequency-analysis): raw/resampled totals, physical voxel sizes, positive counts, pooled fractions and patient/epoch/branch breakdowns. The original JSONL remains external, with SHA-256 recorded in that report; count-based tiny-target thresholds are explicitly descriptive. Both original training logs match its ordered pairs/policies/versions; no separate Improved replay or historical source-image hashing is claimed. Diagnosis Section F now authenticates job 650924 and maps `diagnostics.checkpoint_probabilities.probability_statistics()` / `protected_load()` / `reconstruct_patch()` to conditional probabilities, ranks/margins, production main-head loss and protected provenance. Parameter gradients remain unmeasured.

## H. Known Limitations and Open Questions

1. **Evidence identity:** original 649669/650393 setup/38-volume/summary logs are now authenticated and pooled metrics/coverage recomputed. Full-validation checkpoint binaries/hashes are not local; Job 650924 records equal current Standard `last.pt`/`epoch10.pt` file SHA-256 and state tensors; historical Validation-time hash remains unavailable. E3 setup/aggregation and Standard E1–3 / Improved E1 training events remain missing.
2. **Improved result:** 649912 was CANCELLED; replacement 650393 original log verifies 38/38 and the comparison. Cancellation/scheduler history remains student-supplied; no remote job was queried by this audit.
3. **Class collapse:** Standard predicts no Bones/Rectum/Prostate; Improved predicts only three Bones voxels and no Rectum/Prostate. Original counts establish two TP and one FP; DSC/IoU are negligible positive values that round to zero. Job 650736 now measures configuration-replay exposure: Bones 1,230/1,430 (86.01%) / 9.10% voxels; Prostate 880/1,430 (61.54%) / 1.43%. Job 650924 records Bones second-ranked probabilities, low Rectum/Prostate scores and zero omitted-class TP on both Train crops and B040 center crop. Parameter-gradient strength/causes remain unproven; historical image bytes are not certified.
4. **Anatomical evidence:** four paired patient studies/eight PNGs now exist and were inspected, with matching Z levels. B040 driver JSONs are unavailable locally, but full-validation counts are now authenticated; figure permissions/insertion, clinically grounded base/apex designation, surface measures and consecutive-slice analysis remain unresolved.
5. **Resource evidence:** Section 5 reports three paired single-case inference/GPU measurements and parameter counts. Original logs now record validation loop/inference sums and seven/nine epoch runtimes/GPU peaks. Repeated matched profiling and complete ten-epoch costs remain incomplete; different segment lengths and smoke/CPU RSS are not substitutes.
6. **Authoritative labels:** code is consistent with the student-supplied CSIRO mapping, but the exact authoritative citation/table remains unavailable locally.
7. **Encoding and downsampling compliance:** specification Section 2.2 requests one-hot multi-channel encoding and downsampled 3D data; Dataset returns class indices and spacing can upsample some axes. Explain the mathematical encoding and obtain an authoritative clarification rather than claiming identical compliance.
8. **Submission structure:** Section 4.1 expects substantive `modules.py`, `dataset.py`, `train.py`, `predict.py`, README responsibilities. Helpers are allowed, but Improved model, Patch Dataset and validation are separate entry points. Functional prediction now has local tests and student-run real acceptance for both models; any remaining organisation clarification is separate. No NIfTI prediction export is implemented or required for the PNG demonstration.
9. **Documentation obligations:** feasibility review/check-off, algorithm diagram, authorised medical figure insertion and final Test metrics remain incomplete. Genuine partial loss curves now exist; missing early loss epochs must stay explicit. Both original training/validation logs confirm library/GPU fields, but not Python version. A supported comparison/research-only recommendation now exists; comprehensive task README/PDF, PR and final submission still require completion.
10. **Augmentation:** absence is verified, not uncertain. Section 4.1 mentions augmentation in dataset responsibilities; record the actual absence and clarify expectation. No augmentation was introduced to repair documentation.
11. **Specification authority:** local filename v2.2 versus internal Version 2.0 Draft needs reconciliation with current course announcements. No exact deadline, later clarification, tutor outcome or PR state was independently verified here.

## I. Final Submission Consistency Checklist

- [ ] Retain source hashes/configuration and exact checkpoint identities for the cited E3/E10 runs.
- [x] Authenticate original E10 logs externally; recompute `overall_pooled`, all per-volume/patient/macro summaries, matching 38 identities and inference settings. Keep original logs outside Git.
- [x] Record Improved E10 completion in 650393, cancelled predecessor 649912 and the authentic Section 5 comparison.
- [x] Retain exact confusion-derived prediction coverage and negligible positive Bones overlap; keep causal explanations labelled as hypotheses.
- [x] Authenticate 650736 full Train exposure and raw/resampled/patch denominators; distinguish replay from historical file-byte proof.
- [x] Authenticate/review job 650924 probabilities against exposure/original Validation logs; retain patch scope and unresolved gradients/causes.
- [ ] Add the original CSIRO label mapping citation and full bibliographic references for designs actually discussed.
- [ ] Document exact E10 Python/library/device versions from saved setup/checkpoint metadata.
- [ ] Resolve encoding, downsampled-data and entry-point responsibilities against the latest official clarification.
- [ ] Include the genuine one-page feasibility review and confirm tutor check-off; do not invent approval.
- [x] Document four paired failure autopsies, original external artefacts and the three-case measured resource table.
- [ ] Establish figure permissions, insert originals into the authorised submission, and complete the algorithm figure, missing early loss history and anatomical boundary analysis. Partial recorded-loss curves are available.
- [x] Complete a supported Validation comparison and research-only manager recommendation; clinical validity remains unestablished.
- [ ] Preserve the Test split until the authorised final evaluation and report those metrics distinctly from Validation.
- [ ] Make the task README the comprehensive submitted report; check its PDF export, PR branch and required review-response process.
- [x] Implement the protected prediction driver with CPU tests and student-recorded real MRI/A100 PASS for both models (650442/650468), plus locally inspected paired outputs (650490).
- [ ] Preserve detailed AI/human contributions and evidence levels; extend historical command records only with authentic transcripts.
- [ ] Refresh source-line references and validate all relative links before export/submission.
- [ ] Keep datasets, checkpoints and large experiment logs out of Git; inspect the complete intended diff before any later authorised commit.

## V4 E20 - Paired Prostate False-Positive Experiment (13 October 2026)

### Research question and controlled factor

Question: Does increasing the Prostate false-positive penalty improve
Prostate segmentation without unacceptable recall loss or regression
in other anatomical classes?

The selected V3 E12 checkpoint is the common parent.

- Control: original V3 loss, Prostate Tversky alpha=0.5.
- Treatment: Prostate Tversky alpha=0.7.
- Changed class: Prostate, index 5.
- Other model, data, sampling, optimisation and evaluation policies:
  intended to remain controlled and subject to execution audit.
- Additional training: epochs 13 through 20, 143 steps per epoch,
  1,144 updates per arm.
- Full-volume Validation: epochs 14, 16, 18 and 20.
- Primary paired comparison: epoch 20.

### Parent provenance

- Selected parent checkpoint SHA-256:
  `bb0d03c7a047ed187fadb5e1bf6ac1776448b351621764e9b6f0ff1fb67660d4`
- Selection receipt SHA-256:
  `fffe08a6c6502989e1ad033a0b78ea5117377dbad2de762791a3a895b2e189da`
- Train: 26 patients, 143 volumes.
- Validation: six patients, 38 volumes.
- Held-out Test: six patients, 30 volumes; excluded from V4 development.

### Report-to-code mapping

| Claim or control | Implementing source |
| --- | --- |
| Experimental schedule and declared comparison | `prostate_fp_e20_config.json` |
| Control and Treatment loss | `experiment_prostate_fp_e20.py` |
| Source and parent authentication | `experiment_prostate_fp_e20.py` |
| One-epoch job boundary | `experiment_prostate_fp_e20.py` |
| Step-level hyperparameter and crop logging | `experiment_prostate_fp_e20.py` |
| Checkpoint and resume integrity | `experiment_prostate_fp_e20.py` and inherited V3 helpers |
| Scheduled full-volume Validation | `experiment_prostate_fp_e20.py` and V3 validation helpers |
| Paired audit and decision rules | `experiment_prostate_fp_e20.py` |
| Saved CPU numerical regression | `test_prostate_fp_e20.py` |

### E20 source identities - pre-GPU verification snapshot

| File | SHA-256 |
| --- | --- |
| `experiment_prostate_fp_e20.py` | `e1457b447fcb3d9521bbad6969a52b0790b9d630fbbf53dedde8dd5528c26179` |
| `prostate_fp_e20_config.json` | `d9484eabb3705004b9b310f571e74c39b523a237a06e7555e37c567de38ab3f0` |
| `test_prostate_fp_e20.py` | `cea32d7003d7f0c7062c8638002503664f7cddf93d9a3fa7ed4787917f40b834` |

These hashes identify the files reviewed on 13 October 2026.
They are not evidence that GPU training has started or that a final
training checkpoint has been frozen.

### Verification evidence

| Verification scope | Observed outcome |
| --- | --- |
| CPU numerical loss and gradient tests | 3/3 PASS |
| Mocked resume-boundary tests | 7/7 PASS |
| Synthetic paired-record integrity tests | 8/8 PASS |
| Synthetic CPU training-loop tests | 3/3 PASS |
| CPU checkpoint and journal integrity tests | 5/5 PASS |
| Synthetic failed-E13 recovery tests | 9/9 PASS |
| Mocked runtime lock-lifecycle tests | 4/4 PASS |
| Latest E20 CPU numerical regression | 3/3 PASS |

The first seven rows total 39 scenario checks. The final row is
a further three-test regression run, not three new capabilities.

Numerical and integration tests were run using the Rangpur CPU
PyTorch environment. Several integration tests used synthetic data
or mocked dependencies. They do not establish real-MRI accuracy,
complete Slurm fault recovery or A100 compatibility.

### Predeclared decision criteria

The planned E20 Validation comparison assesses:

- Prostate patient-macro Dice gain versus matched Control >=0.03.
- Pooled Prostate FP reduction versus Control >=15%.
- Prostate recall decrease <=0.05.
- Other-class patient-macro Dice decrease <=0.02 each.
- No Prostate regression for at least four of six Validation patients.
- Required true-positive coverage for classes 2 through 5.
- Prostate performance and minority-class mean relative to
  the selected V3 baseline.

These are acceptance rules, not observed V4 results.

### Evidence and inference boundary

The Validation patients have previously informed development and
parent selection. Any E20 Validation improvement must therefore be
reported as exploratory rather than independent confirmation.

At this snapshot:

- No V4 GPU training or full-volume E20 Validation has run.
- No V4 Held-out Test evaluation has run.
- No V4 performance gain or deployment readiness is claimed.
- Real Slurm SIGKILL, orphaned locks and partial journals still
  require case-specific manual evidence review.
- Final GPU source freeze, checkpoint audit and metric comparison
  remain future execution gates.
