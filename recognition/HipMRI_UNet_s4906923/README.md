# HipMRI Segmentation: Standard 3D U-Net and Residual Improved U-Net v3

**COMP3710 Final Assessment | Student ID: s4906923 | 21 October 2026**

## 1. Problem, scope and engineering decision

This project segments HipMRI volumes into Background, Body, Bones, Bladder, Rectum and Prostate. The prototype supports research into pelvic organ contouring, not clinical treatment planning. The engineering dilemma is whether higher average overlap also gives reliable local boundaries, especially at the prostate base/apex and near adjacent organs, at acceptable computational cost.

The final runnable models are an implemented Standard 3D U-Net and residual Improved v3. v3 leads the original patient-equal Validation criterion and improves existing Test average overlap. It does not dominate Standard for every patient, precision or extreme boundary error. The result supports a qualified research recommendation, not deployment.

**Readiness:** all five required source files, configuration, verified historical results, plots and five real-image cases are included. Local CPU tests and actual-weight parity pass. All 208 Standard/v3 backup files passed server SHA-256 comparison. The new standalone GPU/real-MRI migration replay remains pending; `verify_replay.py` implements that check. Tutor check-off, upstream PR, Turnitin and later code-review response are external actions, not completed by preparing this folder.

![Data and model pipeline](figures/method.png)

## 2. One-page feasibility review

**User need and scope.** Evaluate a six-class improved 3D contouring prototype against an implemented baseline on supplied downsampled HipMRI. Prioritise patient independence, prostate precision and local boundary failures alongside overlap.

**Acceptance criteria.** (1) Paired grids/labels pass checks and patients never cross splits. (2) Implement volumetric residual context and multiscale supervision for the Hard route, reporting the stated Test Dice target of 0.7 with explicit aggregation. (3) Report all-class Dice/IoU, prostate precision/recall and physical boundary errors. (4) Measure peak allocated GPU memory, parameter count and reference latency. (5) Supply runnable commands, configuration, selection provenance and 3-5 representative failure analyses. These research criteria are not clinical safety thresholds.

**Choice and course concepts.** Standard provides volumetric encoder-decoder skip connections. v3 adds residual context, dropout and summed multiscale outputs with GroupNorm(1,C). The work applies 3D convolution, multiclass supervised learning, imbalance weighting and patient-level generalisation analysis. It adapts the three cited papers rather than claiming full reproduction.

**Feasibility evidence.** There are 26/6/6 patients and 143/38/30 volumes. Ten formal models completed 50 epochs. Standard/v3 Validation small-organ scores are 0.8502/0.8564; Test scores are 0.7851/0.8145. A100 profiling below establishes operational feasibility. The original v3 training job completed in 02:02:48 including checkpoint IO and Validation; filesystem load affects wall time.

**Risks, budget and fallback.** Six evaluation patients per split, repeated visits, imbalance, input-context sensitivity and boundary outliers limit inference. Historical all-epoch checkpoint storage exhausted HPC space. The submission trainer saves latest/best by default, with optional epoch weights. Finish packaging and migration checks before further tuning. Keep Standard as a fallback and require new patient-independent evidence for stronger generalisation claims. The earlier proposed edge-sampling intervention was revised after diagnostics; its audit alone is not a completed segmentation improvement.

**Consultation:** this written review does not establish that the compulsory practical check-off was recorded. The student must confirm that separately with the tutor.

## 3. Data, preprocessing and spatial integrity

| Split | Patients | Volumes | Purpose |
|---|---|---|---|
| Train | 26 | 143 | Optimisation and Train-derived weights |
| Validation | 6 | 38 | Checkpoint selection and diagnosis |
| Existing Test | 6 | 30 | Descriptive fixed-checkpoint evaluation |

`config/patient_split.json` freezes the lists. Visits from one patient never cross splits. Files are paired by patient/week. `dataset.py` checks matching 3D shape, affine and voxel spacing, finite arrays and integer labels 0-5. MRI must declare millimetres; matching masks with unknown spatial units are accepted as a documented source-metadata limitation. Metadata agreement is not expert confirmation of anatomy.

MRI and mask are jointly downsampled by (2,2,2). Output centre j samples source centre `(j+0.5)*2-0.5`; the affine uses the same transform. MRI interpolation is linear, mask nearest-neighbour; lost labels trigger failure. Image-only nonzero Z-score preserves outside zeros. GT never defines the inference region.

Arrays remain XYZ until model CDHW conversion. High-end zero padding makes dimensions divisible by 16 and is removed before metrics/export. Evaluation is on the downsampled physical grid, not original resolution. No augmentation is enabled in the selected recipes. CE takes integer labels; Dice converts them to six-channel one-hot internally; `one_hot()` is also available.

Keep raw files outside Git:

```text
HipMRI_Study_open/
  semantic_MRs/PatientID_WeekN_LFOV.nii.gz
  semantic_labels_only/PatientID_WeekN_SEMANTIC.nii.gz
```

## 4. Models, loss and training

| Item | Standard | Improved v3 |
|---|---|---|
| Backbone | Four-level 3D U-Net, width 8 | Residual 3D U-Net, width 8 |
| Normalisation | GroupNorm(1,C) | 14 GroupNorm(1,C) sites |
| Activation / upsampling | ReLU / transposed convolution | LeakyReLU / interpolation plus convolution |
| Residual / dropout | No / none | Yes / 0.3 |
| Output | One full-resolution head | Main + resized half + quarter logits |
| Parameters | 1,412,550 | 2,366,946 |

v3 computes one loss on summed outputs; gradients reach all heads. This is additive multiscale supervision, not the independent head losses tested as v2. v3 changes v1's InstanceNorm to GN1; it is not derived from v2. Standard-v3 differs in several architectural components, so its net difference cannot isolate normalisation alone.

Both use seed 371017 and Adam, fixed LR=0.001, weight decay=0. One shuffled/cycled visit per Train patient supplies ten uniform 64x64x32 XYZ patches per epoch: 260 patches, batch size 2, 130 updates; 50 epochs yield 6,500 updates. The submission preserves the deterministic visit/crop/batch schedule. Exact GPU rerun weights are not guaranteed across kernels, hardware and library versions.

Loss is `0.5 * weighted_cross_entropy + 0.5 * foreground_soft_Dice_loss`. Background remains in CE but is excluded from Dice. Train-derived CE weights are `[1.453871, 1, 2.617199, 4.498586, 8.893215, 9.699950]` in label order. More small-organ emphasis can improve recall at the cost of precision. Lower training loss is not proof of better segmentation.

Validation runs every five epochs. Pool confusion matrices across visits within each patient; compute Bladder/Rectum/Prostate Dice, average within patient, then average six patients equally. Choose the highest score at measured epochs 5,10,...,50, with earlier ties. This is not evaluation of every intermediate epoch. Both selected checkpoints happen to be epoch 50.

![Historical training and checkpoint selection](figures/training_and_selection.png)

## 5. Evaluation definitions and limitations

All models use identical patient splits and evaluation grids. 3D inference processes the complete downsampled volume once, not sliding windows. The supplementary 2D model processes slices and reconstructs a volume. No GT-dependent post-processing is used. Argmax of logits equals argmax after softmax.

Dice=`2TP/(2TP+FP+FN)`, IoU=`TP/(TP+FP+FN)`, precision=`TP/(TP+FP)`, recall=`TP/(TP+FN)`. Absent/absent overlap is null, not automatically perfect. Primary overlap tables pool visits within patients before averaging patients. Foreground macro averages labels 1-5; Background remains in the six-class table. The supplementary post-hoc 50% prostate-weighted score is not a selection rule or official course criterion.

HD95 and ASSD use concatenated directed distances between six-connected mask surfaces, in physical millimetres. HD95 is the 95th percentile; ASSD here is their combined mean. Empty-mask statuses are explicit and null, never zero distance. All five foreground masks were present in all 680 fixed-audit case/model entries. Surface medians/maxima below summarise volumes, unlike patient-equal overlap. Other libraries may use different surface weighting conventions. The new diagnostic refuses sheared grids where spacing-only Euclidean distances would be invalid.

**Test has already been repeatedly inspected.** Checkpoints use Validation, but existing Test is not a new blind holdout and must not guide more tuning. Six patients, not 30 correlated visits, are the independent Test units. No external cohort or independent multi-seed replication was completed.

## 6. Results from every completed model

### 6.1 Best Validation checkpoints

All ten rows completed 50 training epochs. Only Standard/v3 runtime implementations are retained here; other models remain negative/control evidence.

| Model | Best epoch | Val small organs | Test small organs | Test foreground five | Test Prostate |
| --- | --- | --- | --- | --- | --- |
| 2D | 50 | 0.7757 | 0.7082 | 0.7883 | 0.6611 |
| Standard | 50 | 0.8502 | 0.7851 | 0.8370 | 0.7433 |
| v1 | 50 | 0.8420 | 0.8078 | 0.8502 | 0.7520 |
| v2 | 45 | 0.8362 | 0.7959 | 0.8422 | 0.7478 |
| v3 | 50 | 0.8564 | 0.8145 | 0.8631 | 0.7485 |
| A0 | 45 | 0.8450 | 0.7836 | 0.8400 | 0.7318 |
| A1 | 45 | 0.8298 | 0.7801 | 0.8362 | 0.7273 |
| v5-C | 50 | 0.8542 | 0.8040 | 0.8557 | 0.7633 |
| v5-L | 45 | 0.8426 | 0.7858 | 0.8388 | 0.7361 |
| v5-S | 45 | 0.8465 | 0.7788 | 0.8350 | 0.7281 |

| Model | Change and interpretation |
|---|---|
| 2D | Slice-wise width 16, organ-aware crops and different weights; low VRAM, weaker mean prostate overlap. Not a pure 2D/3D factor comparison. |
| v1 | Residual context, InstanceNorm, dropout, additive outputs; severe distant Body false positives. |
| v2 | Independently weighted head losses, main-only inference; no overall Validation gain. |
| v3 | Replace v1 IN with GN1, retaining additive outputs; highest original Validation score. |
| A0 | AdamW + 1e-5 decay + cosine LR; no net gain; combined changes confound attribution. |
| A1 | A0 + paired small rotation/scaling; fewer Body false positives but worse overall overlap. |
| v5-C | Concurrent repeat of v3 recipe; higher Test prostate Dice but lower original Validation score. Same nominal seed is not independent-seed replication. |
| v5-L | Square-root small-organ CE weights relative to C; no overall improvement. |
| v5-S | Adam plus cosine schedule only relative to C; no overall improvement within this budget/seed. |

v3's Test small-organ gain over Standard is +0.0294, but only 3/6 patients improve; excluding influential M015 changes the contrast to approximately -0.0050. Validation improves for 5/6 patients. These descriptive sensitivities do not establish population superiority. C's higher Test prostate Dice is reported without switching the primary selection rule after seeing Test.

### 6.2 Test results by class

| Class | Std Dice | v3 Dice | Std IoU | v3 IoU | Precision Std / v3 | Recall Std / v3 |
| --- | --- | --- | --- | --- | --- | --- |
| Background | 0.9904 | 0.9885 | 0.9814 | 0.9778 | 0.9952 / 0.9961 | 0.9858 / 0.9814 |
| Body | 0.9677 | 0.9732 | 0.9384 | 0.9484 | 0.9748 / 0.9664 | 0.9610 / 0.9807 |
| Bones | 0.8619 | 0.8988 | 0.7578 | 0.8168 | 0.7868 / 0.8793 | 0.9543 / 0.9205 |
| Bladder | 0.7993 | 0.8539 | 0.7455 | 0.7768 | 0.7871 / 0.8807 | 0.8229 / 0.8598 |
| Rectum | 0.8128 | 0.8410 | 0.6918 | 0.7280 | 0.7898 / 0.8086 | 0.8414 / 0.8784 |
| Prostate | 0.7433 | 0.7485 | 0.6081 | 0.6193 | 0.7189 / 0.6983 | 0.8415 / 0.8516 |

![Patient-equal foreground Dice](figures/foreground_comparison.png)

v3 improves all five Test foreground mean Dice values here, but prostate gains only about 0.0052 while precision decreases from 0.7189 to 0.6983. Mean overlap can improve despite more false-positive prostate predictions, and individual cases regress.

### 6.3 Physical boundary reliability

| Class | Median HD95 Std / v3 mm | Worst HD95 Std / v3 mm | Median ASSD Std / v3 mm |
| --- | --- | --- | --- |
| Body | 4.86 / 4.50 | 18.72 / 22.23 | 1.21 / 1.66 |
| Bones | 7.46 / 4.58 | 18.97 / 19.02 | 2.69 / 1.83 |
| Bladder | 3.36 / 3.89 | 109.49 / 72.36 | 1.38 / 1.51 |
| Rectum | 7.06 / 7.28 | 38.56 / 109.19 | 2.46 / 2.09 |
| Prostate | 7.98 / 6.65 | 19.23 / 115.37 | 3.45 / 2.73 |

Median prostate HD95 improves 7.98 to 6.65 mm, while the worst value worsens 19.23 to 115.37 mm. Body/Rectum also retain severe tails. The Hard-route average target of 0.7 is exceeded by v3's Test prostate (0.7485), small-organ (0.8145) and foreground (0.8631) means. The specification does not fully define aggregation; one of six Test patients still has prostate Dice below 0.7. Meeting an average target does not establish local contour safety.

### 6.4 Resources and spatial context

| Model | Parameters | Peak inference GiB | Reference latency s |
| --- | --- | --- | --- |
| 2D | 1,942,374 | 0.0761 | 0.1395 |
| Standard | 1,412,550 | 0.2460 | 0.1552 |
| v3 | 2,366,946 | 0.3458 | 0.1452 |

NVIDIA A100-PCIE-40GB, prepared Validation case K008_Week0, three warm-ups and five repetitions, one resident model. Latency includes transfer, execution, argmax and conversion, excluding decoding/preprocessing/metrics/IO. Peak figures are allocated inference memory, not training memory or device utilisation. Timing depends on shared load/kernels; parameter count is not runtime. Original per-case times and per-patient counts are in the evidence.

2D uses less inference VRAM but its kernels cannot directly see adjacent slices. 3D context can help continuity yet does not ensure correct prostate ends or exclude distant components. The supplementary 2D arm differs in sampling, weights and width; it is not a clean dimension-only ablation and does not claim to meet the separate Easy-route dataset/target. Prostate end-region, OAR and slice-transition diagnostics remain in `evidence/experiments.json`.


## 7. Five real-image failure autopsies

These unchanged historical fixed-checkpoint panels are real MRI, not simulated or retouched contours. Cyan=GT, orange=prediction, red=FP, blue=FN. Seven labelled historical model columns are retained for provenance; compare **Standard** and **v3** for the submitted pair. Open the linked PNGs at full resolution to inspect small structures.

Ordinary panels show the GT-largest-area slice and the slice with largest A1 FP+FN area, shared across models. The Body panel uses A1 maximum-error planes. This purposive selection is not a random sample or a whole-volume score. Prostate base/apex uses physical superior/inferior ordering as a geometric proxy, not expert end-region annotation.

### 7.1 M015_Week7, Test: bladder localisation

![M015 bladder errors](figures/failure_bladder.png)

Standard includes distant non-bladder tissue and misses much of the bladder; v3 reduces distant predictions but still undersegments the target. Whole-volume bladder Dice is 0.0676 Standard versus 0.5224 v3; HD95 is 109.49 versus 58.84 mm. This is improvement within a poor case, not a successful contour. Patient-specific anatomy/intensity is a plausible trigger, not an established cause. The patient's influence requires sensitivity analysis rather than assuming the Test mean represents universal gain.

### 7.2 W029_Week6, Test: prostate ends and adjacent tissue

![W029 prostate ends](figures/failure_prostate_ends.png)

Both models have poor base overlap and extend beyond GT. Whole-volume prostate Dice is 0.4623 Standard versus 0.4377 v3; base-region mean slice Dice is 0.2111 versus 0.2158. Slight end-region improvement does not imply whole-organ improvement. HD95 increases 19.23 to 20.59 mm. Shrinking end cross-sections and neighbouring structures make precision difficult; no dose impact or expert clinical adjudication was measured.

### 7.3 K018_Week4, Validation: Body and distant components

![K018 Body errors](figures/failure_body.png)

v3 greatly reduces the gross outer-volume pattern in v1, but residual distant predictions remain. Standard/v3 Body Dice is almost equal (0.9740/0.9742); HD95 is 4.75/6.24 mm, and distant Background-to-Body voxels are 5/3,128. A large correct interior dominates Dice while small distant components damage surface reliability. Normalisation/context is a supported association, not proof of one isolated cause.

### 7.4 J026_Week4, Validation: rectum extent

![J026 rectum errors](figures/failure_rectum.png)

The difficult lower slice shows predictions beyond the labelled rectum; organ-centre agreement misses this extent failure. Whole-volume Dice improves 0.8110 to 0.8437, ASSD improves 2.00 to 1.73 mm, but HD95 remains 6.24 mm. Precision-recall and extent errors coexist with better average localisation; this is not a claim of anatomical correctness.

### 7.5 B040_Week1, Validation: bone coverage

![B040 Bones errors](figures/failure_bones.png)

The pelvis contains both missing bone and extra predicted regions. v3 improves whole-volume Bones Dice 0.8607 to 0.9026 and HD95 7.41 to 4.58 mm, but the difficult lower slice retains FP and FN. Imbalance/coverage concerns are not limited to three small organs. Bones and Body must be reported explicitly rather than hidden by small-organ-only analysis.

## 8. Mechanisms and engineering recommendation

**Supported observations.** v1 had worse Body HD95 in all 38 Validation volumes: median 76.61 mm versus Standard 4.75 mm. Selected-case sliding windows with unchanged weights reduced distant v1 Body predictions but damaged small-organ overlap. Removing auxiliary outputs from the sum-trained model severely degraded performance. The v3 normalisation intervention improved the recorded primary score and reduced gross v1 Body artifacts, but retained local outliers.

**Unproven explanations.** The evidence supports interactions between context, normalisation and output behaviour, not a unique universal cause. Main-only inference on a sum-trained model is a distribution-changing diagnostic, not a retrained architecture comparison. A Train edge-sampling audit passed, but does not prove segmentation gains. A numerical head-replay control failed with small logit differences; failed controls are not used as causal proof. Loss weighting, optimisation randomness, patient variability and limited exposure remain alternatives.

**Recommendation.** Retain Standard/v3 and finish reproducible delivery; require human contour review. If more research is commissioned, reserve a new patient-independent holdout and estimate repeated-seed uncertainty before another controlled intervention. Evaluate loss/sampling, constrained paired augmentation or inference-context changes using Train/Validation, with both overlap and boundary guardrails. A1/L/S show common methods do not guarantee improvement within this budget. Do not tune against already inspected Test cases or choose a new metric after observing winners. No available evidence proves a better future model is impossible; stopping now is a project-budget decision.

## 9. Environment and executable usage

### 9.1 Install and check

The tested local CPU environment is Python 3.11, PyTorch 2.13.0+cpu, NumPy 2.4.6, SciPy 1.17.1, nibabel 5.4.2 and Matplotlib 3.11.0. `requirements.txt` pins core packages. Historical GPU records specify Python 3.11.15, PyTorch 2.13.0+cu130 and NumPy 2.4.6 on A100; not all historical auxiliary versions were recorded. The CPU lock is not claimed to be a complete historical GPU lock. Install the official CUDA PyTorch wheel compatible with your compute environment; numerical replay is separately verified.

```bash
cd recognition/HipMRI_UNet_s4906923
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B verify_evidence.py
```

In the existing Windows Conda environment, `MKL_THREADING_LAYER=SEQUENTIAL` and `OMP_NUM_THREADS=1` avoided an OpenMP runtime conflict; do not use a duplicate-runtime override. Run GPU training on allocated compute nodes, not the login node. A typical fresh-run allocation is one A100, two CPU cores, up to six hours; load and IO vary.

### 9.2 Train, validate and save

```bash
python -B train.py train --model standard --dataset-root /path/to/HipMRI_Study_open --run-dir /path/to/new_standard --device cuda
python -B train.py train --model v3 --dataset-root /path/to/HipMRI_Study_open --run-dir /path/to/new_v3 --device cuda
```

Training only decodes Train/Validation. Test filenames are enumerated for cohort validation but Test arrays are not loaded. Each completed batch atomically saves `latest.pt` with optimizer, RNG and position. Validation improvements save `best.pt`; epoch JSON and `training.png` are retained. `--keep-epoch-weights` additionally retains all epoch weights if space permits. Compact default storage differs from historical retention, not the mathematical recipe.

```bash
python -B train.py train --model v3 --dataset-root /path/to/HipMRI_Study_open --run-dir /path/to/new_v3 --device cuda --resume
```

Resume requires the same source/config/runtime identity and submission-format recovery files. It does not mutate historical research runs. Two writers are rejected by `.writer.lock`; after a force-kill, confirm no writer remains before removing that stale lock. New 50-epoch training is not required to evaluate existing selected weights.

### 9.3 Evaluate fixed weights and export an example

Supply trusted local checkpoints explicitly; weights are excluded from Git. Best hashes appear in `preparation_verification.json` and the compact experiment evidence. Existing private backups can be used in place without copying into this folder. Historical files contain Python optimizer/RNG objects: load only trusted checkpoints.

```bash
python -B train.py evaluate --model v3 --dataset-root /path/to/HipMRI_Study_open --checkpoint /path/to/v3/best.pt --split validation --output /path/to/new_validation.json --device cuda
python -B train.py evaluate --model v3 --dataset-root /path/to/HipMRI_Study_open --checkpoint /path/to/v3/best.pt --split test --output /path/to/new_fixed_test.json --device cuda
python -B predict.py --model v3 --checkpoint /path/to/v3/best.pt --dataset-root /path/to/HipMRI_Study_open --split validation --case K008_Week0 --output-dir /path/to/new_example --device cuda
```

Use `--model standard` with Standard weights for the baseline. Outputs: `metrics.json`, `overlay.png`, `prediction_downsampled.nii.gz`. Inference uses MRI only; paired GT is used afterwards for metrics/illustration. The exported NIfTI uses the downsampled affine, not original-resolution geometry. Outputs refuse overwrite. Without authorised MRI/weights, use the synthetic tests and evidence verifier; random weights must not be substituted for experimental results.

### 9.4 Real-MRI migration check

```bash
python -B verify_replay.py --dataset-root /path/to/HipMRI_Study_open --standard-checkpoint /path/to/standard/best.pt --v3-checkpoint /path/to/v3/best.pt --output /path/to/new_migration_receipt.json --device cuda
```

This read-only gate compares checkpoint/input hashes, all 76 Standard/v3 Validation confusion matrices, and physical surface measurements with the original audit. It does not train, decode Test or replace historical evidence. CPU actual-weight parity passed; full real-MRI GPU replay remains pending until the new receipt is provided.

### 9.5 Export the report for Turnitin

```bash
python -m pip install reportlab==5.0.1
python -B export_report.py
```

`REPORT.pdf` is generated directly from this README and its figures. The feasibility review occupies one dedicated page. Review the export before submitting; this command does not upload to Turnitin.

## 10. Package map, provenance and assessment mapping

```text
HipMRI_UNet_s4906923/
  modules.py             PyTorch-only Standard/v3 components
  dataset.py             Frozen split, paired preprocessing and sampling
  train.py               Training, Validation selection, explicit fixed evaluation
  predict.py             Saved-model example, metrics, overlay and NIfTI export
  utils.py               Loss, inference, overlap and physical boundaries
  config/                Frozen patient split and training recipe
  tests/                 Synthetic data, geometry, model and recovery tests
  evidence/              Ten-model count records, CSVs and source identities
  figures/               Method, curves, comparison and five real case panels
  verify_evidence.py     Recompute metrics without data or weights
  verify_replay.py       Read-only real-Validation migration gate
  README.md              This report, with a companion PDF export
```

The runtime has no import/path dependency on `20261017`, 2D or retired models. `evidence/experiments.json` preserves 500 epoch summaries and 680 best-model/case entries; integer confusion counts permit overlap recalculation. The verifier checks 400 foreground metric values, checkpoint selection and five original figure hashes. Source/checkpoint identities preserve research provenance without submitting MRI or parameter files. Packaging does not overwrite historical metrics.

Tests cover split leakage rejection, NIfTI pairing, spatial resampling, label retention, padding/crop inversion, patient aggregation, anisotropic distances/empty masks, both models' gradients/loading and interrupted-versus-uninterrupted optimizer recovery. A synthetic 211-case file-list fixture ran the predict CLI end to end; it is not a research cohort or a performance result. Earlier tests also checked actual downloaded best-weight logits against historical implementations. These checks do not substitute for the remaining real-MRI GPU gate.

| Coursework requirement | Included evidence or remaining action |
|---|---|
| Hard volumetric residual model and deep supervision | Model code, architecture explanation, all-head gradients and six-class results |
| Implemented baseline, same split | Standard/v3, frozen split and input/weight identities |
| Per-class DSC/IoU and resource profiling | Six-class table, boundaries, parameters/VRAM/latency |
| Boundary/context dilemma and 3-5 failures | Five original MRI cases, prostate ends, OAR/slice diagnostics, recommendation |
| Modular code and reproducibility | Five required files, dependencies, commands, tests and audit scripts |
| Progressive originality | Existing staged Git history; adapted literature cited below |
| No datasets/model files in Git | Code, count records and selected report visuals only |
| Feasibility hurdle | Written review included; tutor registration must be confirmed |
| Upstream PR and code-review response | Lodge to upstream topic-recognition before the applicable deadline; respond during later feedback window |
| Turnitin | Submit PDF export of this README; confirm it matches the final committed report |

Git push does not prove PR/Turnitin/check-off completion. Check Blackboard for the applicable specification/deadlines. The reviewed local source is the 21-page COMP3710 draft specification; final marking and acceptance remain the teaching team's decision. No claim of full marks is made.

## 11. References and implementation attribution

1. Ronneberger, Fischer and Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. 2015. https://arxiv.org/abs/1505.04597
2. Cicek et al. *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*. 2016. https://arxiv.org/abs/1606.06650
3. Isensee et al. *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge*. 2018. https://arxiv.org/abs/1802.10508

These motivate skips, volumetric context, residual blocks and multiscale supervision. GroupNorm, width, objective and preprocessing are explicit project adaptations; no paper is claimed to validate this exact HipMRI system. The package refactors this project's own verified implementation, not an externally pretrained model. Retain attribution when reusing components.

## Artificial Intelligence Usage Disclosure

ChatGPT/Codex assisted with scaffolding/refactoring, debugging, audits, metric checks, literature interpretation, plotting and report drafting, including the standalone submission packaging. Verification included source/weight hashes, historical initialization and actual-weight CPU logit parity, paired geometry/metric tests, all-head gradients, recovery equivalence and recomputation from integer confusion counts. Historical GPU audits reproduced fixed predictions; the new standalone replay is explicitly pending. Failed controls and negative experiments were not converted into positive findings. The student remains personally responsible for every formula/code line, understanding the work and responding to reviewers; this disclosure is not independent certification of all AI output.
