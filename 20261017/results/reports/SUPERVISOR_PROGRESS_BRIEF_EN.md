# HipMRI Segmentation: Research Progress and Questions for Guidance

**Purpose: Discuss current progress, assessment requirements, and the next research direction with my supervisor or teaching assistant**

**Report date: 21 October 2026**

## 1. What I Hope to Clarify

I have completed a common Validation/Test evaluation of seven earlier models and a Validation-only screen of three new v5 models (C, L, S), including review of real segmentation errors. The three Validation-selected v5 checkpoints have now also completed the descriptive evaluation of the existing Test cohort. I would appreciate guidance on the following: **Does the current work satisfy the assessment's implementation and research requirements? Should I now prioritise the final deliverables and interpretation, or continue training? If further experiments are warranted, which change would provide the greatest research value?**

My current interpretation is:

- **The earlier Improved v3 remains the leading overall candidate.** It leads the current ten-model comparison on the predefined Validation selection metric and existing Test small-organ Dice. In the new concurrent Validation screen, L and S score 0.011675 and 0.007745 below control C, respectively, so neither replaces v3.
- **The goal of clearly outperforming Standard in every respect has not been achieved.** v3 still has extreme boundary errors and performs worse than Standard for some patients and individual metrics. A1 reduces distant Body false positives but weakens overall segmentation performance.
- **Meeting an average Test score target does not establish reliable local contours.** The prostate base and apex, confusion between organs, and individual difficult patients remain important problems.
- All ten models now have Validation/Test class and boundary measurements; v5 also has the same reference-case inference benchmark. The earlier seven models and v5 were measured in separate jobs, so timings are not a strict speed ranking. **I would still like guidance on final assessment compliance, the intended metric aggregation, and whether the generalisation evidence is sufficient**.

## 2. Data and Common Evaluation Method

### 2.1 Dataset

| Item | Train | Validation | Test |
|---|---:|---:|---:|
| Patients | 26 | 6 | 6 |
| MRI volumes | 143 | 38 | 30 |
| Purpose | Parameter optimisation and training sampling | Checkpoint selection and problem diagnosis | Descriptive evaluation of fixed models |

The six classes are **Background, Body, Bones, Bladder, Rectum, and Prostate**. Splitting is performed at patient level: different visits from the same patient do not cross splits.

**How source data are organised and loaded:** Each case is a pair of 3D NIfTI files from one patient's `Week` visit: `PatientID_WeekN_LFOV.nii.gz` for MRI and `PatientID_WeekN_SEMANTIC.nii.gz` for the six-class mask, for example `K018_Week4`. `Week` is a visit identifier in the filename; it does not mean every patient has a scan every week. The 26 Train patients contribute 143 volumes. Code pairs files by patient ID and week, checks the 3D shape, affine, and voxel spacing, then loads the whole volume, downsamples MRI and mask together, and normalises MRI. A preprocessed training visit is cached for repeated crops. See [case discovery and pairing](../../code/hipmri_rewrite/splits.py) and [3D loading and preprocessing](../../code/hipmri_rewrite/data.py).

**What enters the networks:** In each epoch, the fixed visit-cycle rule selects **one volume per Train patient**; ten paired MRI/mask samples are then drawn from that volume. This yields 26×10=260 samples and 130 updates at batch size 2. Standard and Improved models, including v5 C/L/S, choose a uniform origin along all three spatial axes and crop a **64×64×32-voxel 3D patch** for the 3D network. They do not train on one 2D slice at a time. Only the supplementary 2D U-Net selects one Z slice and a **64×64-pixel 2D crop**: of ten draws per patient, four choose slices uniformly and two each target Bladder, Rectum, and Prostate. Convolution kernels then scan positions *within* each input patch: the 2D model uses 2D kernels, while the 3D models use 3D kernels spanning neighbouring slices. Drawing a training patch and sliding convolution kernels inside it are separate operations. See [3D sampling](../../code/hipmri_rewrite/sampling.py), [2D sampling](../../code/hipmri_rewrite/sampling_2d.py), and [convolution layers](../../code/hipmri_rewrite/models/common.py).

Validation/Test inference differs from training crops. In the formal evaluation, the **whole downsampled 3D MRI** is padded as needed and passed through the 3D network once; the 2D network processes every Z slice in batches of up to eight, then reassembles a 3D prediction. See [inference implementation](../../code/hipmri_rewrite/inference.py).

### 2.2 Conditions Shared Across Models

| Condition | Implementation |
|---|---|
| Preprocessing | MRI and masks share the same grid; downsampling by a factor of 2 along each axis; linear interpolation for MRI and nearest-neighbour interpolation for masks; per-volume Z-score normalisation using nonzero MRI voxels |
| Training budget | All seven earlier runs and three new v5 runs completed 50 epochs; 130 batches per epoch, batch size=2, giving 6500 parameter updates per run |
| Patient sampling | One visit per Train patient per epoch, cycling through visits according to a predefined rule; 10 training samples per patient |
| Model input | 3D patches: 64×64×32; 2D crops: 64×64 |
| Validation and selection | Validation every 5 epochs; select the best at epochs 5, 10, …, 50, giving 10 measured candidates; choose the earlier epoch in a tie |
| Primary selection metric | Mean Dice for Bladder, Rectum, and Prostate, with equal weighting of the six Validation patients |
| Patient aggregation | Pool confusion matrices across each patient's visits, calculate metrics, then average across patients; the 30 Test volumes are not treated as 30 independent patients |
| Saving and reproducibility | Retain epoch weights, training records, the best checkpoint, and recovery state; preserve configuration and source identities, including interruption and recovery records |

**Clarification:** “50 training epochs” does not mean 50 batches. The Test cohort has already been inspected during the research process, so this evaluation cannot be described as a new blind test. All checkpoints were selected using Validation, rather than choosing the best epoch on Test.

## 3. What Changed Across All Ten Runs?

### 3.1 Architecture and Supervision

| Model | Architecture and width | Normalisation | Residual connections / Dropout | Output and supervision | Main research purpose |
|---|---|---|---|---|---|
| 2D U-Net | Standard 2D U-Net; base channels=16 | GroupNorm (GN), 1 group | No residual connections; no explicit Dropout | Single output head | Supplementary slice-wise comparison with low VRAM use |
| Standard 3D | Standard 3D U-Net; base channels=8 | GN, 1 group | No residual connections; no explicit Dropout | Single output head | Establish the 3D baseline |
| Improved v1 | Residual 3D U-Net; base channels=8 | InstanceNorm | Residual connections; Dropout=0.3 | Sum the main logits and upsampled half- and quarter-scale logits; calculate loss on the sum | Introduce residual architecture and multiscale supervision |
| Improved v2 | Retains the v1 backbone | InstanceNorm | Same as v1 | Separate losses for three heads, weighted by 1, 0.5, and 0.25 and divided by 1.75; main head only at inference | Investigate supervision and output fusion |
| Improved v3 | Modified from v1; **not from v2** | Replace 14 IN sites with GN, 1 group | Same as v1 | Retains the v1 additive-logit scheme | Investigate normalisation as a single architectural change |
| v4 A0 | Same as v3 | GN, 1 group | Same as v3 | Same as v3 | Change Adam to AdamW, add 0.00001 weight decay, and replace the fixed learning rate with per-update cosine decay; no spatial augmentation |
| v4 A1 | Same as v3 | GN, 1 group | Same as v3 | Same as v3 | Retain A0's AdamW, weight decay, and cosine schedule; additionally apply paired MRI/mask rotations of up to ±10° on three axes and 0.9–1.1 scaling with 30% probability |
| v5 C | Same as v3; concurrent control | GN, 1 group | Same as v3 | Same as v3 | Repeat the v3 recipe alongside L and S |
| v5 L | Same as C | GN, 1 group | Same as C | Same as C | Take the square root of the Bladder/Rectum/Prostate CE weights only |
| v5 S | Same as C | GN, 1 group | Same as C | Same as C | Add per-update cosine learning-rate decay to Adam only |

The standard U-Nets use pooling and transposed-convolution upsampling. The Improved family uses strided convolutions and interpolation followed by convolution, and also differs in its activation functions. Standard and v1 therefore differ by more than one module. The Improved model is a HipMRI adaptation inspired by Isensee and colleagues, rather than a claimed full reproduction of the original paper.

### 3.2 Training Recipe and Class Imbalance

| Model | Optimiser | Learning rate | Weight decay | Patch / crop sampling | Spatial augmentation |
|---|---|---|---|---|---|
| 2D | Adam | Fixed at 0.001 | 0 | Patient-balanced, organ-aware 2D crops | None |
| Standard | Adam | Fixed at 0.001 | 0 | Patient-balanced, uniform 3D patches | None |
| v1 | Adam | Fixed at 0.001 | 0 | Same as Standard | None |
| v2 | Adam | Fixed at 0.001 | 0 | Same as Standard | None |
| v3 | Adam | Fixed at 0.001 | 0 | Same as Standard | None |
| A0 | AdamW | Cosine decay at every parameter update, from 0.001 towards 0.00001 | 0.00001; bias and normalisation parameters excluded | Same as Standard | None |
| A1 | AdamW | Same as A0 | Same as A0 | Same as A0 | Probability 30%; rotation up to ±10° around each physical XYZ axis; isotropic scaling of 0.9–1.1 |
| C | Adam | Fixed at 0.001 | 0 | Same uniform 3D patch rule as v3 | None |
| L | Adam | Fixed at 0.001 | 0 | Same as C; only small-organ CE weights differ | None |
| S | Adam | Per-update cosine decay from 0.001 to 0.00001 | 0 | Same as C | None |

The base loss for all ten runs is **0.5 weighted cross-entropy + 0.5 foreground Soft Dice**. v2 applies this loss separately to each head. Static class weights originate from Train data; L applies a declared square-root transformation to the three small-organ weights:

| Weight order | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| Earlier six 3D models and v5 C/S | 1.453871 | 1.000000 | 2.617199 | 4.498586 | 8.893215 | 9.699950 |
| v5 L | 1.453871 | 1.000000 | 2.617199 | 2.120987 | 2.982149 | 3.114474 |
| Supplementary 2D model | 1.517047 | 1.000000 | 2.520386 | 4.124822 | 7.521774 | 8.819274 |

In A1, MRI and mask receive exactly the same spatial transformation, with nearest-neighbour interpolation for the mask. No flips, strong elastic deformation, or intensity augmentation were used. The augmentation audit across 26 Train patients passed before training.

**Comparison limitations:** A1−A0 primarily measures spatial augmentation within this recipe. A0−v3 simultaneously changes Adam to AdamW, weight decay, and the learning-rate schedule, so that historical contrast does not isolate the schedule. The new S−C comparison changes only the learning-rate schedule and is reported in Section 8; one seed limits its interpretation. The 2D and 3D recipes also differ in sampling, channel width, and class weights; their performance differences cannot all be attributed to spatial dimensionality.

## 4. All Ten Models: Each Model's Best Validation Checkpoint

### 4.1 Overall Results

All Dice values below give equal weight to patients; higher is better. “Small organs” refers to Bladder, Rectum, and Prostate.

| Model | Best epoch | Validation small-organ Dice | Test small-organ Dice | Test Prostate Dice | Test mean Dice over five foreground classes |
|---|---:|---:|---:|---:|---:|
| 2D | 50 | 0.7757 | 0.7082 | 0.6611 | 0.7883 |
| Standard | 50 | 0.8502 | 0.7851 | 0.7433 | 0.8370 |
| v1 | 50 | 0.8420 | 0.8078 | 0.7520 | 0.8502 |
| v2 | 45 | 0.8362 | 0.7959 | 0.7478 | 0.8422 |
| v3 | 50 | **0.8564** | **0.8145** | 0.7485 | **0.8631** |
| A0 | 45 | 0.8450 | 0.7836 | 0.7318 | 0.8400 |
| A1 | 45 | 0.8298 | 0.7801 | 0.7273 | 0.8362 |
| C | 50 | 0.8542 | 0.8040 | 0.7633 | 0.8557 |
| L | 45 | 0.8426 | 0.7858 | 0.7361 | 0.8388 |
| S | 45 | 0.8465 | 0.7788 | 0.7281 | 0.8350 |

Historical v3 leads the ten Validation small-organ means (0.8564 versus C's 0.8542) and the existing Test small-organ and five-foreground means. C has the highest Test Prostate Dice (0.7633) and also a higher Rectum Dice. No model is established as best on every measured metric.

### 4.2 Six-Class Dice: Test and Validation

Test results are now complete for all ten models, with every checkpoint still fixed by Validation. All numbers are patient-equal means.

| Model | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.9896 | 0.9681 | 0.8487 | 0.7508 | 0.7127 | 0.6611 |
| Standard | 0.9904 | 0.9677 | 0.8619 | 0.7993 | 0.8128 | 0.7433 |
| v1 | 0.9683 | 0.9470 | 0.8808 | 0.8428 | 0.8285 | 0.7520 |
| v2 | 0.9783 | 0.9565 | 0.8668 | 0.8484 | 0.7916 | 0.7478 |
| v3 | 0.9885 | **0.9732** | **0.8988** | **0.8539** | 0.8410 | 0.7485 |
| A0 | 0.9892 | 0.9704 | 0.8786 | 0.8002 | 0.8189 | 0.7318 |
| A1 | 0.9890 | 0.9686 | 0.8721 | 0.8236 | 0.7894 | 0.7273 |
| C | 0.9883 | 0.9713 | 0.8951 | 0.7951 | 0.8536 | 0.7633 |
| L | 0.9891 | 0.9707 | 0.8661 | 0.8068 | 0.8144 | 0.7361 |
| S | 0.9882 | 0.9692 | 0.8694 | 0.7887 | 0.8198 | 0.7281 |

For a measured six-class comparison of **all ten models**, the following table gives patient-equal Validation Dice at each model's selected checkpoint:

| Model | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.9954 | 0.9717 | 0.8328 | 0.8775 | 0.6891 | 0.7605 |
| Standard | 0.9968 | 0.9770 | 0.8583 | 0.9303 | 0.7989 | 0.8215 |
| v1 | 0.9693 | 0.9349 | 0.8810 | 0.9437 | 0.7853 | 0.7970 |
| v2 | 0.9780 | 0.9475 | 0.8534 | 0.9322 | 0.7714 | 0.8050 |
| v3 | 0.9953 | 0.9790 | 0.8907 | 0.9213 | 0.8186 | 0.8292 |
| A0 | 0.9960 | 0.9776 | 0.8732 | 0.9082 | 0.8034 | 0.8234 |
| A1 | 0.9942 | 0.9728 | 0.8603 | 0.8917 | 0.7855 | 0.8121 |
| C | 0.9966 | 0.9804 | 0.8875 | 0.9082 | 0.8287 | 0.8258 |
| L | 0.9944 | 0.9737 | 0.8665 | 0.9292 | 0.7834 | 0.8151 |
| S | 0.9961 | 0.9774 | 0.8720 | 0.9078 | 0.8034 | 0.8283 |

### 4.3 Six-Class IoU: Test and Validation

Test IoU is now available for the three v5 branches; six-class Validation Dice appears above and Validation IoU below.

| Model | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.9799 | 0.9388 | 0.7380 | 0.6668 | 0.5552 | 0.5080 |
| Standard | 0.9814 | 0.9384 | 0.7578 | 0.7455 | 0.6918 | 0.6081 |
| v1 | 0.9414 | 0.9018 | 0.7902 | 0.7734 | 0.7089 | 0.6183 |
| v2 | 0.9587 | 0.9177 | 0.7664 | 0.7721 | 0.6568 | 0.6114 |
| v3 | 0.9778 | **0.9484** | **0.8168** | **0.7768** | **0.7280** | 0.6193 |
| A0 | 0.9791 | 0.9431 | 0.7841 | 0.7289 | 0.6971 | 0.5969 |
| A1 | 0.9786 | 0.9393 | 0.7736 | 0.7412 | 0.6573 | 0.5916 |
| C | 0.9776 | 0.9453 | 0.8107 | 0.7343 | 0.7471 | 0.6322 |
| L | 0.9791 | 0.9436 | 0.7659 | 0.7404 | 0.6908 | 0.6023 |
| S | 0.9772 | 0.9410 | 0.7709 | 0.7187 | 0.6992 | 0.5951 |

Patient-equal **Validation IoU** for all ten selected checkpoints:

| Model | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.9907 | 0.9450 | 0.7138 | 0.7856 | 0.5270 | 0.6139 |
| Standard | 0.9936 | 0.9551 | 0.7519 | 0.8709 | 0.6670 | 0.6986 |
| v1 | 0.9410 | 0.8801 | 0.7876 | 0.8943 | 0.6497 | 0.6671 |
| v2 | 0.9571 | 0.9013 | 0.7445 | 0.8745 | 0.6304 | 0.6777 |
| v3 | 0.9906 | 0.9590 | 0.8031 | 0.8561 | 0.6954 | 0.7089 |
| A0 | 0.9921 | 0.9561 | 0.7751 | 0.8347 | 0.6733 | 0.7007 |
| A1 | 0.9885 | 0.9471 | 0.7553 | 0.8095 | 0.6492 | 0.6843 |
| C | 0.9932 | 0.9616 | 0.7979 | 0.8356 | 0.7100 | 0.7045 |
| L | 0.9889 | 0.9488 | 0.7648 | 0.8695 | 0.6465 | 0.6903 |
| S | 0.9922 | 0.9558 | 0.7732 | 0.8339 | 0.6730 | 0.7080 |

Dice and IoU are calculated separately before averaging across patients. Mean IoU is not obtained by transforming the mean Dice shown in the table.

## 5. Boundaries and Local Risks Beyond Average Dice

### 5.1 Test Prostate End Regions and Validation Surface Errors

HD95 is measured in millimetres, with lower values indicating better performance. The table reports medians and maxima across volumes. Regional Dice is a patient-equal average of slice-region scores and differs from whole-volume Dice.

| Model | Prostate apex Dice | Central Dice | Base Dice | Median Body HD95 / mm | Median Prostate HD95 / mm | Maximum Prostate HD95 / mm |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.5186 | 0.7125 | 0.3473 | 6.055 | 9.700 | 15.600 |
| Standard | **0.7544** | 0.7824 | 0.5097 | 4.861 | 7.979 | 19.229 |
| v1 | 0.7397 | **0.7997** | **0.5993** | 51.813 | 6.719 | 122.347 |
| v2 | 0.7120 | 0.7877 | 0.5836 | 28.555 | 7.460 | 16.717 |
| v3 | 0.7523 | 0.7931 | 0.5069 | 4.500 | **6.645** | 115.371 |
| A0 | 0.7169 | 0.7687 | 0.4839 | 4.751 | 8.611 | 23.712 |
| A1 | 0.7251 | 0.7620 | 0.5472 | 5.684 | 9.265 | 22.211 |
| C | 0.7175 | 0.7971 | 0.4969 | 3.359 | 7.175 | 86.844 |
| L | 0.7147 | 0.7843 | 0.4519 | 5.638 | 8.033 | 75.569 |
| S | 0.7001 | 0.7637 | 0.4388 | 4.696 | 9.749 | 21.121 |

The following **Validation** HD95 measurements cover all ten selected checkpoints. They are volume medians or maxima across the 38 scans, not patient-equal scores; lower is better. The v5 Test apex/base measurements are now complete under the same definition; no new Validation end-region measurement was added.

| Model | Body median / mm | Rectum median / mm | Rectum maximum / mm | Prostate median / mm | Prostate maximum / mm |
|---|---:|---:|---:|---:|---:|
| 2D | 6.72 | 29.80 | 47.96 | 7.43 | 43.68 |
| Standard | 4.75 | 12.00 | 28.74 | 5.68 | 14.20 |
| v1 | 76.61 | 14.16 | 167.00 | 5.68 | 150.46 |
| v2 | 68.26 | 17.12 | 238.87 | 5.08 | 13.92 |
| v3 | 4.58 | 7.09 | 31.83 | 4.67 | 106.12 |
| A0 | 4.75 | 8.51 | 26.03 | 5.13 | 21.16 |
| A1 | 5.68 | 9.36 | 28.08 | 6.02 | 13.34 |
| C | 4.50 | 6.24 | 22.10 | 4.58 | 20.16 |
| L | 5.22 | 13.65 | 175.48 | 4.58 | 112.51 |
| S | 4.86 | 9.42 | 35.77 | 4.58 | 23.82 |

**Main concern:** 3D improves overall end-region performance relative to 2D, but the base remains weak. v3's favourable median does not remove its extreme Test surface error of 115.37 mm; an extreme Validation case also reaches 106.12 mm. Neither average Dice nor median HD95 establishes suitability for automated clinical contouring.

### 5.2 False Positives, Organ Confusion, and Continuity

| Model | Validation distant Body FP voxels | Test distant Body FP voxels | Test predicted Prostate voxels inside true Bladder/Rectum | Test volumes with multiple Prostate components / 30 |
|---|---:|---:|---:|---:|
| 2D | 48,152 | 11,580 | 2.983% | 14 |
| Standard | 1,007 | 2,804 | 3.007% | 4 |
| v1 | 1,358,516 | 667,239 | 4.473% | 5 |
| v2 | 820,635 | 402,929 | 1.902% | 1 |
| v3 | 73,204 | 81,145 | 3.004% | 6 |
| A0 | 9,572 | 9,636 | 2.790% | 2 |
| A1 | 1,171 | **535** | 4.937% | 2 |
| C | 2,229 | 33,341 | 1.614% | 3 |
| L | 104,524 | 39,115 | 3.855% | 8 |
| S | 7,272 | 5,919 | 3.131% | 4 |

Distant Body false positives are restricted to true Background voxels predicted as Body and at least 20 mm from the true Body. C/L/S counts are now available for both Validation and Test, covering 38 and 30 volumes respectively; counts are not added across splits. The organ-at-risk (OAR) percentage uses predicted Prostate voxels as its denominator; it measures class confusion, not radiation dose or probability of organ injury. Components use 6-connectivity, and their count does not directly establish anatomical correctness.

**A1 shows a clear trade-off:** its distant Body false positives on Test are 94.45% lower than A0, but small-organ Dice, prostate surface errors, and OAR confusion do not improve together. An improvement in one metric should not be presented as overall model superiority.

## 6. Findings from Real-Image Review

| Case | Observations and whole-volume measurements | Research implication |
|---|---|---|
| K018_Week4, Validation, Body | v1 predicts large regions outside the body; Body HD95 is 125.632 mm. A1 reduces this to 5.684 mm, but undersegmentation remains near the contour | Reducing distant FP and precisely following the boundary are different objectives |
| M015_Week7, Test, Bladder | Standard produces extensive predictions outside GT, with Dice 0.0676. v3 improves to 0.5224 but still has substantial undersegmentation | Difficult patients influence the average Improved-model gain; improvement does not imply success |
| W029_Week6, Test, Prostate | 2D Dice is 0.6485, above Standard 0.4623, v3 0.4377, and A1 0.4057. A1 recall is 0.9829 but precision only 0.2556 | High 3D recall can result from oversegmentation; recall alone is insufficient |
| J026_Week4, Validation, Rectum | v3 Dice is 0.8437 versus A1 at 0.7851; A1 precision falls to 0.6628 | Augmentation does not improve every organ; local expansion needs a specific explanation |
| B040_Week1, Validation, Bones | v1 Dice is 0.9102, v3 0.9026, and A1 0.8743; misclassification and omission remain near the ends | Bones must be evaluated alongside the three small organs |

MRI, GT, predictions, and FP/FN visualisations were inspected for these cases. Metrics were not estimated from screenshots. These are purposeful failure analyses, not estimates of failure prevalence. No independent clinical expert review has yet been conducted.

## 7. Computational Cost and Generalisation Evidence

### 7.1 Resource Measurements on the A100 40 GB GPU

The first table now contains reference-volume inference VRAM and per-Test-volume inference timing for all ten models. The v5 completion uses the same reference case and warm-up/repeat protocol but a separate job. These are not training-memory or training-time measurements.

| Model | Parameters | Reference-volume peak allocated VRAM / GiB | Median single-pass inference per Test volume / s |
|---|---:|---:|---:|
| 2D | 1,942,374 | 0.076 | 0.1401 |
| Standard | 1,412,550 | 0.246 | 0.1740 |
| v1 | 2,366,946 | 0.346 | 0.1144 |
| v2 | 2,366,946 | 0.346 | 0.1073 |
| v3 | 2,366,946 | 0.346 | 0.1666 |
| A0 | 2,366,946 | 0.346 | 0.1666 |
| A1 | 2,366,946 | 0.346 | 0.1666 |
| C | 2,366,946 | 0.346 | 0.1645 |
| L | 2,366,946 | 0.346 | 0.1520 |
| S | 2,366,946 | 0.346 | 0.1432 |

For the earlier seven models, the reference volume uses 3 warm-ups and 5 synchronised repetitions; single-pass inference is also recorded for every Test volume. Timing excludes disk access, preprocessing, and metric calculation. The networks are small and inputs are downsampled, so these costs should not be extrapolated to large models at original resolution. v4 was interrupted by full storage and then resumed; its total training wall-clock time is unsuitable as a direct comparison of pure GPU training speed.

The **new v5 C/L/S runs** use the same v3 architecture and therefore have the same 2,366,946 trainable parameters. Their separate training logs provide the measured cells below; comparable training-resource logs for the earlier seven runs are not included in this v5 analysis. Training durations sum each epoch's recorded batch processing and per-batch checkpoint I/O; Validation durations sum the ten recorded full-Validation passes. They exclude queue time, initialisation, end-of-epoch file copies, and other unmeasured overhead.

| Model | Trainable parameters | Training + checkpoint time / h | Recorded Validation time / min | Peak training allocated VRAM / GiB | Matched v5 inference benchmark |
|---|---:|---:|---:|---:|---|
| 2D | 1,942,374 | Not available under v5 protocol | Not available | Not available | Not applicable; earlier inference above |
| Standard | 1,412,550 | Not available under v5 protocol | Not available | Not available | Not applicable; earlier inference above |
| v1 | 2,366,946 | Not available under v5 protocol | Not available | Not available | Not applicable; earlier inference above |
| v2 | 2,366,946 | Not available under v5 protocol | Not available | Not available | Not applicable; earlier inference above |
| v3 | 2,366,946 | Not available under v5 protocol | Not available | Not available | Not applicable; earlier inference above |
| A0 | 2,366,946 | Not available under v5 protocol | Not available | Not available | Not applicable; earlier inference above |
| A1 | 2,366,946 | Not available under v5 protocol | Not available | Not available | Not applicable; earlier inference above |
| C | 2,366,946 | 3.083 | 21.76 | 0.285 | Reference-case median 0.1535 s |
| L | 2,366,946 | 3.001 | 22.65 | 0.285 | Reference-case median 0.1407 s |
| S | 2,366,946 | 3.033 | 20.17 | 0.285 | Reference-case median 0.1190 s |

These v5 peak figures are PyTorch *allocated training memory*, not process-wide GPU use, and are measured under a different workload from the first table's reference-volume inference peaks. The v5 timings are observational runs on the same A100 model, not a controlled speed benchmark; their small differences do not establish that a class-weight or learning-rate change makes inference faster. The completion now includes the same K008_Week0 benchmark, with 3 warm-ups and 5 synchronised repetitions; cross-job GPU load and kernel differences remain. [v5 resource summary](../experiments/improved_3d_v5/comparisons/validation_analysis/tables/summary.csv).

### 7.2 What Can Currently Be Said About Generalisation?

| Evidence | Observation | Limitation |
|---|---|---|
| Patient-level Validation comparison | v3 improves over Standard by 0.00613; 5/6 patients improve | Only 6 patients, with repeated reuse of Validation during research |
| Patient-level Test comparison | v3 improves over Standard by 0.02938, but only 3/6 patients improve | Not every patient benefits |
| M015 sensitivity analysis | Removing M015 for a descriptive sensitivity check changes v3−Standard to −0.00496 | This is not permission to remove the patient from the official score; the advantage is sensitive to cohort composition |
| A1 versus v3 | A1 has lower Test small-organ scores for 6/6 patients | The results do not support stronger generalisation for A1 |
| Replication | One fixed seed per recipe, including v5 C/L/S | Stability across seeds is untested; there is no external cohort or new blind test |

## 8. New v5 Models: Validation Single-Factor Screen and Fixed-Checkpoint Test Completion

The new C/L/S runs use the v3 residual 3D U-Net with eight base channels, one-group GroupNorm, Dropout=0.3, and loss on the sum of three output logits. They share the initial weights, seed 371017, patient and patch order, 50 epochs, and 130 updates per epoch; none uses spatial or intensity augmentation. The best checkpoint is selected every five epochs using equal-patient mean Bladder/Rectum/Prostate Dice across 38 scans from six Validation patients. The first seven rows below provide historical context; only L−C and S−C are concurrent single-factor contrasts. The fixed C50/L45/S45 checkpoints subsequently completed descriptive evaluation on the existing Test cohort, without selecting checkpoints on Test.

| Model | Recipe or comparison role | Best epoch | Validation small-organ Dice | Difference from C | Patients above C |
|---|---|---:|---:|---:|---:|
| 2D | Historical slice-wise baseline; separate sampling and weights | 50 | 0.775688 | Not a concurrent contrast | Not compared |
| Standard | Historical 3D baseline | 50 | 0.850223 | Not a concurrent contrast | Not compared |
| v1 | Historical residual 3D, InstanceNorm, summed logits | 50 | 0.841996 | Not a concurrent contrast | Not compared |
| v2 | Historical v1 backbone with separate head losses | 45 | 0.836199 | Not a concurrent contrast | Not compared |
| v3 | Historical GN1 residual 3D, summed logits | 50 | **0.856356** | Not a concurrent contrast | Not compared |
| A0 | Historical AdamW, weight decay, cosine schedule | 45 | 0.845002 | Not a concurrent contrast | Not compared |
| A1 | Historical A0 plus paired spatial augmentation | 45 | 0.829764 | Not a concurrent contrast | Not compared |
| C | Concurrent v3 control: Adam, fixed 0.001 learning rate, zero weight decay; small-organ CE weights 4.498586/8.893215/9.699950 | 50 | **0.854239** | — | — |
| L | Class weights only: take the square root of Bladder/Rectum/Prostate CE weights, giving 2.120987/2.982149/3.114474; retain Adam at fixed 0.001 | 45 | 0.842564 | −0.011675 | 2/6 |
| S | Learning rate only: retain C's CE weights, Adam, and zero weight decay; change fixed 0.001 to per-update cosine decay ending at 0.00001 | 45 | 0.846494 | −0.007745 | 2/6 |

All ten runs use 0.5 weighted cross-entropy plus 0.5 foreground Soft Dice, with v2 applying separate head losses. Weighted CE computes a weighted mean, so changing L's weights also changes its normalising denominator. S isolates the learning-rate schedule, whereas v4 A0 also changed the optimiser and weight decay. Historical v3 scored 0.856356 on Validation, 0.002118 above concurrent C: even the nominally identical recipe was not reproduced numerically. Each new branch has only one seed.

| Validation measure | C | L | S | Specific trade-off |
|---|---:|---:|---:|---|
| Bladder Dice | 0.90824 | **0.92916** | 0.90778 | L gains 0.02091 |
| Rectum Dice | **0.82869** | 0.78341 | 0.80339 | L loses 0.04528; S loses 0.02529 |
| Prostate Dice | 0.82579 | 0.81513 | **0.82831** | S gains only 0.00252 |
| Body / Bones Dice | **0.98041 / 0.88751** | 0.97368 / 0.86646 | 0.97740 / 0.87202 | Both interventions lose on both classes |
| Rectum precision / recall | **0.78137 / 0.89041** | 0.69352 / 0.91042 | 0.72402 / 0.91036 | Higher recall comes with lower precision |
| Prostate precision / recall | **0.82867 / 0.83078** | 0.77758 / 0.86331 | 0.79964 / 0.86534 | Both trade precision for recall |
| Distant Body false-positive voxels | **2,229** | 104,524 | 7,272 | L is about 46.9 times C; S about 3.3 times C |
| Rectum median / maximum HD95, mm | **6.24 / 22.10** | 13.65 / 175.48 | 9.42 / 35.77 | L has a particularly severe boundary tail |
| Prostate median / maximum HD95, mm | 4.58 / **20.16** | 4.58 / 112.51 | 4.58 / 23.82 | Equal medians conceal different tails |

Real Validation images show opposite effects across organs within the same intervention. On S022_Week6, L raises Bladder Dice from C's 0.594 to 0.925, while Rectum Dice falls from 0.874 to 0.785 and L's Rectum HD95 reaches 175.48 mm. On J026_Week6, L lowers Prostate Dice from 0.810 to 0.646 and raises HD95 from 4.75 to 112.51 mm. On K018_Week4, distant Body false positives increase from 313 with C to 19,913 with L. S improves K018_Week2 Prostate Dice from 0.782 to 0.882, but its six-patient small-organ mean remains below C. These cases were selected to explain differences, not estimate error prevalence; central slices alone cannot explain extreme full-volume HD95 values.

**Decision:** Retain historical v3 as the existing main candidate and C as the concurrent v5 control; do not promote L or S or use existing Test scores to select a v5 checkpoint. The 156 JSON files covering 50 epochs plus frozen configuration and manifest per branch were checked; patient confusion matrices reproduce all 30 Validation scores. Reloaded weights reproduced 12 confusion matrices on four shared cases. Test completion is available, but there is no second seed, external cohort, or independent clinical review, so statistical significance or better generalisation is not established. See the [full v5 Validation report (Chinese)](../experiments/improved_3d_v5/reports/VALIDATION_SCREEN_ZH.md) and [six-class numerical metrics](../experiments/improved_3d_v5/comparisons/validation_analysis/tables/six_class_patient_equal.csv).

### 8.1 Fixed-Checkpoint Test Completion and Decision

Job 682322 evaluated the existing Test cohort after fixing C50/L45/S45, completing in 6 minutes with exit code 0:0. Each model first reproduced its K008_Week0 Validation confusion matrix, then evaluated the same 6 patients and 30 volumes. This is not a new blind test and Test was not used to change epochs.

| Model | Test small-organ Dice | Difference from C | Test Prostate Dice | Test distant Body FP |
|---|---:|---:|---:|---:|
| C | 0.804004 | +0.000000 | 0.763266 | 33,341 |
| L | 0.785782 | -0.018222 | 0.736148 | 39,115 |
| S | 0.778836 | -0.025168 | 0.728097 | 5,919 |

L/S remain below concurrent C on existing Test, consistent with the Validation decision; S reduces distant Body false positives but loses small-organ performance. Historical v3 and nominally identical C differ by 0.01047 in Test small-organ Dice, so small recipe effects cannot all be treated as proven causal mechanisms. Retain v3 as the main candidate and Standard as the strong baseline; if further research is required, prioritise discussion of a matched second-seed confirmation rather than seeking recipes that win on the inspected Test cohort.

## 9. Development Diagnostics: Has Class Imbalance Been Addressed?

In addition to the ten formal training runs, development included learning diagnostics on two fixed patches from the same training case:

| Fixed-sample diagnostic | Updates | Training Rectum Dice | Training Prostate Dice | Supported interpretation |
|---|---:|---:|---:|---|
| Standard, without class weights in the CE term | 100 | 0.0000 | 0.0000 | The two small organs were not learned under this setting |
| Standard, without class weights, diagnostic budget increased to 500 steps | 500 | 0.0000 | 0.0000 | Additional updates alone did not resolve this fixed-sample problem |
| Standard, with Train-derived class weights in CE | 500 | 0.9796 | 0.9722 | The network can fit the small organs, and weighting has a measurable effect |
| Improved v1, without the above class weighting | 100 | 0.9586 | 0.9393 | Faster fitting on this fixed sample does not establish better generalisation |

These are training scores on the same fixed batch, not Validation or Test results, and they are not four additional formal generalisation models. Subsequent formal training consistently used weighted CE plus foreground Dice; 2D additionally used organ-aware sampling. v5 L separately tested taking the square root of all three small-organ CE weights: Bladder precision rose from 0.87857 to 0.91249, while Rectum precision fell from 0.78137 to 0.69352 and Prostate precision from 0.82867 to 0.77758. Class imbalance has therefore been addressed, but these opposing changes cannot yet be assigned to any one class weight or the loss composition; there is no independent ablation of a single class weight or the CE/Dice ratio.

## 10. Does the Work Currently Meet the Assessment Requirements?

The following is a provisional interpretation of the course specification draft currently available to me. **I would appreciate confirmation from my supervisor or teaching assistant.**

| Assessment requirement | Current status | Clarification needed |
|---|---|---|
| Hard: 3D residual units, deep supervision, and downsampled HipMRI | Implementation, training, and evaluation evidence available | Does the adapted architecture satisfy the Hard implementation requirement? |
| Test Dice ≥0.7 | All nine 3D models have patient-equal mean Test Prostate Dice ≥0.7 | Does the target apply to Prostate, five foreground classes, all classes, or another aggregate? Must every patient meet it? |
| Standard baseline and per-class Test Dice/IoU on identical splits | Completed for all ten models | Are these comparisons sufficient? Is the supplementary 2D model derived from the same 3D cohort acceptable? |
| GPU VRAM, parameters, and inference time | All ten models have reference-case VRAM and Test inference timing; v5 additionally has training VRAM/time logs. Benchmarks use the same input/protocol across separate jobs | Is original-resolution or end-to-end timing including preprocessing also required? |
| 2D/3D context, base/apex, continuity, and failure analysis | Quantitative measurements, five earlier real-image cases, and four shared v5 Validation cases reviewed | Do the geometric proxies and image interpretations adequately address the assessment's clinical-boundary dilemma? |
| Reproducibility and documentation | Configurations, source identities, saving/recovery, figures, and AI disclosure available | Final training/prediction entry points and accessible images for assessors still need to be organised |
| Feasibility check-off, correct PR, and response to review | This report does not claim these are complete | Please confirm the midpoint record, submission branch, and subsequent review requirements |

**My provisional assessment:** there is a working Hard candidate and substantial research evidence, but “the average exceeds the threshold” should not be stated as “all assessment requirements are satisfied.” Clearly outperforming Standard remains my research objective; current evidence shows conditional average gains and identifiable local trade-offs.

## 11. Questions for Guidance and Proposed Next Steps

| Priority | Question I Would Like to Discuss | Provisional Approach, Subject to Guidance |
|---|---|---|
| 1 | Is the research depth sufficient now that all ten models have Validation/Test, boundary, and resource evaluation? Should I stop adding models and prioritise final delivery? | Retain v3 as the existing main candidate and Standard as the baseline; report that L/S underperform concurrent C on Validation and existing Test, without repeatedly optimising recipes against Test |
| 2 | How should the assessment's Dice target and aggregation be interpreted? Does Hard require Improved to exceed the baseline on every metric? | Report Prostate, three-organ, five-foreground, and per-class results explicitly, rather than choosing the most favourable average |
| 3 | How should repeated Test inspection be disclosed? Is an additional confirmation experiment needed? | Do not tune on Test; if stability evidence is needed, consider patient-level cross-validation within Train or a limited number of repeated seeds, subject to resource limits |
| 4 | Should the next priority be average Dice, extreme boundary errors, the base, or organ confusion? | Four shared v5 Validation cases have been reviewed; if continuing, locate the full-volume errors behind S022_Week6 Rectum's 175.48 mm and J026_Week6 Prostate's 112.51 mm HD95 before specifying a single-variable experiment |
| 5 | L lowers Bladder/Rectum/Prostate CE weights from 4.498586/8.893215/9.699950 to 2.120987/2.982149/3.114474; Bladder benefits but Rectum and Prostate precision fall. Is an organ-specific weighting experiment justified? | First distinguish distant false positives from near-boundary expansion on fixed Validation cases; if training continues, change one class weight and pair it with concurrent C and a second seed |
| 6 | S changes only Adam's fixed 0.001 learning rate to per-update cosine decay ending at 0.00001, but loses 0.007745 in mean small-organ Dice. Is a different decay timing worth testing? | Do not generalise one run to all schedules; if continuing, prespecify the schedule, matched C and training budget, and repeated seed without changing optimiser or weight decay |
| 7 | Should checkpoint selection be redesigned to constrain extreme boundary errors? | Predeclare the primary endpoint and boundary constraints before a new experiment; do not change the rule after inspecting Test |
| 8 | Can we learn from the more restrained localisation/precision behaviour of 2D in W029? | First analyse similar errors in Validation; do not design a new network directly around one Test example |

**The specific guidance I am seeking:** whether the project should now converge on a final submission and, if further research is justified, which **one** controlled experiment is most worthwhile, rather than adding training strategies indiscriminately.

---

### Data and Supporting Material

The earlier seven-model Validation/Test evaluation covers 476 model/volume comparisons; its 37 downloaded files passed SHA verification, and the Validation matrices and historical Standard/v1 Test matrices were checked. Sections 4–5 now contain complete ten-model Validation/Test results; Sections 7.1 and 8 include v5 training and inference measurements. The 156 training JSON files match remote aggregate hashes. Test job 682322 completed successfully. The ten-model selected-checkpoint comparison contains 680 model/split/volume entries: the earlier 476 plus 114 selected v5 Validation entries and 90 v5 Test entries; other candidate epochs during training, additional Validation replays, and benchmark repetitions are excluded. The Test result SHA-256 is `454a8080d1ecff3fb0bfe0d675af420cad5a26f69f236458e0036ed56e511986`, matching the server; input case hashes match the earlier audit and checkpoint hashes match the previous Validation replay. Formal evaluation, development diagnostics, and unproven mechanisms are reported separately.

- [Full research report and links to real case images (Chinese)](../experiments/improved_3d_v4/reports/RESEARCH_REPORT_ZH.md)
- [Seven-model Validation/Test Dice and IoU for all six classes](../experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/six_class_validation_test.svg)
- [Prostate end-region measurements](../experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/prostate_base_apex.svg)
- [Complete numerical tables](../experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/)
- [v5 C/L/S Validation analysis and four-case review (Chinese)](../experiments/improved_3d_v5/reports/VALIDATION_SCREEN_ZH.md)

This report supports a research progress discussion. It does not guarantee clinical safety, dosimetric benefit, external generalisation, or assessment marks. ChatGPT/Codex assisted with code, statistics, and writing. Automated checks and review of specified real images were performed; these do not constitute independent clinical expert assessment.

- [Latest ten-model figures and numerical tables](model_performance_comparison/README.md)
