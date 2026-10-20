# HipMRI Segmentation: Research Progress and Questions for Guidance

**Purpose: Discuss current progress, assessment requirements, and the next research direction with my supervisor or teaching assistant**

**Report date: 21 October 2026**

## 1. What I Hope to Clarify

I have completed seven formal model training runs, a common evaluation, and an analysis of real segmentation errors. I would appreciate guidance on the following: **Does the current work satisfy the assessment's implementation and research requirements? Should I now prioritise the final deliverables and interpretation, or continue training? If further experiments are warranted, which change would provide the greatest research value?**

My current interpretation is:

- **Improved v3 is the leading overall candidate, rather than the latest model, A1.** It performs best on the predefined Validation selection metric and also has the highest mean small-organ Dice on the existing Test cohort.
- **The goal of clearly outperforming Standard in every respect has not been achieved.** v3 still has extreme boundary errors and performs worse than Standard for some patients and individual metrics. A1 reduces distant Body false positives but weakens overall segmentation performance.
- **Meeting an average Test score target does not establish reliable local contours.** The prostate base and apex, confusion between organs, and individual difficult patients remain important problems.
- The main accuracy, boundary, and resource measurements are complete, but **I would still like guidance on final assessment compliance, the intended metric aggregation, and whether the generalisation evidence is sufficient**.

## 2. Data and Common Evaluation Method

### 2.1 Dataset

| Item | Train | Validation | Test |
|---|---:|---:|---:|
| Patients | 26 | 6 | 6 |
| MRI volumes | 143 | 38 | 30 |
| Purpose | Parameter optimisation and training sampling | Checkpoint selection and problem diagnosis | Descriptive evaluation of fixed models |

The six classes are **Background, Body, Bones, Bladder, Rectum, and Prostate**. Splitting is performed at patient level: different visits from the same patient do not cross splits.

### 2.2 Conditions Shared Across Models

| Condition | Implementation |
|---|---|
| Preprocessing | MRI and masks share the same grid; downsampling by a factor of 2 along each axis; linear interpolation for MRI and nearest-neighbour interpolation for masks; per-volume Z-score normalisation using nonzero MRI voxels |
| Training budget | All seven runs completed 50 epochs; 130 batches per epoch, batch size=2, giving 6500 parameter updates per run |
| Patient sampling | One visit per Train patient per epoch, cycling through visits according to a predefined rule; 10 training samples per patient |
| Model input | 3D patches: 64×64×32; 2D crops: 64×64 |
| Validation and selection | Validation every 5 epochs; select the best at epochs 5, 10, …, 50, giving 10 measured candidates; choose the earlier epoch in a tie |
| Primary selection metric | Mean Dice for Bladder, Rectum, and Prostate, with equal weighting of the six Validation patients |
| Patient aggregation | Pool confusion matrices across each patient's visits, calculate metrics, then average across patients; the 30 Test volumes are not treated as 30 independent patients |
| Saving and reproducibility | Retain epoch weights, training records, the best checkpoint, and recovery state; preserve configuration and source identities, including interruption and recovery records |

**Clarification:** “50 training epochs” does not mean 50 batches. The Test cohort has already been inspected during the research process, so this evaluation cannot be described as a new blind test. All checkpoints were selected using Validation, rather than choosing the best epoch on Test.

## 3. What Changed Across the Seven Formal Runs?

### 3.1 Architecture and Supervision

| Model | Architecture and width | Normalisation | Residual connections / Dropout | Output and supervision | Main research purpose |
|---|---|---|---|---|---|
| 2D U-Net | Standard 2D U-Net; base channels=16 | GroupNorm (GN), 1 group | No residual connections; no explicit Dropout | Single output head | Supplementary slice-wise comparison with low VRAM use |
| Standard 3D | Standard 3D U-Net; base channels=8 | GN, 1 group | No residual connections; no explicit Dropout | Single output head | Establish the 3D baseline |
| Improved v1 | Residual 3D U-Net; base channels=8 | InstanceNorm | Residual connections; Dropout=0.3 | Sum the main logits and upsampled half- and quarter-scale logits; calculate loss on the sum | Introduce residual architecture and multiscale supervision |
| Improved v2 | Retains the v1 backbone | InstanceNorm | Same as v1 | Separate losses for three heads, weighted by 1, 0.5, and 0.25 and divided by 1.75; main head only at inference | Investigate supervision and output fusion |
| Improved v3 | Modified from v1; **not from v2** | Replace 14 IN sites with GN, 1 group | Same as v1 | Retains the v1 additive-logit scheme | Investigate normalisation as a single architectural change |
| v4 A0 | Same as v3 | GN, 1 group | Same as v3 | Same as v3 | Evaluate a new optimisation recipe without spatial augmentation |
| v4 A1 | Same as v3 | GN, 1 group | Same as v3 | Same as v3 | Extend the A0 recipe with paired 3D spatial augmentation |

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

The base loss for all formal runs is **0.5 weighted cross-entropy + 0.5 foreground Soft Dice**. v2 applies this loss separately to each head. Static class weights are derived from Train data, but differ between the 2D and 3D recipes:

| Weight order | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| Six 3D models | 1.453871 | 1.000000 | 2.617199 | 4.498586 | 8.893215 | 9.699950 |
| Supplementary 2D model | 1.517047 | 1.000000 | 2.520386 | 4.124822 | 7.521774 | 8.819274 |

In A1, MRI and mask receive exactly the same spatial transformation, with nearest-neighbour interpolation for the mask. No flips, strong elastic deformation, or intensity augmentation were used. The augmentation audit across 26 Train patients passed before training.

**Comparison limitations:** A1−A0 primarily measures spatial augmentation within this recipe. A0−v3 simultaneously changes Adam to AdamW, weight decay, and the learning-rate schedule, so it does not isolate whether learning-rate decay is beneficial. The 2D and 3D recipes also differ in sampling, channel width, and class weights; their performance differences cannot all be attributed to spatial dimensionality.

## 4. Main Results: Each Model's Best Validation Checkpoint

### 4.1 Overall Results

All Dice values below give equal weight to patients; higher is better. “Small organs” refers to Bladder, Rectum, and Prostate.

| Model | Best epoch | Validation small-organ Dice | Test small-organ Dice | Test Prostate Dice | Test mean Dice over five foreground classes |
|---|---:|---:|---:|---:|---:|
| 2D | 50 | 0.7757 | 0.7082 | 0.6611 | 0.7883 |
| Standard | 50 | 0.8502 | 0.7851 | 0.7433 | 0.8370 |
| v1 | 50 | 0.8420 | 0.8078 | **0.7520** | 0.8502 |
| v2 | 45 | 0.8362 | 0.7959 | 0.7478 | 0.8422 |
| v3 | 50 | **0.8564** | **0.8145** | 0.7485 | **0.8631** |
| A0 | 45 | 0.8450 | 0.7836 | 0.7318 | 0.8400 |
| A1 | 45 | 0.8298 | 0.7801 | 0.7273 | 0.8362 |

v3 leads on the principal averages, but v1 has the highest Test Prostate Dice. No model currently dominates all metrics.

### 4.2 Test Dice for All Six Classes

| Model | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.9896 | 0.9681 | 0.8487 | 0.7508 | 0.7127 | 0.6611 |
| Standard | 0.9904 | 0.9677 | 0.8619 | 0.7993 | 0.8128 | 0.7433 |
| v1 | 0.9683 | 0.9470 | 0.8808 | 0.8428 | 0.8285 | 0.7520 |
| v2 | 0.9783 | 0.9565 | 0.8668 | 0.8484 | 0.7916 | 0.7478 |
| v3 | 0.9885 | **0.9732** | **0.8988** | **0.8539** | **0.8410** | 0.7485 |
| A0 | 0.9892 | 0.9704 | 0.8786 | 0.8002 | 0.8189 | 0.7318 |
| A1 | 0.9890 | 0.9686 | 0.8721 | 0.8236 | 0.7894 | 0.7273 |

### 4.3 Test IoU for All Six Classes

| Model | Background | Body | Bones | Bladder | Rectum | Prostate |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.9799 | 0.9388 | 0.7380 | 0.6668 | 0.5552 | 0.5080 |
| Standard | 0.9814 | 0.9384 | 0.7578 | 0.7455 | 0.6918 | 0.6081 |
| v1 | 0.9414 | 0.9018 | 0.7902 | 0.7734 | 0.7089 | 0.6183 |
| v2 | 0.9587 | 0.9177 | 0.7664 | 0.7721 | 0.6568 | 0.6114 |
| v3 | 0.9778 | **0.9484** | **0.8168** | **0.7768** | **0.7280** | **0.6193** |
| A0 | 0.9791 | 0.9431 | 0.7841 | 0.7289 | 0.6971 | 0.5969 |
| A1 | 0.9786 | 0.9393 | 0.7736 | 0.7412 | 0.6573 | 0.5916 |

Dice and IoU are calculated separately before averaging across patients. Mean IoU is not obtained by transforming the mean Dice shown in the table.

## 5. Boundaries and Local Risks Beyond Average Dice

### 5.1 Test Prostate End Regions and Surface Errors

HD95 is measured in millimetres, with lower values indicating better performance. The table reports medians and maxima across volumes. Regional Dice is a patient-equal average of slice-region scores and differs from whole-volume Dice.

| Model | Prostate apex Dice | Central Dice | Base Dice | Median Body HD95 / mm | Median Prostate HD95 / mm | Maximum Prostate HD95 / mm |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 0.5186 | 0.7125 | 0.3473 | 6.055 | 9.700 | 15.600 |
| Standard | **0.7544** | 0.7824 | 0.5097 | 4.861 | 7.979 | 19.229 |
| v1 | 0.7397 | **0.7997** | **0.5993** | 51.813 | 6.719 | 122.347 |
| v2 | 0.7120 | 0.7877 | 0.5836 | 28.555 | 7.460 | 16.717 |
| v3 | 0.7523 | 0.7931 | 0.5069 | **4.500** | **6.645** | 115.371 |
| A0 | 0.7169 | 0.7687 | 0.4839 | 4.751 | 8.611 | 23.712 |
| A1 | 0.7251 | 0.7620 | 0.5472 | 5.684 | 9.265 | 22.211 |

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

Distant Body false positives are restricted to true Background voxels predicted as Body and at least 20 mm from the true Body. The organ-at-risk (OAR) percentage uses predicted Prostate voxels as its denominator; it measures class confusion, not radiation dose or probability of organ injury. Components use 6-connectivity, and their count does not directly establish anatomical correctness.

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

### 7.1 Resource Measurements on the Same A100 40 GB GPU

| Model | Parameters | Reference-volume peak allocated VRAM / GiB | Median single-pass inference per Test volume / s |
|---|---:|---:|---:|
| 2D | 1,942,374 | 0.076 | 0.1401 |
| Standard | 1,412,550 | 0.246 | 0.1740 |
| v1 | 2,366,946 | 0.346 | 0.1144 |
| v2 | 2,366,946 | 0.346 | 0.1073 |
| v3 | 2,366,946 | 0.346 | 0.1666 |
| A0 | 2,366,946 | 0.346 | 0.1666 |
| A1 | 2,366,946 | 0.346 | 0.1666 |

The reference volume uses 3 warm-ups and 5 synchronised repetitions; single-pass inference is also recorded for every volume. Timing excludes disk access, preprocessing, and metric calculation. The networks are small and inputs are downsampled, so these costs should not be extrapolated to large models at original resolution. v4 was interrupted by full storage and then resumed; its total training wall-clock time is unsuitable as a direct comparison of pure GPU training speed.

### 7.2 What Can Currently Be Said About Generalisation?

| Evidence | Observation | Limitation |
|---|---|---|
| Patient-level Validation comparison | v3 improves over Standard by 0.00613; 5/6 patients improve | Only 6 patients, with repeated reuse of Validation during research |
| Patient-level Test comparison | v3 improves over Standard by 0.02938, but only 3/6 patients improve | Not every patient benefits |
| M015 sensitivity analysis | Removing M015 for a descriptive sensitivity check changes v3−Standard to −0.00496 | This is not permission to remove the patient from the official score; the advantage is sensitive to cohort composition |
| A1 versus v3 | A1 has lower Test small-organ scores for 6/6 patients | The results do not support stronger generalisation for A1 |
| Replication | One fixed seed per recipe | Stability across seeds is untested; there is no external cohort or new blind test |

## 8. Development Diagnostics: Has Class Imbalance Been Addressed?

In addition to the seven formal experiments, development included learning diagnostics on two fixed patches from the same training case:

| Fixed-sample diagnostic | Updates | Training Rectum Dice | Training Prostate Dice | Supported interpretation |
|---|---:|---:|---:|---|
| Standard, without class weights in the CE term | 100 | 0.0000 | 0.0000 | The two small organs were not learned under this setting |
| Standard, without class weights, diagnostic budget increased to 500 steps | 500 | 0.0000 | 0.0000 | Additional updates alone did not resolve this fixed-sample problem |
| Standard, with Train-derived class weights in CE | 500 | 0.9796 | 0.9722 | The network can fit the small organs, and weighting has a measurable effect |
| Improved v1, without the above class weighting | 100 | 0.9586 | 0.9393 | Faster fitting on this fixed sample does not establish better generalisation |

These are training scores on the same fixed batch, not Validation or Test results, and they are not four additional formal generalisation models. Subsequent formal training consistently used weighted CE plus foreground Dice; 2D additionally used organ-aware sampling. Class imbalance has therefore been addressed, but whether the current weights cause an unfavourable precision–recall trade-off has not been established through an independent loss ablation.

## 9. Does the Work Currently Meet the Assessment Requirements?

The following is a provisional interpretation of the course specification draft currently available to me. **I would appreciate confirmation from my supervisor or teaching assistant.**

| Assessment requirement | Current status | Clarification needed |
|---|---|---|
| Hard: 3D residual units, deep supervision, and downsampled HipMRI | Implementation, training, and evaluation evidence available | Does the adapted architecture satisfy the Hard implementation requirement? |
| Test Dice ≥0.7 | All six 3D models have patient-equal mean Prostate Dice ≥0.7 | Does the target apply to Prostate, five foreground classes, all classes, or another aggregate? Must every patient meet it? |
| Standard baseline and per-class Test Dice/IoU on identical splits | Completed for seven models | Are these comparisons sufficient? Is the supplementary 2D model derived from the same 3D cohort acceptable? |
| GPU VRAM, parameters, and inference time | Common measurements completed | Is original-resolution or end-to-end timing including preprocessing also required? |
| 2D/3D context, base/apex, continuity, and failure analysis | Quantitative measurements and five real failure analyses available | Do the geometric proxies and image interpretations adequately address the assessment's clinical-boundary dilemma? |
| Reproducibility and documentation | Configurations, source identities, saving/recovery, figures, and AI disclosure available | Final training/prediction entry points and accessible images for assessors still need to be organised |
| Feasibility check-off, correct PR, and response to review | This report does not claim these are complete | Please confirm the midpoint record, submission branch, and subsequent review requirements |

**My provisional assessment:** there is a working Hard candidate and substantial research evidence, but “the average exceeds the threshold” should not be stated as “all assessment requirements are satisfied.” Clearly outperforming Standard remains my research objective; current evidence shows conditional average gains and identifiable local trade-offs.

## 10. Questions for Guidance and Proposed Next Steps

| Priority | Question I Would Like to Discuss | Provisional Approach, Subject to Guidance |
|---|---|---|
| 1 | Is the research depth sufficient given the seven models, boundary analysis, and resource measurements? Should I stop adding models and prioritise final delivery? | Retain v3 as the Validation-selected main candidate and Standard as the baseline; include negative results in the analysis |
| 2 | How should the assessment's Dice target and aggregation be interpreted? Does Hard require Improved to exceed the baseline on every metric? | Report Prostate, three-organ, five-foreground, and per-class results explicitly, rather than choosing the most favourable average |
| 3 | How should repeated Test inspection be disclosed? Is an additional confirmation experiment needed? | Do not tune on Test; if stability evidence is needed, consider patient-level cross-validation within Train or a limited number of repeated seeds, subject to resource limits |
| 4 | Should the next priority be average Dice, extreme boundary errors, the base, or organ confusion? | First review v3's extreme fixed-Validation cases to distinguish distant isolated predictions, localisation errors, and boundary expansion; then propose a single-variable experiment |
| 5 | Are weighted CE plus Dice and the current class weights appropriate? Would an ablation of weight strength or loss composition be worthwhile? | If Validation supports an oversegmentation mechanism, change one loss factor and explicitly track precision, recall, and boundary gains and regressions |
| 6 | How should learning-rate decay and spatial augmentation be tested fairly? | A0 does not isolate the scheduler; a further experiment could add only a scheduler to v3, or test augmentation alone with a small budget, rather than combining changes |
| 7 | Should checkpoint selection be redesigned to constrain extreme boundary errors? | Predeclare the primary endpoint and boundary constraints before a new experiment; do not change the rule after inspecting Test |
| 8 | Can we learn from the more restrained localisation/precision behaviour of 2D in W029? | First analyse similar errors in Validation; do not design a new network directly around one Test example |

**The specific guidance I am seeking:** whether the project should now converge on a final submission and, if further research is justified, which **one** controlled experiment is most worthwhile, rather than adding training strategies indiscriminately.

---

### Data and Supporting Material

All formal results in this report come from the completed common seven-model evaluation, covering 476 model/volume comparisons. The 37 downloaded files passed SHA verification, and the Validation matrices and historical Standard/v1 Test matrices were checked. Formal evaluation, development diagnostics, and unproven mechanisms are reported separately.

- [Full research report and links to real case images (Chinese)](../experiments/improved_3d_v4/reports/RESEARCH_REPORT_ZH.md)
- [Seven-model Validation/Test Dice and IoU for all six classes](../experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/six_class_validation_test.svg)
- [Prostate end-region measurements](../experiments/improved_3d_v4/assessment_analysis/assessment_required/figures/prostate_base_apex.svg)
- [Complete numerical tables](../experiments/improved_3d_v4/assessment_analysis/assessment_required/tables/)

This report supports a research progress discussion. It does not guarantee clinical safety, dosimetric benefit, external generalisation, or assessment marks. ChatGPT/Codex assisted with code, statistics, and writing. Automated checks and review of specified real images were performed; these do not constitute independent clinical expert assessment.
