# HipMRI 3D U-Net Final Report

The [unified experiment history](docs/experiments.md) records Baseline through V4 in a common protocol/results/decision format. This course-facing report interprets the E10 comparison, failures and V4 decision; detailed source mappings reside in [traceability](docs/evidence/traceability.md).

## 1. Dataset and Preprocessing

### Dataset

This project uses the HipMRI dataset for multi-class semantic segmentation of pelvic structures in three-dimensional magnetic resonance imaging (MRI) volumes. Each MRI volume is paired with a corresponding segmentation mask containing six semantic classes: Background (0), Body (1), Bones (2), Bladder (3), Rectum (4), and Prostate (5).

The dataset consists of 211 paired MRI volumes and segmentation masks collected from 38 patients.

### Data Splitting

To minimise patient-level data leakage, the dataset was divided into mutually exclusive training, validation, and test subsets at the patient level. All volumes belonging to a given patient were assigned to the same subset.

| Dataset split | Patients | MRI volumes |
|---|---:|---:|
| Training | 26 | 143 |
| Validation | 6 | 38 |
| Test | 6 | 30 |
| Total | 38 | 211 |

The training set was used for model optimisation, while the validation set was used to assess segmentation performance during model development. The test set was reserved for final evaluation and was not used to select training configurations.

### Image Preprocessing

MRI volumes and their segmentation masks were resampled to a common voxel spacing of approximately 1.68 × 1.68 × 1.56 mm.

Linear interpolation was used for MRI intensities, while nearest-neighbour interpolation was applied to segmentation masks to preserve discrete class labels.

For intensity normalisation, nonzero MRI intensities were clipped at the 0.5th and 99.5th percentiles and subsequently standardised using z-score normalisation.

### Patch-Based Training

Both models were trained using three-dimensional patches of size 64 × 64 × 64 voxels.

Training sampled one patch from each training volume per epoch, resulting in 143 training steps per epoch with a batch size of one.

To increase exposure to the target organ, the patch sampling procedure incorporated prostate-focused foreground sampling with a configured probability of 0.5.

The same dataset split and preprocessing pipeline were used for the Standard 3D U-Net and Improved 3D U-Net to support a controlled architectural comparison.



## 2. Model Architecture

### Standard 3D U-Net Baseline

A Standard 3D U-Net was implemented as the baseline for six-class volumetric segmentation of pelvic MRI data. The network follows an encoder–decoder architecture with skip connections, adapting the principles of 3D U-Net proposed by Çiçek et al. (2016).

The model accepts a single-channel three-dimensional MRI patch with shape `[B, 1, 64, 64, 64]`. Its encoder contains four downsampling stages, with feature widths of 16, 32, 64, and 128 channels. Each convolutional block applies two 3D convolutions, Group Normalisation, and ReLU activation. A 256-channel bottleneck captures deeper volumetric features.

The decoder progressively restores spatial resolution using transposed 3D convolutions. Encoder feature maps are concatenated with the corresponding decoder features through skip connections, preserving spatial information that may otherwise be lost during downsampling.

A final 1 × 1 × 1 convolution produces six-channel segmentation logits with shape `[B, 6, 64, 64, 64]`. The model contains **5,646,470 trainable parameters**.

### Improved 3D U-Net

The Improved 3D U-Net extends the baseline with volumetric residual units and deep supervision, following the architectural principles referenced by Isensee et al. (2018).

**Residual units** introduce additive feature connections within the network, allowing transformed features to be combined with shortcut information. These connections provide alternative paths for gradient propagation and are intended to support the optimisation of a deeper feature hierarchy.

**Deep supervision** introduces auxiliary segmentation predictions at lower spatial resolutions. In addition to the full-resolution output of `[B, 6, 64, 64, 64]`, the implemented model produces auxiliary outputs at spatial resolutions of 32 × 32 × 32 and 16 × 16 × 16 voxels.

During training, losses from the main output and two auxiliary outputs are combined with relative weights of 1.0, 0.5, and 0.25, respectively, and normalised by their sum (1.75). This provides segmentation supervision at multiple decoder resolutions rather than relying exclusively on the final prediction.

The Improved model contains approximately **5.73 million trainable parameters**.

### Architectural Comparison and Experimental Hypothesis

Both models use the same six-class segmentation task, patient-level dataset split, spatial preprocessing, training patch dimensions, and primary evaluation metrics.

The primary experimental hypothesis is that combining volumetric residual feature learning with deep supervision will improve segmentation of anatomically challenging structures, particularly the prostate and adjacent organs at risk, compared with the Standard 3D U-Net.

The comparison investigates whether these architectural modifications provide measurable gains in segmentation accuracy and boundary quality that justify their additional computational cost.

Because residual units and deep supervision are introduced together, this comparison evaluates their combined effect rather than isolating the individual contribution of either component.

The architectures are adaptations of the cited methods rather than exact reproductions of their original experimental configurations.

## 3. Experimental Setup and Training Configuration

### Training Protocol

Both the Standard 3D U-Net and Improved 3D U-Net were trained using the same frozen patient-level training split, consisting of 143 MRI volumes from 26 patients.

Training was performed using three-dimensional patches of size 64 × 64 × 64 voxels, with a batch size of one. One training patch was sampled per volume in each epoch, resulting in 143 optimisation steps per epoch.

Both models completed 10 training epochs, corresponding to 1,430 optimiser updates per model.

The downloaded original training logs for Rangpur jobs **648790** (Standard) and **649668** (Improved) end at ten completed epochs / 1,430 updates. They resume from three and one completed epochs respectively: this audit directly verifies 1,001 Standard updates in epochs 4–10 and 1,287 Improved updates in epochs 2–10, together with the recorded resume/final counters. Earlier epoch execution is not reconstructed from these files. Recorded source hashes match the current implementation, and the 143 ordered training pairs and preprocessing policies match between models. These are matched update budgets, not matched compute budgets. Checkpoint binaries remain unavailable locally.

The models were trained using the AdamW optimiser with an initial learning rate of 0.001 and a weight decay of 0.00001. Training was conducted in float32 precision using NVIDIA A100 GPUs on the UQ Rangpur high-performance computing cluster.

The training configuration used a fixed random seed of 3710 and a reproducible patient-level data split. Model checkpoints were saved at completed epoch boundaries, allowing training to resume without intentionally restarting the optimisation process.

### Loss Function

The Standard 3D U-Net was trained using a combined Cross-Entropy and Soft Dice objective, with equal weighting:

L_standard = 0.5 × L_CE + 0.5 × L_Dice

Cross-Entropy provides voxel-level classification supervision, whereas Soft Dice promotes overlap between predicted segmentation probabilities and ground-truth regions. The Soft Dice component was computed over foreground classes 1–5.

The Improved 3D U-Net used the same underlying segmentation loss at the main and auxiliary output resolutions, with an additional deep supervision objective.

The three output losses were combined using relative weights of 1.0, 0.5, and 0.25 and normalised by their sum of 1.75.

Deep supervision was introduced to encourage segmentation-relevant feature learning at multiple spatial resolutions and improve gradient propagation throughout the network.

### Sampling Strategy

Training patches were generated through an on-demand preprocessing pipeline. Each sampled patch preserved spatial alignment between the MRI image and its segmentation mask.

A prostate-focused sampling strategy was used, with a configured foreground-sampling probability of 0.5. When a prostate-focused sample was selected and prostate voxels were available, the crop was constructed to contain the target label. Otherwise, the sampling procedure used a random crop.

This strategy was intended to increase the exposure of the models to a clinically important small anatomical structure while retaining variation in the training patches.

### Recorded Training-Loss Trajectory

![Recorded CE, foreground Soft Dice and total training losses, with missing epochs shaded](figures/e10_recorded_training_losses.png)

**Figure 3.1.** Only genuine epoch events are plotted: Standard 4–10 and Improved 2–10, each with 143 verified step records. Standard epochs 1–3 and Improved epoch 1 require earlier logs and are not estimated. [Exact recorded values and timings](docs/evidence/failure-cases.md#recorded-training-loss-trajectory) accompany the plot. Standard total loss falls from 0.791711 to 0.699227 over its recorded segment; Improved falls from 0.810483 to 0.611152. Both trajectories are non-monotonic, and both Soft Dice losses rise from epoch 9 to 10. Improved components include weighted auxiliary heads, whereas Standard uses its main head only; neither their absolute difference nor declining within-model losses establishes class-specific convergence or generalisation.

### Experimental Control

The two models were compared using the same training dataset, validation patients, patch dimensions, total number of optimisation steps, and principal evaluation metrics.

The main experimental difference was the introduction of residual units and deep supervision in the Improved 3D U-Net.

Both architectures were evaluated using full-volume segmentation, rather than estimating performance exclusively from sampled training patches.

The controlled comparison was designed to examine whether the additional architectural complexity improved segmentation accuracy for small and anatomically challenging foreground structures.

However, the comparison evaluates the combined effect of residual units and deep supervision, rather than isolating their individual contributions.

### Evaluation Protocol

The validation set contained 38 MRI volumes from six patients who were excluded from the training set.

Segmentation performance was assessed using per-class Dice Similarity Coefficient (DSC) and Intersection-over-Union (IoU). Prediction-class coverage was also inspected to detect cases where a model failed to predict particular anatomical structures.

The held-out test set contains 30 MRI volumes from six additional patients and remains reserved for final evaluation.

The 10-epoch training budget was an initial controlled experimental configuration, not a claim of optimal convergence. Model convergence and task success must be assessed from validation behaviour and the final held-out test results.


## 4. Baseline Results and Failure Analysis

### Standard 3D U-Net Validation Results

The Standard 3D U-Net was evaluated on the held-out validation split containing 38 MRI volumes from six patients. Full-volume inference was performed using a sliding-window approach. Epoch 10 segmentation quality is reported using per-class pooled Dice Similarity Coefficient (DSC) and Intersection-over-Union (IoU); the original Epoch 3 aggregation transcript is unavailable locally.

Two training checkpoints were evaluated to examine changes in segmentation performance: Epoch 3 (429 optimisation steps) and Epoch 10 (1,430 optimisation steps).

| Anatomical Class | Epoch 3 DSC | Epoch 10 DSC | Epoch 10 IoU |
|---|---:|---:|---:|
| Background | 0.9220 | 0.9413 | 0.8892 |
| Body | 0.8072 | 0.8428 | 0.7283 |
| Bones | 0.0000 | 0.0000 | 0.0000 |
| Bladder | 0.0000 | 0.6220 | 0.4514 |
| Rectum | 0.0000 | 0.0000 | 0.0000 |
| Prostate | 0.0000 | 0.0000 | 0.0000 |

All 38 validation volumes were evaluated successfully at Epoch 10, with no failed inference cases.

The Epoch 10 results are `overall_pooled` metrics recomputed locally from all 38 per-volume confusion matrices in the downloaded original job **649669** JSONL, agreeing with its summary. Epoch 3 values retain their historical student-recorded provenance; its original aggregation/setup transcript remains unavailable.

### Effect of Additional Training

Increasing the training duration from three to ten epochs improved the pooled segmentation scores for several classes.

Background DSC increased from 0.9220 to 0.9413, while Body DSC improved from 0.8072 to 0.8428. The most substantial improvement was observed for the Bladder class, whose DSC increased from 0.0000 to 0.6220.

These results demonstrate that additional optimisation steps improved the model's ability to distinguish certain anatomical regions. However, the improvement was not uniform across all foreground classes.

### Class Prediction Collapse

Despite the improvements, the Epoch 10 model exhibited a significant limitation in multi-class segmentation.

An inspection of predicted label frequencies across all 38 validation volumes revealed that the model generated only Background, Body, and Bladder labels. No voxels were assigned to Bones, Rectum, or Prostate in any validation volume.

This observation provides direct evidence of partial class prediction collapse: the model systematically omitted three anatomical categories from its predictions.

The failure cannot be explained solely as inaccurate localisation of those classes, because their predicted voxel counts were exactly zero.

Although the model learned to segment several larger or more distinguishable regions, it failed to produce a complete six-class anatomical segmentation.

### Potential Contributing Factors

Several factors may explain the observed failure, although the current experiments do not establish their individual contributions.

First, class imbalance may cause larger or more frequently observed anatomical regions to dominate the optimisation process.

Second, reconstructed patch sampling shows unequal voxel shares, but literal missing-target exposure is not supported for Bones: job 650736 records Bones in 1,230/1,430 patches (86.01%) with 9.10% pooled voxel share and 10.08% median occupancy in positive patches. Prostate occurs in 880 patches (61.54%) but has only 1.43% pooled share. These are authenticated configuration-replay measurements, not independent proof of unchanged historical input image bytes.

Third, the combined Cross-Entropy and foreground Soft Dice objective may not provide sufficient corrective pressure for the omitted classes under the current sampling distribution.

Finally, the ten-epoch training budget may be insufficient for the network to learn all anatomical structures reliably.

These explanations remain hypotheses. Source/synthetic loss checks and reconstructed Train-patch counts support the measurement methods. The subsequent student-run Job 650924 measured E10 checkpoint probabilities on two fixed Train crops and one Validation center crop; its downloaded aggregate records were authenticated locally, as described in [Section 5.4](#54-cross-case-failure-patterns). These selected-patch measurements show probability suppression and class competition, not their causes or full-volume performance. Per-class parameter-gradient strength and causal attribution remain unmeasured; any bounded Train-only gradient audit requires separate approval before retraining is considered.

### Critical Evaluation

The Standard 3D U-Net provides a functional baseline and demonstrates measurable improvement between Epochs 3 and 10. However, the validation results reveal that successful optimisation and improved aggregate segmentation quality do not necessarily imply successful segmentation of every target structure.

In particular, the absence of Prostate predictions represents a major limitation for the target application.

The results also highlight the importance of reporting per-class metrics and predicted label distributions rather than relying exclusively on a single aggregate score.

The current findings are restricted to validation performance. They do not establish the model's final held-out test performance, clinical reliability, or convergence.

Improved Epoch 10 full validation subsequently completed in job **650393** (38 successful, zero failed; reported elapsed time 8m04s). The original job **649912** was cancelled and replaced by this successful run. Section 5 compares the completed Validation results and authentic paired-case evidence; final Test evaluation remains pending.

## 5. Controlled Validation Comparison and Failure Case Evidence

### 5.1 Controlled Validation Comparison

Both architectures completed ten epochs and 1,430 optimiser updates using the same 143 training volumes, frozen patient split, preprocessing, 64³ patches and sampling policy. Standard full Validation completed in job **649669**; Improved full Validation completed in **650393**, replacing cancelled job **649912**. Each successful evaluation covered the same 38 volumes from six validation patients. These results are Validation evidence, not final held-out Test performance.

The following table is recomputed from the downloaded original **649669 / 650393** per-volume confusion matrices and agrees with both `overall_pooled` summaries. Counts are pooled across all 38 volumes before computing class overlap; this is not an equal-patient or equal-volume macro average. Six-decimal entries agree with the previous Section 5.1 table; scientific notation now distinguishes negligible positive Improved Bones overlap from exact zero. [The evidence manifest](docs/evidence/failure-cases.md#full-validation-numerical-provenance) records full-precision scores, pooled counts and exact prediction coverage. This is local log authentication and arithmetic verification, not a rerun of Rangpur inference.

| Class | Standard DSC | Improved DSC | Standard IoU | Improved IoU |
| --- | ---: | ---: | ---: | ---: |
| Background | 0.941342 | 0.976012 | 0.889184 | 0.953148 |
| Body | 0.842790 | 0.906781 | 0.728295 | 0.829460 |
| Bones | 0.000000 | 3.943110e-7* | 0.000000 | 1.971555e-7* |
| Bladder | 0.622004 | 0.768230 | 0.451383 | 0.623680 |
| Rectum | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Prostate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

*Improved Bones overlap is positive: **two true-positive and one false-positive voxel**, against 10,144,275 reference voxels. Predictions occur only in K018_Week1 (one voxel) and K018_Week4 (two voxels). Both scores round to 0.000000 at six decimals; this does not constitute useful Bones segmentation.

Dice is `2TP/(Ntrue+Npred)`; IoU is `TP/(Ntrue+Npred−TP)`. Both-absent classes are null and excluded from macro averages; one-sided absence scores zero. All per-volume, pooled, patient and macro summaries agree with recomputation. Background, Body and Bladder improve, with Bladder DSC increasing by approximately **0.1462**. Nevertheless, both models predict no Rectum or Prostate voxels, despite reference labels in every case. Standard predicts no Bones voxels. Background, Body and Bladder occur in predictions for all 38 cases in both arms. Pooled Bladder DSC is 0.622004 / 0.768230; equal-volume macro DSC is 0.545139 / 0.617475 and equal-patient macro DSC is 0.583946 / 0.711025. These different aggregations must not be interchanged.

Both original validation logs confirm 64³ windows, stride 32, uniform mean-softmax aggregation, matching per-case affine/shape/spacing/window coverage, the same 38 identities and the same absent-class policy. Setup metadata identifies the appropriate architecture at epoch 10 / 1,430 steps. Standard full validation names `standard_unet_last.pt`, whereas paired-case prediction JSONs name `standard_unet_epoch10.pt`; Improved full validation names `improved_unet_epoch10.pt`. The full-validation logs do not record checkpoint binary hashes, so identical Standard weights across those differently named artefacts are not independently established. All six available paired-case confusion matrices match the corresponding full-validation records. Equal updates do not imply equal compute, and one seed is not a statistical replication. Residual units and deep supervision changed together, so these results cannot isolate either component's causal contribution or measure a 2D-versus-3D effect.

### 5.2 Representative Case Selection

Four cases from four distinct validation patients were selected purposively: B040 and M030 illustrate substantial Bladder gains; S022 provides a genuine regression counterexample; K008 challenges a small-prostate-volume-only explanation. K008 has the largest annotated prostate voxel count identified in the representative-case screening, not necessarily in the entire dataset. These cases illustrate failure modes and are not an unbiased estimate of population performance.

All pairs use Epoch 10 checkpoints, the same MRI identity, shared resampled grid, 64³ windows, stride 32, mean-softmax aggregation and metric definitions. Local JSON inspection confirms matching affines, shapes, preprocessing policies, truth class counts and Z indices for M030, S022 and K008. B040's two PNGs and supplied execution records are available; its prediction-driver JSONs remain unavailable, so PNG model attribution retains student-record provenance. B040 numerical counts now also agree with the original full-validation confusion matrices.

| Case | Z slices | Standard Bladder DSC | Improved Bladder DSC | GT Prostate voxels | Prostate predicted, both models |
| --- | --- | ---: | ---: | ---: | ---: |
| B040_Week0 | 51, 62, 73 | 0.5365188544 | 0.8694770977 | 7,105 | 0 |
| M030_Week5 | 43, 55, 67 | 0.4278739088 | 0.8515260362 | 5,932 | 0 |
| S022_Week5 | 44, 54, 64 | 0.6243267375 | 0.2489808244 | 5,966 | 0 |
| K008_Week0 | 56, 70, 84 | 0.5598804442 | 0.6861527220 | 11,114 | 0 |

Six local JSONs are the full-precision numerical sources for the last three pairs. Dice/IoU were recomputed from their confusion matrices and agree; precision is `TP/Npred` and recall is `TP/Ntrue`. B040 values also agree with both original full-validation records; its missing prediction-driver JSON is not claimed inspected. Exact artefact paths, hashes, checkpoint identities and evidence limitations are in the [failure-case evidence manifest](docs/evidence/failure-cases.md).

### 5.3 Paired Anatomical Failure Case Studies

**Figure handling.** Eight original PNGs were inspected without modification. The specification requests failure panels and permits README visual content (Sections 2.2, 4.1 and 8), but explicit HipMRI permission for public redistribution of derived MRI images was not established from the available material. Originals remain outside Git. The following are labelled insertion placeholders, not embedded figures. For a final local PDF, include the original PNGs only if course/dataset rules permit that use, label the two model panels explicitly, retain legends and use size/layout adjustments that preserve scientific meaning. The repository's Apache licence alone does not establish HipMRI image rights.

Each original has four aligned columns: Normalised MRI, Ground Truth, Prediction, and Error Map, repeated at three Z indices. Class colours are Background black, Body yellow, Bones grey, Bladder blue, Rectum green and Prostate pink. Error colours separately encode prostate false positives red, prostate false negatives cyan, and other class errors yellow. All arrays use `[:,:,z].T` on the same resampled grid. Source-axis Z slices have not been anatomically reoriented; low/high Z is not designated prostate apex/base here.

#### Case 1 — B040_Week0

Bladder DSC rises from **0.5365 to 0.8695**, with recorded recall approximately **36.9% → 80.7%** and precision **98.6% → 94.2%**. In the inspected slices, Improved Body predictions contain fewer conspicuous internal gaps, while the high-Z slice shows Bladder overlap. The full-volume numbers establish the Bladder gain; three displayed slices do not establish uniform boundary accuracy.

Both predictions omit Bones, Rectum and Prostate. All **7,105** reference Prostate voxels become other labels: Standard assigns **7,093 to Body and 12 to Bladder**; Improved assigns **7,042 to Body and 63 to Bladder**. Each row sums correctly to 7,105. Pink reference prostate regions correspond to cyan false-negative regions, including Z=62. This is complete organ omission, not a small contour displacement. The original B040 narrative is preserved byte-for-byte in the local-only archive at `research_archive/07_model_development_snapshots/P5-08_superseded_active_docs_2026-10-15/recognition/Failiure-analysis.md`.

> **Figure 5.1 — insertion placeholder.** B040_Week0, Epoch 10: (A) Standard and (B) Improved; rows Z=51,62,73, columns MRI / Ground Truth / Prediction / Error Map. Improved Body coverage is more continuous in these slices; reference Prostate, Rectum and Bones are absent from both predictions. Artefacts: [B040 manifest entry](docs/evidence/failure-cases.md#b040_week0).

#### Case 2 — M030_Week5

Bladder DSC increases from **0.4279 to 0.8515**; recall rises from **27.26% to 75.10%**. Local counts show **35,929 → 98,975** correctly predicted Bladder voxels out of **131,783** reference voxels. The Z=67 prediction covers more of the blue reference Bladder region, while Improved Body overlays show fewer gaps across the selected slices.

Bones, Rectum and Prostate remain absent from both predictions despite their presence in ground truth. The **5,932** Prostate voxels yield zero predicted Prostate voxels, and the central pink region remains a cyan false-negative region at Z=55. Improved Bladder coverage therefore does not resolve the principal target-organ failure.

> **Figure 5.2 — insertion placeholder.** M030_Week5: (A) Standard and (B) Improved, Z=43,55,67; four aligned columns as in Figure 5.1. Larger Bladder overlap at Z=67 and fewer Body gaps coexist with persistent Prostate false negatives. Artefacts: [M030 manifest entry](docs/evidence/failure-cases.md#m030_week5).

#### Case 3 — S022_Week5

Bladder DSC decreases from **0.6243 to 0.2490**. Standard correctly predicts **10,838 / 20,816** reference Bladder voxels; Improved correctly predicts only **3,298 / 20,816**. Recall falls **52.07% → 15.84%**, and precision falls **77.95% → 58.10%**. The local confusion matrices support all four percentages.

At Z=64, the Improved blue Bladder prediction is visibly much smaller than the reference region, with extensive other-class error overlay. Although Body overlays appear more continuous in some slices, this does not compensate for worse Bladder coverage. Both models still omit the **5,966** reference Prostate voxels. This case demonstrates that pooled improvement can conceal a substantial patient-specific regression; its cause cannot be inferred from these panels.

> **Figure 5.3 — insertion placeholder.** S022_Week5: (A) Standard and (B) Improved, Z=44,54,64; MRI / Ground Truth / Prediction / Error Map. The high-Z row illustrates reduced Improved Bladder coverage; cyan Prostate false negatives persist. Artefacts: [S022 manifest entry](docs/evidence/failure-cases.md#s022_week5).

#### Case 4 — K008_Week0

Bladder DSC improves from **0.5599 to 0.6862**, with recall **41.96% → 57.74%**. Improved Body overlays show fewer internal gaps at Z=56 and Z=70, and the blue Bladder region at Z=84 covers more of the reference. These are selected-slice observations, not a measured slice-transition metric.

Ground truth contains **11,114** Prostate voxels, yet neither model predicts any. The pink reference region at Z=70 remains cyan in both error maps. Failure on this larger screened prostate challenges the explanation that omission is merely a consequence of exceptionally small prostate volume; it does not disprove class imbalance or identify a cause.

> **Figure 5.4 — insertion placeholder.** K008_Week0: (A) Standard and (B) Improved, Z=56,70,84; four aligned columns. Improved Body/Bladder coverage coexists with total Prostate omission. Artefacts: [K008 manifest entry](docs/evidence/failure-cases.md#k008_week0).

### 5.4 Cross-Case Failure Patterns

The central pattern is **partial class prediction collapse**: both architectures systematically omit Prostate and Rectum, while Bones are absent for Standard and effectively unsegmented for Improved. In all four selected cases, annotated prostate voxels are assigned to other classes. Better Body appearance and Bladder overlap therefore do not establish usable prostate contours.

Bladder under-segmentation varies considerably. Improved gains in three selected cases coexist with S022's marked regression; the full 38-volume pooled gain should not be presented as a guarantee for individual patients. Sparse displayed Z levels support qualitative continuity observations but cannot quantify consecutive-slice transitions, surface distances or clinical base/apex errors. Full Train reconstruction now quantifies imbalance and sampling variation; their causal role, loss behaviour, optimisation and anatomical variability remain unproven.

The subsequent [fixed-patch probability probe](docs/analysis/class-prediction-collapse.md#authenticated-real-probability-results), job 650924, evaluates two Train crops and one Validation center crop with both E10 models. Bones true-class means are 0.135745-0.199654 and usually rank second, yet Body wins every true Bones voxel. Rectum means are only 0.013411-0.028011; Prostate improves in the Improved model but has zero TP in all selected outputs. Failure already on both Train crops weakens a Validation-only explanation. These selected-patch manifestations are not causes or replacements for the full-volume metrics above.

### 5.5 Computational and Operational Considerations

Default trainable parameter counts are **5,646,470** (Standard) and **5,734,114** (Improved), a difference of **87,644**, approximately **1.55%**. Auxiliary heads contribute parameters even though inference returns only the main head. Both models use on-demand preprocessing; no production training cache was integrated.

The following matched-case measurements are recorded in the six locally inspected Job 650490 JSONs, using NVIDIA A100-PCIE-40GB, PyTorch 2.13.0+cu130 and CUDA build 13.0. Times are full sliding-window function measurements, including CPU probability accumulation and argmax; preprocessing, metric calculation and PNG generation are separate fields.

| Case | Resampled XYZ | Windows | Standard inference (s) | Improved inference (s) |
| --- | --- | ---: | ---: | ---: |
| M030_Week5 | 256×256×128 | 147 | 9.625768 | 9.464645 |
| S022_Week5 | 256×256×128 | 147 | 9.444015 | 9.523250 |
| K008_Week0 | 268×268×128 | 192 | 7.790201 | 7.931949 |

| Model | Trainable parameters | Peak allocated GPU bytes, each recorded case | Peak reserved GPU bytes, each recorded case |
| --- | ---: | ---: | ---: |
| Standard | 5,646,470 | 362,597,376 | 379,584,512 |
| Improved | 5,734,114 | 362,163,200 | 396,361,728 |

These are single runs for three paired cases, not a repeated or complete 38-volume resource benchmark. Device name matches, but node identity, warm-up and repeated-run variability are not established. Small timing/memory differences do not demonstrate an architecture-wide efficiency advantage. Reported Improved Validation elapsed time **8m04s** is a job duration, not isolated inference latency or a matched Standard timing comparison. Historical CPU peak RSS and one-step GPU training memory describe different operations and cannot substitute for these measurements or full-training resource costs.

Original validation logs additionally record loop times of **508.611325 s** (Standard) and **458.372391 s** (Improved), including preprocessing and metrics; summed sliding-window function times are **315.113957 s / 277.927740 s**. These are one run per model, not repeated timing trials or scheduler durations. The available training segments sum to **5,146.637220 s** for seven Standard epochs and **6,906.425768 s** for nine Improved epochs; neither is a measured ten-epoch total or a matched training-time comparison. Recorded training process peak allocated GPU memory is **487,463,424 / 487,471,616 bytes**, and peak reserved memory **813,694,976 / 836,763,648 bytes**, for these respective resumed runs, not inference or whole-process CPU RAM.

### 5.6 Critical Evaluation and Limitations

The Improved model improves pooled Background, Body and Bladder segmentation under the recorded ten-epoch protocol, but does not deliver reliable prostate segmentation. Complete Prostate/Rectum omission and S022's regression preclude recommending either current checkpoint as an autonomous radiotherapy contouring tool. For the project manager, the supported recommendation is to retain them as research prototypes and require specialist review of all contours; selected Bladder gains do not establish clinical safety.

The findings establish observed failures, not their root causes or the isolated efficacy of residual learning or deep supervision. Source/synthetic loss checks and the authentic recorded loss trajectory do not establish class-specific learning. [Job 650736 Train-only exposure evidence](docs/analysis/class-prediction-collapse.md#original-full-train-audit-authentication) now verifies 143 volumes / 26 patients / 1,430 reconstructed patches. Every raw/resampled volume contains all six classes. Prostate-aware sampling enriches all four investigated classes, and strong Bones exposure contrasts with negligible Validation predictions; simple rarity or tiny fragments alone cannot explain that failure. Background+Body still occupy 84.20% of patch voxels. Job 650924 now provides authenticated selected-patch probabilities; component/parameter gradients remain unmeasured, and replay does not independently prove historical image-file immutability. One seed, purposive case selection, missing early training segments, absent historical Standard Validation checkpoint hash and incomplete repeated matched compute profiling limit the conclusions. Original E10 training/validation logs are now inspected locally, resolving the previous full-run-log evidence gap. Exact anatomical base/apex analysis, surface metrics and clinical annotation review remain uncompleted.

The probe's current Standard snapshot and `last.pt` have identical file SHA-256 and state tensors in its protected preflight, but the original Validation log does not hash the past `last.pt`. All six probability outputs/fixed identities are authenticated locally from the downloaded log; raw arrays and binaries are not independently rerun. Improved raises selected-patch Bladder recall but lowers precision, and K019 Rectum NLL/Soft Dice slightly regress. A [bounded Train-only CE/Dice parameter-gradient attribution measurement](docs/analysis/class-prediction-collapse.md#recommended-next-measurement) is recommended before retraining: probabilities/class frequencies do not measure effective gradients. No new Validation performance would result from that read-only measurement.

The held-out **30-volume Test** evaluation remains separate and unexecuted here; no test Dice target attainment is claimed. A separate `predict_v3.py` adapter now has synthetic CPU admission checks, but its real E12 checkpoint/receipt preflight and Validation equivalence remain pending; it is not a formally frozen final path. Tutor check-off and final submission are also unverified. The comprehensive task README/PDF still needs its outstanding assessment material and authorised figure insertion; this report and manifest do not by themselves establish submission completion.

## V4 E20: Completed Paired Validation Experiment

**Current status (14 October 2026):** the frozen Control/Treatment continuation and scheduled E14/E16/E18/E20 full-volume Validation completed. Treatment was rejected under the frozen conjunctive rule; V3 E12 remains the protected candidate. Control E20 was not automatically selected. No formal model freeze or Held-out Test evaluation has occurred.

### Motivation

The selected V3 E12 model provides the parent checkpoint for
an exploratory follow-up targeting Prostate false positives.

This follow-up tested a loss-function intervention rather than
introducing a new architecture or changing the dataset split.

### Experimental design

The Control continued the original V3 objective. The Treatment
increased the Prostate Tversky false-positive coefficient from
alpha=0.5 to alpha=0.7.

Both arms started from the selected V3 E12 checkpoint and
continued through E20.

The paired comparison used the same Train/Validation split,
143 optimiser updates per epoch and full-volume Validation
at E14, E16, E18 and E20.

### Evaluation criteria

The predeclared E20 decision rules include:

- Prostate patient-macro Dice gain versus Control >=0.03.
- Pooled Prostate FP reduction versus Control >=15%.
- Prostate recall decrease <=0.05.
- Other-class patient-macro Dice decrease <=0.02 each.
- No Prostate regression in at least four of six
  Validation patients.
- Required class coverage and comparison with the
  selected V3 parent baseline.

These criteria were specified before E20; their observed outcomes are reported below.

### Engineering verification

The implementation has passed bounded CPU tests covering
loss parity, gradients, synthetic training, paired records,
checkpoint restoration, journals, failure-retry admission
and lock lifecycle.

The latest E20 Runner also passed its final three-test
CPU numerical regression.

Several checks used synthetic data or mocked dependencies.
These bounded checks alone did not establish real-MRI performance; the later full-volume Validation evidence is reported below.

### Results and limitations

Treatment reduced **pooled Prostate false positives by 32.30%**. At E20, pooled Prostate Dice was 0.617164 for Control and 0.650778 for Treatment; **patient-macro** Prostate Dice was 0.616010 and 0.643060, respectively. Pooled, volume-macro and patient-macro scores are distinct aggregation methods; the independent audit reports all three.

Three mandatory conjunctive gates failed: patient-macro Prostate Dice gain was 0.027050, below 0.03; pooled Prostate Recall loss was 0.076124, above 0.05; and maximum other-class patient-macro Dice decrease was 0.048440, above 0.02, driven by Rectum. The frozen decision (local-only: `research_archive/10_scientific_documentation/COMP3710_V4_FINAL_DOCS_20261014/V4_ACCEPTANCE_DECISION.md`) rejected Treatment and retained V3 E12 as the protected candidate. Control E20 was not automatically selected. The P4-04C independent audit (local-only: `research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md`) recomputed the Validation metrics and rule; P4-04E (local-only: `research_archive/08_validation_and_independent_audit/COMP3710_P4-04E_20261014/V4_EVIDENCE_CLOSURE.md`) closed V4 numerical reproducibility while local checkpoint-payload inspection remains unknown.

The six Validation patients had previously informed model development and parent selection, so this comparison is exploratory, not independent confirmation. The retained candidate is **not formally frozen**. The P4-05A readiness audit (local-only: `research_archive/12_handover_and_pretest_readiness/COMP3710_P4-05A_20261014/P4-05A_READINESS_AUDIT.md`) found that the existing `predict.py` is not a verified V3 E12 final inference path. Held-out Test has not been evaluated; no Test metric, clinical reliability or final submission completion is claimed.
