# COMP3710 HipMRI — Week 11 Experiment Review, Open Questions and File Index

[中文版](W11-ZH.md)

> Current as of 15 October 2026. **Implementation verified** means checked against current code; **original-log/independent evidence** means authentic records were inspected and their arithmetic recomputed; **student-reported** means a remote run or hash receipt was recorded without local access to the original binary; **derived** means calculated from stated counts; **hypothesis** means causal attribution remains untested. Every performance result below concerns development Validation. No held-out Test arrays were decoded or evaluated. Relative links start from this document. `research_archive/` is a local, Git-ignored evidence archive, separate from the tracked submission.

## Part 1 — Completed Work, Model Evolution and Experimental Results

### 1.1 Project overview and evidence boundaries

The HipMRI task is six-class 3D semantic segmentation: 0 Background, 1 Body, 2 Bones, 3 Bladder, 4 Rectum and 5 Prostate. The research aim is stronger foreground segmentation, especially structures relevant to prostate radiotherapy planning. There are **two architecture families**: [StandardUNet3D](recognition/hipmri_3d_improved_unet_hard/modules.py) and [ImprovedUNet3D](recognition/hipmri_3d_improved_unet_hard/improved_modules.py). Improved V1–V4 reuse the same architecture. V2 changes class weighting, V3 continues V2 and uses protected selection, and V4 compares a paired change to the Prostate overlap objective. These versions are experiments, not four separate U-Net designs.

V3 E12 remains a **protected candidate**, without a formal final freeze. V4 Treatment failed the frozen conjunctive E20 rule and was rejected; Control E20 was not automatically selected. Final V3 inference compatibility with the original remote checkpoint and the held-out Test remain unverified. The authoritative navigation starts with the [consolidated experiment history](recognition/docs/experiments.md), [technical report](recognition/FINAL-REPORT.md), [claim-to-code register](recognition/docs/evidence/traceability.md) and [P4-05A readiness audit](research_archive/12_handover_and_pretest_readiness/COMP3710_P4-05A_20261014/P4-05A_READINESS_AUDIT.md).

### 1.2 What happened to the dataset, in execution order

Discovery of file names is distinct from decoding MRI arrays. Header agreement establishes geometry, not clinical annotation accuracy. A patient, rather than an individual longitudinal scan, is the independent split unit.

| Step | Implemented operation, purpose and parameters | Implementation, verification and remaining limit |
|---:|---|---|
| 1 | Enumerate `semantic_MRs/*_LFOV.nii.gz` and `semantic_labels_only/*_SEMANTIC.nii.gz`; pair by patient ID and Week; reject unmatched or duplicate pairs. | [`dataset.py`: `parse_volume_filename`, `discover_dataset`](recognition/hipmri_3d_improved_unet_hard/dataset.py). Production discovery recorded 38 patients and 211 matched pairs. This stage does **not** decode arrays. |
| 2 | Keep all Weeks for a patient together. The approved split is Train **26 patients/143 volumes**, Validation **6/38**, Test **6/30**. Seed 3710 records split provenance; it does not redraw the assignments at runtime. | `APPROVED_PATIENTS` and `validate_approved_split` reject overlap, unexpected identities and incorrect counts; see the [package README](recognition/hipmri_3d_improved_unet_hard/README.md). Training and selection do not use Test. Repeated Weeks raise volume count without increasing independent patient count. |
| 3 | Require labels 0–5, allowing a subset in one scan. Compare paired 3D shape, effective affine, spacing, orientation and compatible units. | [`volume_io.py`: `load_volume_pair`](recognition/hipmri_3d_improved_unet_hard/volume_io.py). A student-run audit of all 211 headers found no grid mismatch. MRI headers declare mm; masks declare unknown. The working unit is conditionally inferred only after geometric checks, and the original mask declaration is retained. |
| 4 | Decode one paired NIfTI array via `get_fdata(float32)`; require finite MRI values and integral mask labels 0–5 before conversion to `uint8`. Preserve metadata. | `load_volume_pair` and [`test_volume_io.py`](recognition/hipmri_3d_improved_unet_hard/test_volume_io.py). The real K019_Week1 pair loaded at `(256,256,144)`; subsequent Train/Validation processing covered 181 pairs. This says nothing about Test-array access or annotation quality. |
| 5 | Resample both arrays to one physical grid at the training-derived candidate spacing `(1.6796875,1.6796875,1.55999755859375)` mm. Preserve voxel-axis directions and voxel-edge field of view. MRI uses linear interpolation, order 1 with `nearest` boundary; the discrete mask uses nearest-neighbour, order 0 with `grid-constant`, background 0. | [`resampling.py`: `make_target_grid`, `resample_volume_pair`](recognition/hipmri_3d_improved_unet_hard/resampling.py), [`resampling_config.json`](recognition/hipmri_3d_improved_unet_hard/resampling_config.json) and six-face synthetic boundary tests. Shared shape/affine and foreground-label survival are checked. The 512 MiB fourfold estimate is an allocation guard, not a bound on peak RAM. Spacing provenance is a student Train-only audit, not an optimum-spacing experiment. |
| 6 | On each resampled MRI, select original nonzero voxels, clip at percentiles 0.5/99.5 and z-score using `float64` population statistics; emit `float32` and preserve original zeros. Masks never select intensity statistics. | [`intensity_normalization.py`](recognition/hipmri_3d_improved_unet_hard/intensity_normalization.py), [`normalization_config.json`](recognition/hipmri_3d_improved_unet_hard/normalization_config.json) and synthetic tests. Per-volume statistics avoid fitting across patients but may leave contrast variability. |
| 7 | Apply the same crop and XYZ→ZYX transpose `(2,1,0)` to MRI and mask. Tensors are MRI `[1,D,H,W]` and integer mask `[D,H,W]`. | [`patch_dataset.py`: `HipMRIPatchDataset.__getitem__`](recognition/hipmri_3d_improved_unet_hard/patch_dataset.py), tests and real patch records. A one-hot mask is not materialised; whether this satisfies assignment guidance needs tutor clarification. |
| 8 | Draw one aligned `64³` Train patch per volume per epoch. `SeedSequence(seed,epoch,index)` separates crop identity from DataLoader order. The 0.5 parameter chooses a prostate-aware branch that includes a true class-5 voxel, versus a random branch, with fallback for absence. There is no production augmentation or full-volume cache. | [`patch_dataset_config.json`](recognition/hipmri_3d_improved_unet_hard/patch_dataset_config.json) and the independently checked [Job 650736 exposure reconstruction](recognition/docs/analysis/class-prediction-collapse.md#verified-real-patch-exposure-job-650736): 1,430 patches, 754 prostate-branch and 676 random. 880/1,430 actually contain Prostate; 0.5 is not the positive-patch rate or a guarantee of whole-organ context. |
| 9 | `DataLoader(batch_size=1, shuffle=True, num_workers=0)` gives 143 mini-batches and optimiser updates per epoch; ten epochs give 1,430 updates. Save model/optimiser/RNG at atomic completed-epoch boundaries; use seed 3710, float32, disabled TF32 and warn-only deterministic algorithms. | [`train.py`](recognition/hipmri_3d_improved_unet_hard/train.py), versioned trainers and original E10 JSONL. Resume checks full configuration and source hashes. An epoch is **not** a mini-batch; GPU bitwise reproducibility is not guaranteed. |
| 10 | Evaluate only the 38 Validation volumes from six patients. Full-volume inference uses `64³` windows, stride 32, high-side coverage, mean overlapping softmax probabilities and final argmax on the shared resampled XYZ grid. | [`validate.py`: `sliding_window_predict`, `select_validation`](recognition/hipmri_3d_improved_unet_hard/validate.py), both original E10 38/38 logs and V3/V4 audit. A selected center patch is only a local diagnostic. |
| 11 | Build a 6×6 truth-row/prediction-column confusion matrix; report TP, FP, FN, predicted and GT counts, Dice, IoU, Precision and Recall. Distinguish **pooled** voxel totals, **volume-macro** means and **patient-macro** means after pooling each patient's Weeks. Both-absent classes are null/excluded from macros; one-sided absence scores zero. | `validate.py` metric functions, the [E10 evidence manifest](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance) and [P4-04C independent arithmetic](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md). A few illustrated slices cannot establish apex/base or surface errors. |

### 1.3 Model and training-version comparison

| Feature | Standard E10 | Improved V1 E10 | Improved V2 E10 | Improved V3 E12 | V4 Control E20 | V4 Treatment E20 |
|---|---|---|---|---|---|---|
| Architecture / trainable parameters | StandardUNet3D; 5,646,470 | ImprovedUNet3D; 5,734,114 | Same as V1 | Same as V1 | Same as V1 | Same as V1 |
| Encoder/decoder, 3D conv, skips | Four downsamplings, double 3D conv, GroupNorm(1)/ReLU; transposed conv and concatenated skips | Residual units in encoder, bottleneck and decoder; 3D conv and skips; main plus two auxiliary heads | Unchanged | Unchanged | Unchanged | Unchanged |
| Input/output and base channels | 1→6 logits; base 16 | Same as Standard | Unchanged | Unchanged | Unchanged | Unchanged |
| Main / auxiliary output | Main only | Main plus half- and quarter-resolution auxiliary outputs during training; main only at inference | Same as V1 | Same as V1 | Same as V1 | Same as V1 |
| Loss | .5 CE + .5 foreground Soft Dice | Three heads with original loss; weights 1:.5:.25 / 1.75 | Three heads with .5 weighted CE + .5 original Dice | Same as V2 | Same as V3; alpha .5 equivalent to original Dice | Replace only Prostate overlap with soft Tversky alpha .7/beta .3; weighted CE fixed |
| Optimiser | AdamW, LR .001, weight decay .00001, betas .9/.999; no recorded schedule | Unchanged | Unchanged | Restore V2 model/optimiser/RNG; LR .001 | Same V3 E12 parent; LR .001 | Matched parent, crops and training order |
| Data/training | 143 Train volumes; batch 1; one 64³ patch/volume/epoch; prostate branch .5; no augmentation; seed 3710 | Unchanged | Unchanged | Unchanged | Unchanged | Unchanged |
| Split/Validation | 26/6/6 patients; E10 38-volume 64³/stride-32 evaluation | Unchanged | Unchanged | E12 A/B train, C full Validation and protected selection | Full Validation at E14/16/18/20; E20 primary endpoint | Same as Control |
| Parent and update budget | Initial E1–10; 1,430 updates | Initial E1–10; 1,430 | Initial E1–10; 1,430 | V2 E10→E11–12; +286, cumulative 1,716 | V3 E12→E13–20; +1,144 per arm, cumulative 2,860 | Same as Control |
| Actual intervention / limit | Architectural baseline | Residual blocks and deep supervision change together | Class-weighted CE; network fixed | Continuation and predeclared protected selection | Original-objective continuation; not an automatic winner | Prostate overlap alpha .5→.7, other objective terms fixed |

The parameter counts and network structure are verified in the [traceability register](recognition/docs/evidence/traceability.md#section-2--model-architecture). The [experiment history](recognition/docs/experiments.md) details V2–V4 protocol. Standard→V1 changes residual units and deep supervision together, so this comparison cannot isolate either component. Even V4's tightly paired experiment has one seed, shared repeatedly observed Validation patients and possible GPU numerical variation.

### 1.4 Implemented loss functions

Let `p_ic=softmax(logits)_ic`, `y_ic=1[y_i=c]`, `N` be batch×voxel count for one head and `ε=10⁻⁵`. The source sums over batch and spatial dimensions and averages the foreground classes `c=1..5`; absent classes still contribute to training Soft Dice. Define `CE=−N⁻¹Σ_i log p_i,y_i`, `D_c=(2Σ_i p_ic y_ic+ε)/(Σ_i p_ic+Σ_i y_ic+ε)` and `L_D=1−(1/5)Σ_{c=1}^5 D_c`. **Standard:** `L=.5CE+.5L_D`. This is a differentiable probability-based training objective, distinct from hard-label Validation Dice after argmax.

**Improved V1** applies that Standard objective independently to the main, half-resolution and quarter-resolution outputs. Auxiliary logits are upsampled by trilinear interpolation with `align_corners=False` to the original target grid. The final loss, and the logged CE/Dice components, are `L_DS=(L_main+.5L_half+.25L_quarter)/1.75`.

**V2 and V3** retain this deep-supervision scheme and the original foreground Dice. From the 1,430 training patches, let `f_c` denote the fraction of sampled voxels in class `c`. V2 sets `w_c=f_c^(−1/2)/Σ_j sqrt(f_j)`, so `Σ f_cw_c=1`, and `WCE=N⁻¹Σ_i w_{y_i}(−log p_i,y_i)`. It does **not** divide by the sum of selected weights. The class-ordered weights are `1.1775485, 0.6250278, 1.6794010, 2.5227875, 4.5640100, 4.2290216`. Each head uses `.5WCE+.5L_D`; ordinary CE is logged but not reused in the objective. V3 changes continuation and selection, not this loss. [Source](recognition/hipmri_3d_improved_unet_hard/train_improved_v2.py), [frozen V2 config](recognition/hipmri_3d_improved_unet_hard/improved_v2_config.json).

**V4 Control** calls the unchanged V2/V3 loss. In **V4 Treatment**, only class 5's overlap score becomes `T_5=(TP_s+ε/2)/(TP_s+0.7FP_s+0.3FN_s+ε/2)`, with soft `TP_s=Σp_5y_5`, `FP_s=Σp_5(1−y_5)`, `FN_s=Σ(1−p_5)y_5`. Each head has overlap loss `L_D+(D_5−T_5)/5` and total `.5WCE+.5[L_D+(D_5−T_5)/5]`; the same three-head combination follows. Alpha .7 increases the FP penalty and beta .3 weights FN. Weighted CE and the other four foreground Dice terms stay fixed. Calling the whole objective “Tversky loss” would obscure its actual composition. [Frozen E20 implementation](recognition/hipmri_3d_improved_unet_hard/experiment_prostate_fp_e20.py).

### 1.5 Quantitative evidence across the experiments

With batch size one, a mini-batch gives one optimiser update; an epoch has 143 updates; Validation gives **zero** updates. `Not recorded` and `Not independently verified` are not numeric zeros. The original E10 training files contain only resumed segments, Standard E4–10 and V1 E2–10, although their endpoint counters read 10 epochs/1,430 updates. V2 original per-volume records are not local. Sources: [E10 log manifest](recognition/docs/evidence/failure-cases.md#original-e10-log-authentication), [consolidated history](recognition/docs/experiments.md), [P4-04C audit](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md).

| Run / checkpoint | Parent; epoch / added / cumulative updates | Train / Validation; batch; LR | Training, Validation and selection status | GPU, timing and evidence |
|---|---|---|---|---|
| Standard E10 (train 648790; Val 649669) | Initial 10 / 10 / 1,430 | 143/38; 1; .001→.001, no schedule | Ten completed; 38/38; architectural baseline | A100; observed E4–10 training 5,146.637 s; Validation loop 508.611 s; original JSONL recomputed, binary absent |
| V1 E10 (train 649668; Val 650393) | Initial 10 / 10 / 1,430 | Same | Ten completed; 38/38; cancelled Val 649912 replaced | A100; observed E2–10 training 6,906.426 s; Validation loop 458.372 s, reported job 8m04s; original JSONL recomputed |
| V2 E10 (train 652649; Val 652948) | Initial 10 / 10 / 1,430 | Same; weighted CE | Student-reported training and 38/38 Validation completed; V3 parent | GPU/total runtime not independently verified; remote records/hashes, local per-volume file and binary absent |
| V3 E11 stage A | V2 E10 / +1 / 1,573 | Same | 143 updates, validation_pending; no selection | Authenticated split journal; detailed runtime not recorded here |
| V3 E12 stage B; C Validation | V2 E10 / +2 / 1,716 | Same | B: 143 updates; C: zero updates/38 Validation; E12 protected | B SHA bb0d03c7…67660d4, receipt fffe08a6…95b2e189da; journal/hash receipts, local payload unopened |
| V4 Control E14/16/18/20 | V3 E12 / +2/+4/+6/+8; 2,002/2,288/2,574/2,860 | Same; alpha .5 | E13–20: eight epochs, 1,144 updates/arm; four 38/38 Validation runs; E20 not automatically selected | Eight epoch journals, four Val runs; P4-04C arithmetic recomputed; total wall time not recorded here |
| V4 Treatment E14/16/18/20 | V3 E12 / same as Control | Same; alpha .7 | Matched protocol; E20 failed three required gates and was rejected | Same evidence scope; E20 checkpoint SHA 11be4077…101113 is a hash receipt, not local payload inspection |

**Six-class Validation Dice.** In every six-number cell the order is Background, Body, Bones, Bladder, Rectum, Prostate. Values display six decimals. E10 comes from recomputed [original matrices](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance); V2 is [student-reported](recognition/docs/experiments.md#improved-v2--weighted-loss-e10); V3/V4 are from [P4-04C's independently aggregated 38-volume JSON](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/independent_results.json). Pooled, equal-volume and equal-patient estimands must never be substituted for each other.

| Model / endpoint | Pooled DSC, six classes in order | Patient-macro DSC, six classes in order | Volume-macro DSC, six classes in order |
|---|---|---|---|
| Standard E10 | .941342, .842790, 0, .622004, 0, 0 | .935937, .835655, 0, .583946, 0, 0 | Full six-class series not recorded locally |
| V1 E10 | .976012, .906781, **3.9431×10⁻⁷**, .768230, 0, 0 | .973940, .904107, 0 (rounded), .711025, 0, 0 | Full six-class series not recorded locally |
| V2 E10 | .953976, .862141, .362280, .826771, .461378, .445695 | .948549, .854755, .361053, .779908, .439592, .453977 | Not independently verified |
| V3 E12 | .969837, .868735, .476785, .821240, .537022, .426721 | .966336, .866039, .476872, .785414, .529170, .415424 | .970080, .867989, .483219, .732820, .468399, .416253 |
| Control E14 | .978476, .907809, .540260, .832454, .503847, .443944 | .976655, .906884, .537156, .792843, .509126, .448163 | .978497, .906996, .540310, .757735, .467099, .439084 |
| Treatment E14 | .977796, .906782, .532547, .830511, .508864, .464330 | .975945, .905765, .530180, .790622, .518236, .461007 | .977838, .905830, .532225, .755654, .472707, .455346 |
| Control E16 | .989507, .942148, .557011, .840684, .583710, .590097 | .989309, .942529, .550730, .798259, .577954, .585625 | .989420, .941413, .552954, .771764, .530700, .581150 |
| Treatment E16 | .989778, .942325, .552775, .842253, .579458, .587816 | .989709, .943034, .551371, .800183, .574215, .587122 | .989672, .941534, .549010, .775710, .525896, .580254 |
| Control E18 | .986332, .919894, .622998, .786453, .553822, .381516 | .985485, .920189, .623358, .738965, .542708, .388404 | .986237, .919572, .626445, .702082, .508290, .380725 |
| Treatment E18 | .985669, .919387, .621311, .713926, .564680, .418833 | .984909, .919635, .620918, .678479, .553599, .425146 | .985565, .918688, .624138, .632502, .519137, .415446 |
| Control E20 | .989198, .946139, .670668, .843456, .593401, .617164 | .988608, .946059, .665497, .803031, .584171, .616010 | .989099, .945737, .669831, .757965, .537333, .621048 |
| Treatment E20 | .984951, .936111, .676095, .841011, .543693, .650778 | .983738, .935057, .671595, .797808, .535731, .643060 | .984891, .935426, .675757, .757564, .503413, .653441 |

The tiny V1 Bones Dice is **positive**, not exact zero: three predicted voxels, two TP. At E10, Standard never predicted Bones/Rectum/Prostate; V1 never predicted Rectum/Prostate. V2's complete local per-volume matrices and full class-wise IoU/Precision/Recall are unavailable. Its Dice cannot reconstruct every confusion count.

**Pooled Prostate trajectory.** TP, FP and FN pool all 38 volumes; GT is 251,012 at each point and `pred=TP+FP`. TP-positive coverage means cases with actual TP, not merely any prediction. The V3 selection/audit records 38/38 coverage for classes 2–5 even though the summary JSON lacks this one field. Counts and six-decimal Precision/Recall are from [P4-04C](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/independent_results.json); E10 absence is from its [original logs](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance).

| Endpoint | TP | FP | FN | pred | GT | pooled Precision / Recall | TP-positive volumes |
|---|---:|---:|---:|---:|---:|---|---:|
| Standard / V1 E10 | 0 | 0 | 251,012 | 0 | 251,012 | 0 / 0 (no positive prediction) | 0/38 |
| V3 E12 | 175,508 | 396,069 | 75,504 | 571,577 | 251,012 | .307059 / .699202 | 38/38; selection rule |
| Control E14 | 179,248 | 377,266 | 71,764 | 556,514 | 251,012 | .322091 / .714101 | 38/38 |
| Treatment E14 | 172,275 | 318,750 | 78,737 | 491,025 | 251,012 | .350848 / .686322 | 38/38 |
| Control E16 | 205,328 | 239,573 | 45,684 | 444,901 | 251,012 | .461514 / .818001 | 38/38 |
| Treatment E16 | 207,781 | 248,166 | 43,231 | 455,947 | 251,012 | .455713 / .827773 | 38/38 |
| Control E18 | 238,266 | 759,770 | 12,746 | 998,036 | 251,012 | .238735 / .949222 | 38/38 |
| Treatment E18 | 231,042 | 621,211 | 19,970 | 852,253 | 251,012 | .271096 / .920442 | 38/38 |
| Control E20 | 216,950 | 235,093 | 34,062 | 452,043 | 251,012 | .479932 / .864301 | 38/38 |
| Treatment E20 | 197,842 | 159,163 | 53,170 | 357,005 | 251,012 | .554172 / .788177 | 38/38 |

**Complete pooled error composition at V3 E12 and E20.** These six-class TP/FP/FN, IoU, Precision, Recall and prediction counts are taken from the [independently recomputed JSON](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/independent_results.json). `Precision=TP/pred`, `Recall=TP/GT`, `IoU=TP/(TP+FP+FN)`. In class order, the GT voxel counts shared by all three checkpoints are `195612026,105931724,10144275,1742051,500065,251012`. The linked JSON also contains the corresponding full E14/E16/E18 class fields and per-volume/patient evidence; their six-class Dice trajectory appears above.

| Endpoint / class | IoU | Precision | Recall | TP | FP | FN | pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 Background | .941441 | .954112 | .986089 | 192,890,936 | 9,277,090 | 2,721,090 | 202,168,026 |
| V3 Body | .767933 | .925826 | .818276 | 86,681,423 | 6,944,622 | 19,250,301 | 93,626,045 |
| V3 Bones | .313012 | .390753 | .611396 | 6,202,165 | 9,670,162 | 3,942,110 | 15,872,327 |
| V3 Bladder | .696698 | .910514 | .747909 | 1,302,896 | 128,050 | 439,155 | 1,430,946 |
| V3 Rectum | .367075 | .530644 | .543555 | 271,813 | 240,419 | 228,252 | 512,232 |
| V3 Prostate | .271230 | .307059 | .699202 | 175,508 | 396,069 | 75,504 | 571,577 |
| E20 Control Background | .978628 | .988196 | .990203 | 193,695,567 | 2,313,638 | 1,916,459 | 196,009,205 |
| E20 Control Body | .897784 | .944526 | .947758 | 100,397,607 | 5,896,511 | 5,534,117 | 106,294,118 |
| E20 Control Bones | .504515 | .722625 | .625682 | 6,347,090 | 2,436,294 | 3,797,185 | 8,783,384 |
| E20 Control Bladder | .729290 | .798453 | .893835 | 1,557,107 | 393,048 | 184,944 | 1,950,155 |
| E20 Control Rectum | .421870 | .511031 | .707428 | 353,760 | 338,488 | 146,305 | 692,248 |
| E20 Control Prostate | .446303 | .479932 | .864301 | 216,950 | 235,093 | 34,062 | 452,043 |
| E20 Treatment Background | .970349 | .985721 | .984183 | 192,518,020 | 2,788,773 | 3,094,006 | 195,306,793 |
| E20 Treatment Body | .879896 | .937336 | .934890 | 99,034,490 | 6,620,807 | 6,897,234 | 105,655,297 |
| E20 Treatment Bones | .510682 | .685131 | .667294 | 6,769,211 | 3,110,953 | 3,375,064 | 9,880,164 |
| E20 Treatment Bladder | .725642 | .784288 | .906578 | 1,579,305 | 434,374 | 162,746 | 2,013,679 |
| E20 Treatment Rectum | .373337 | .412250 | .798190 | 399,147 | 569,068 | 100,918 | 968,215 |
| E20 Treatment Prostate | .482336 | .554172 | .788177 | 197,842 | 159,163 | 53,170 | 357,005 |

**E10 class counts and prediction coverage.** Coverage below is volumes with any predicted voxels, not Dice>0. `FP=pred−TP` and `FN=GT−TP` can be calculated from the displayed entries. Source: [authenticated original E10 matrices](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance).

| Class | GT | Standard pred / TP / coverage | V1 pred / TP / coverage |
|---|---:|---|---|
| Background | 195,612,026 | 204,051,044 / 188,109,738 / 38 | 197,074,176 / 191,633,198 / 38 |
| Body | 105,931,724 | 108,979,831 / 90,562,636 / 38 | 115,514,860 / 100,401,777 / 38 |
| Bones | 10,144,275 | 0 / 0 / 0 | 3 / 2 / 2 |
| Bladder | 1,742,051 | 1,150,278 / 899,520 / 38 | 1,592,114 / 1,280,703 / 38 |
| Rectum | 500,065 | 0 / 0 / 0 | 0 / 0 / 0 |
| Prostate | 251,012 | 0 / 0 / 0 | 0 / 0 / 0 |

**Additional E10 IoU and V2 Precision/Recall.** Six-class pooled IoU in the same order is Standard `0.889184,0.728295,0,0.451383,0,0` and V1 `0.953148,0.829460,1.971555×10⁻⁷,0.623680,0,0` ([E10 original evidence](recognition/docs/evidence/failure-cases.md#full-validation-numerical-provenance)). V2's [student pooled-confusion summary](recognition/docs/analysis/class-prediction-collapse.md#current-optimisation-decision---12-october-2026) gives Bones Precision/Recall `.528495/.275602`, FP/FN `2,494,303/7,348,494`; Bladder `.788611/.868812`, `405,702/228,536`; Rectum `.597848/.375633`, `126,354/312,224`; Prostate `.340148/.646212`, `314,665/88,805`. These four classes have recorded TP-positive coverage of 38/38. The other two classes' V2 detailed counts are not independently verified locally.

**Recorded E10 training-loss trajectory.** Triples are `CE / Soft Dice loss / total`; objectives in different stages have different scales and should not be compared solely by total magnitude. Standard: E4 `.752481/.830941/.791711`, E5 `.667956/.823227/.745591`, E6 `.721923/.802126/.762024`, E7 `.651950/.781117/.716534`, E8 `.646269/.772173/.709221`, E9 `.687573/.760931/.724252`, E10 `.628552/.769903/.699227`. V1: E2 `.802051/.818914/.810483`, E3 `.655318/.798339/.726828`, E4 `.618251/.775927/.697089`, E5 `.515769/.773786/.644778`, E6 `.622856/.764036/.693446`, E7 `.513024/.735074/.624049`, E8 `.504781/.724838/.614809`, E9 `.516944/.705877/.611411`, E10 `.507582/.714722/.611152`. Both Soft Dice losses increased E9→E10. Missing earlier epochs are not interpolated. Complete V2/V3/V4 per-epoch CE/Dice/total series were not independently extracted into this review. Exact recorded timings and updates are in the [loss manifest](recognition/docs/evidence/failure-cases.md#recorded-training-loss-trajectory).

**Resources and sampling exposure.** [Final Report §5.5](recognition/FINAL-REPORT.md#55-computational-and-operational-considerations) reports Standard/V1 resumed-run peak allocated GPU memory `487,463,424 / 487,471,616` bytes, reserved `813,694,976 / 836,763,648` bytes, complete 38-volume Validation loop times `508.611 / 458.372` s, and summed sliding-window function times `315.114 / 277.928` s. These are different measurements from end-to-end single-case latency. Job 650490 has three paired case resource measurements, not a repeated architecture benchmark. Job 650736 reconstructed 1,430 patches and `374,865,920` sampled voxels: Background+Body account for `84.20%`; Bones-positive patches are `86.01%`, Prostate-positive patches `61.54%`. Thus simple complete non-exposure cannot explain E10 omission; reconstructed targets do not prove historical source-image byte identity. [Full exposure distributions](recognition/docs/analysis/class-prediction-collapse.md#verified-real-patch-exposure-job-650736).

**Frozen E20 acceptance gates.** Patient-macro Dice gates and pooled FP/Recall gates are deliberately different. Thresholds were fixed in the [E20 configuration](recognition/hipmri_3d_improved_unet_hard/prostate_fp_e20_config.json) and independently recomputed in [P4-04C](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md).

| Frozen criterion | Observed / threshold | Outcome |
|---|---|---|
| Prostate patient-macro Dice gain | .027050 / ≥.030000 | **FAIL** |
| Pooled Prostate FP reduction | .322979 / ≥.150000 | PASS |
| Pooled Prostate Recall loss | .076124 / ≤.050000 | **FAIL** |
| Maximum other-class patient-macro Dice drop | .048440 (Rectum) / ≤.020000 | **FAIL** |
| Patients without Prostate Dice regression | 4/6 / ≥4/6 | PASS |
| Minimum TP-positive volume coverage, classes 2–5 | 38/38 / 38/38 | PASS |
| Treatment Prostate patient-macro gain over V3 | +.227635 / >0 | PASS |
| Bones+Rectum+Prostate patient-macro sum gain over V3 | +.428920 / >0 | PASS |

At E20 the patient-level Prostate Dice Treatment−Control changes were B040 `+.028194`, J026 `+.051247`, K008 `−.022817`, K018 `−.019811`, M030 `+.002517`, S022 `+.122968` ([P4-04C](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md)). E18 pooled Prostate FP rose sharply in both arms (759,770/621,211). Later improvement in one metric is not a monotonic learning trend or overall model improvement.

### 1.6 Observed conclusions and source discrepancies

Standard and V1 omitted Rectum/Prostate throughout E10 full-volume Validation; V1 improved Body/Bladder, but S022 Bladder Dice fell `.6243→.2490`. Student-reported V2 weighted CE recovered Bones/Rectum/Prostate overlap on all 38 Validation volumes while some larger-class Dice declined. Its pooled Prostate Precision `.340148` and Recall `.646212` signal overprediction. Job 650924 measured class probabilities and competition on two selected Train patches and one Validation crop. It measured **neither parameter gradients nor full-volume performance**, and cannot establish the mechanism. V3 E12's protected minority patient-macro selection did not improve every class: pooled Prostate Dice `.426721` is below V2 `.445695`. V4 E20 Treatment had fewer FP and higher Prostate Precision/Dice, but it lost Recall and Rectum performance; three required gates failed. V3 remains protected, not frozen.

Do not silently reconcile these evidence limits: Standard full Validation names `standard_unet_last.pt`, whereas paired-case JSON names `standard_unet_epoch10.pt`; the old full Validation log lacks a binary hash proving the two files contain identical weights. The original E10 training files omit Standard E1–3 and V1 E1. V2 original per-volume matrices and V3/V4 checkpoint binaries are not packaged locally. The P4-04D Rangpur hash receipts corroborate **file identity**, not local inspection of optimiser/RNG/model payloads; P4-04E closes a missing exporter-source provenance issue, not that payload gap. [P4-04D](research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/P4-04D_provenance_20261014/P4-04D_PROVENANCE_ADDENDUM.md), [P4-04E](research_archive/08_validation_and_independent_audit/COMP3710_P4-04E_20261014/V4_EVIDENCE_CLOSURE.md), [P4-05A](research_archive/12_handover_and_pretest_readiness/COMP3710_P4-05A_20261014/P4-05A_READINESS_AUDIT.md).

## Part 2 — Open Questions, Possible Causes and Tutor Consultation

### 2.1 Inventory of unresolved issues

**A. Training duration and optimisation.** The partial E10 Train losses and the V4 E14–E20 Prostate trajectory are non-monotonic; 10, 12 and 20 epochs do not establish convergence. More V3 training needs a predeclared patient-level Validation hypothesis, a compute budget and a stop rule, rather than merely available GPU time. Repeated observation of the same six patients raises selection bias. Define the primary metric, meaningful change, patience and allowed Validation checks before any continuation. LR `.001` has no controlled scheduler comparison. A larger batch changes gradient variance, memory and usually updates per epoch; extra epochs at batch one add 143 updates each. They are not interchangeable. Existing A100 memory and timing numbers are run-specific, and complete V3/V4 costs need journal/Slurm aggregation.

**B. Dataset and preprocessing.** The patient-disjoint 26/6/6 split prevents longitudinal leakage between partitions, but 143 Train volumes represent only 26 independent patients and six Validation patients yield uncertain patient-macro estimates. Every Train volume contained all six labels; 61.54% of reconstructed patches contained Prostate and 86.01% Bones. Nevertheless, patch organ fraction, boundary coverage and patient weighting may be inadequate. The ~1.68×1.68×1.56 mm grid and `64³` context were not ablated. Physically plausible augmentation, crop size, branch probability and patient-balanced sampling are open hypotheses. Audit Train/existing Validation counts, geometry, low-quality cases and annotation definitions without opening held-out Test labels. Conditional mask-unit inference must not be mistaken for verified clinical labels.

**C. Architecture.** Standard→V1 changed residual units and deep supervision together, so their individual effects are unknown. Whether ~5.65M/5.73M parameters are too small, too large or insufficiently contextual for 26 patients is untested. V2's improvement with unchanged architecture weakens an architecture-only explanation without ruling out context limits. Attention or extra multiscale components cost compute and raise overfitting/multiple-comparison risk. Analyse class-conditioned errors, boundaries and sampling before architectural complexity.

**D. Objective and optimisation mechanism.** E10 hard-label class omission is observed; Job 650736 shows exposure, and Job 650924 shows selected-patch probability suppression and competing predictions. Neither estimates per-component parameter-gradient strength. V2 inverse-square-root weighted CE shifted the class trade-off; its student-recorded Prostate prediction count 476,872 exceeds GT 251,012. V4 raised the class-5 soft-FP coefficient .5→.7, reducing E20 pooled FP by 32.30% while worsening Recall by .076124 and Rectum patient-macro Dice by .048440. Before another loss change, a fixed Train-only no-update probe could measure CE/Dice parameter-gradient norms and directions. A later predeclared experiment should change one coefficient and retain recall/other-class safeguards. Probability values alone cannot attribute gradients or establish the loss as root cause.

**E. Evaluation and scientific validity.** Prostate Dice balances overlap but under- and over-contouring may carry different consequences; report Recall, Precision, volume error, other structures and patient variability as well. Pooled metrics weight voxels and repeated Weeks, while patient-macro gives each patient one vote. Both are valid for different estimands. Background/Body should not silently dominate a small-organ model-selection score; confirm course/clinical weighting with the tutor. Four purposively selected cases and sparse Z panels cannot establish apex/base errors, surface distances or population rates. HD95/ASSD/surface Dice, contiguous slices and expert review need predeclared geometry, outlier and case-selection rules. Validation reuse makes it increasingly developmental. Freeze checkpoint identity, metric rules, preprocessing, driver and output path before a separately authorised one-time Test.

**F. Engineering and reproducibility.** The V3 C journal and student hash receipts identify E12, but the original `selection.json` fields and checkpoint model/optimiser/RNG payload have not been independently inspected locally. The original [`predict.py`](recognition/hipmri_3d_improved_unet_hard/predict.py) Improved route is E10-specific. The workspace also contains an untracked [`predict_v3.py`](recognition/hipmri_3d_improved_unet_hard/predict_v3.py) protected adapter and [synthetic tests](recognition/hipmri_3d_improved_unet_hard/test_prediction_v3.py); their existence does **not** establish remote original-checkpoint/real-MRI acceptance. Verify receipt and payload, strict state/source/config compatibility, six-class mapping, affine/spacing, `64³`/stride-32 mean-softmax parity, repeatability and fail-closed behaviour. Original E10 and V4 journals can be independently recomputed; V2 detailed records and some remote resources remain unavailable locally. Freeze environment versions and paths. P4-05A says **NOT READY TO FREEZE**. Treat course one-hot, metric and deliverable requirements as tutor questions unless confirmed by the official specification.

### 2.2 Hypotheses and discriminating investigations

Every investigation below is **proposed, not executed**. Real-data analysis is limited to Train and already-used Validation; none should decode the Test set.

| Category / observed problem | Supporting evidence | Possible cause | Counterevidence / uncertainty | Diagnostic investigation | Expected information | Cost / risk | Priority |
|---|---|---|---|---|---|---|---|
| Engineering: final V3 path unaccepted | P4-05A; old E10 loader | Checkpoint schema/source-guard mismatch | V3 C stage-specific inference did succeed | Read-only original payload/receipt strict-load, synthetic and authorised prior-Validation parity | Separate engineering failure from segmentation quality | Low compute; remote originals required | Critical |
| Evaluation: pooled value hides patient regressions | S022 Bladder; E20 K008/K018 | Longitudinal weighting or anatomical heterogeneity | Only six Validation patients | Existing 38-case patient table and leave-one-patient-out sensitivity | Robustness of primary selection metric | Low compute; further Validation tuning risk | Critical |
| Loss: weak/competing minority signal | E10 omission, V2 weighted-CE recovery | CE/Dice gradient magnitude or direction | Job 650736 documents exposure; Job 650924 has no gradients | Fixed Train-patch separate component backward passes, no optimiser step | Distinguish objective mechanism from simple non-exposure | Moderate; not a historical gradient replay | High |
| Sampling: inadequate organ context | `64³`; Rectum in 777/1,430 patches | Truncated organ or patient sampling bias | Prostate branch ensures at least one voxel; Bones exposure high | Per-patient Train-patch fraction, truncation and spatial coverage audit | Whether sampler or context should change | Low–moderate audit; retraining costly | High |
| Optimisation: unstable E12–E20 behaviour | E18 FP spike; E10 losses non-monotonic | LR, stochasticity or overfitting | No multi-seed result or full local V2 curve | Summarise recorded epoch loss/Validation/LR, then predeclare any controlled continuation | Distinguish trend from stochastic fluctuation | High A100 and selection bias | High |
| Data: spacing or annotation detail | Resampling; unknown mask unit | Small-boundary detail or label inconsistency | All 211 audited headers aligned; labels survived resampling | Train/prior-Validation original vs resampled boundary review and annotation-policy check | Whether data governance comes first | Expert time; preserve frozen truth | High |
| Architecture: capacity/context | Prostate remains difficult; `64³` context | Receptive field or capacity | Same architecture recovered some classes in V2 | Only after diagnostics, single-component architecture or patch ablation | Net architecture contribution | High GPU/multiple comparisons | Later |
| Loss: FP versus Recall | E20 −32.30% FP, −.076124 Recall, Rectum −.048440 | Higher alpha favours conservative predictions | E14/E16 effects were inconsistent | Fixed-model probability/calibration diagnostics; any new alpha trial predeclared | Separate threshold from training-objective effects | Moderate; Validation tuning risk | High |
| Boundary: apex/base failures | Only selected sparse Z panels | Local anatomical position or jagged contours | No established apex/base labels | Expert-labelled slices and specified HD95/ASSD/surface inspection | Nature of clinical errors | Expert/geometry work | Later |

### 2.3 Decision framework and next investigation order

| Direction | Current evidence and one variable to change | Support / rejection criterion | Cost and timing |
|---|---|---|
| Audit data and annotations | Geometry/label ranges pass; clinical contour quality unknown. Hold model fixed and inspect Train/prior Validation. | Repeated mislabelling or abnormal spacing supports correction; consistent labels weaken it. | Expert time; **now**. |
| Change patch sampling | Prostate 61.54% and Bones 86.01% positive; whole-organ context unknown. Change one branch probability or patient weight. | Predeclared patient-macro minority and coverage gains without harm; otherwise reject. | Retraining; **after audit**. |
| Introduce augmentation | None in current production pipeline; generalisation unknown. Add one physically sensible transform policy. | Stable independent-patient gain with intact label geometry; otherwise revert. | Moderate/high GPU; **later**. |
| Change LR or schedule | Fixed `.001`, E18 volatility; no LR ablation. Change one LR/schedule factor. | Sustained multi-checkpoint/patient gain on predeclared metric; otherwise reject. | High GPU; **first review logs**. |
| Increase epochs | V3 already +2 and V4 each +8 with non-monotonic outcomes. Change budget only under a stop rule. | Sustained patient improvement with protected other classes; otherwise stop. | High GPU/Validation bias; **not immediately**. |
| Adjust batch size | Batch one fits 3D patches; larger-batch memory unknown. Explicitly preserve or report changed update count. | Quality or throughput benefit without conflated update budgets. | High VRAM; **later**. |
| Tune weighted CE/Dice/Tversky | V2 recovery and V4 trade-off observed. First measure no-update Train component gradients, then vary one coefficient. | Joint FP/Recall/other-class gates pass; otherwise reject. | Moderate/high; **after diagnosis**. |
| Modify architecture | V1 components changed together; V2 improved with architecture fixed. Ablate one component before attention. | Patient-level benefit under controlled compute; otherwise avoid complexity. | High; **last**. |
| Improve selection criteria | V3 protected and V4 rule reproducible, yet no formal freeze. Clarify assignment/clinical priorities. | A fixed, recomputable rule robust to classes and cases. | Low compute; **now**. |
| Improve evaluation and failure diagnostics | Confusion counts and four panels exist; surface/uncertainty/apex-base analysis missing. | A repeatable error pattern discriminates FP/FN or boundary hypotheses. | Low–moderate; **now**. |

**Cautious order:** (1) close P4-05A engineering and read-only receipt/payload gaps; (2) fix the evaluation target with the tutor and analyse existing patient/class failures; (3) audit Train patches, labels and spacing; (4) perform a fixed-Train-patch CE/Dice gradient diagnostic; (5) only then consider one predeclared, bounded experimental change. This document runs no experiment, selects no new checkpoint and freezes no model.

### 2.4 Tutor consultation: the ten first questions

The first five fit a short meeting. Each question connects to a recorded result and a specific decision; course requirements are asked, not assumed.

| Priority | Topic / uncertainty | Direct question to ask (short spoken form in parentheses) | Evidence, why it matters and decision affected |
|---|---|---|---|
| Critical 1 | Is further V3 training justified? | “Given the non-monotonic E10 loss and the E14–E20 validation trajectory, what evidence would justify any further training from the protected V3 E12 checkpoint?” (“Would more epochs be justified?”) | E18 FP spike; whether to stop or set a new budget. |
| Critical 2 | Diagnose sampling before loss? | “With positive prostate patches in 61.54% of reconstructed training crops and V2 recovery after weighted CE, which diagnostic would best separate sampling from objective effects?” | Job 650736 and V2; order of Train audit and gradient probe. |
| Critical 3 | Interpret the FP–Recall trade-off | “Would you accept the E20 treatment’s 32.30% pooled prostate FP reduction when pooled recall falls by 0.076 and rectum patient-macro Dice falls by 0.048?” | Three failed frozen gates; interpretation of clinical and joint criteria. |
| Critical 4 | Adequacy of E12 selection | “Is the V3 E12 patient-macro minority-class selection rule adequate for this assessment, given six reused validation patients and a lower pooled prostate Dice than V2?” | V3 versus V2; keep candidate or require further diagnostics. |
| Critical 5 | Freeze and Test boundary | “What evidence and approval do you expect before we freeze the exact V3 checkpoint and run the held-out Test once?” | P4-05A not ready; final driver and freeze receipt. |
| High 6 | Official metric aggregation | “Should the report’s primary metric be pooled, volume-macro or patient-macro Dice, and are there course-specific class or patient weighting requirements?” | These answer different questions; choose reporting and selection rule. |
| High 7 | Class-specific failure evaluation | “For the prostate and rectum errors, would per-patient precision/recall plus surface distance or expert-reviewed apex/base slices be the most useful evidence?” | Existing panels have sparse Z slices; choose diagnostic scope. |
| High 8 | Early stopping | “How would you predefine early stopping here without repeatedly tuning to the same six validation patients?” | E10/V4 non-monotonicity; patience and minimum benefit. |
| High 9 | Expected project depth | “Do the Baseline–V4 controls, independent E20 audit and explicit rejected treatment meet the expected experimental depth, or is a specific additional comparison required?” | Determines whether more experimentation is required by the assessment. |
| High 10 | One-hot guidance and submission | “Is using integer masks with cross-entropy and soft-Dice acceptable under the assignment’s one-hot guidance, and which exact figures or evidence must accompany the final report?” | Current long-mask implementation and external figures; submission preparation. |

**Optional technical follow-ups:**

| Priority / topic | Direct question | Evidence and decision |
|---|---|---|
| Optional / gradients | “Would a no-update, fixed-training-patch CE-versus-Dice gradient audit be a useful next diagnostic?” | Job 650924 measured probabilities, not gradients; decide mechanism probe. |
| Optional / longitudinal weighting | “Should longitudinal weeks be weighted equally within each patient or should every volume count equally during training?” | 26 Train patients/143 scans; sampling design. |
| Optional / spacing and context | “Is the training-only median spacing and a 64-cube patch defensible for prostate boundaries, or should we audit context before changing either?” | Existing spacing/patch policy; scope of data audit. |
| Optional / uncertainty | “Would patient-level bootstrap or leave-one-patient-out sensitivity be appropriate with only six validation patients?” | Two E20 patient regressions; uncertainty reporting. |
| Optional / augmentation | “Which physically plausible MRI augmentations would you permit without corrupting label geometry?” | No production augmentation; design of one later change. |
| Optional / architecture | “Would you value a residual-only or deep-supervision-only ablation before considering attention?” | V1 changed both; compute priority. |
| Optional / single Test | “After protocol freeze, should the held-out Test be evaluated exactly once, and how should we report an unexpected failure without retuning?” | Protect independent final evaluation. |

## Part 3 — Complete Annotated Project File Tree

The tree below is generated from the actual project directory. The corresponding Chinese-comment version is in [W11-ZH.md](W11-ZH.md). `research_archive/` contains local historical evidence and is excluded from Git; active source and archived snapshots are labelled accordingly. `.git` internals, virtual environments, `__pycache__`, `.pytest_cache` and generated dependency caches are omitted.

### 3.1 English annotated directory tree

```text
Final project/
├── .vscode/
│   └── settings.json  # VS Code workspace settings
├── recognition/
│   ├── docs/
│   │   ├── analysis/
│   │   │   └── class-prediction-collapse.md  # E10 omission, patch and probability analysis
│   │   ├── evidence/
│   │   │   ├── failure-cases.md  # Raw E10 logs and paired failure evidence
│   │   │   └── traceability.md  # Report-to-code evidence register
│   │   ├── experiments.md  # Consolidated Baseline–V4 research history
│   │   └── FEASIBILITY_REPORT.md  # Project feasibility and remaining requirements
│   ├── figures/
│   │   └── e10_recorded_training_losses.png  # Plot of actual E10 epoch losses
│   ├── hipmri_3d_improved_unet_hard/
│   │   ├── diagnostics/
│   │   │   ├── __init__.py  # Diagnostics package marker
│   │   │   ├── checkpoint_probabilities.py  # Fixed-patch checkpoint probability probe
│   │   │   ├── class_collapse.py  # Train patch-exposure reconstruction
│   │   │   ├── test_checkpoint_probabilities.py  # Synthetic tests: probability probe
│   │   │   └── test_class_collapse.py  # Synthetic tests: patch exposure
│   │   ├── __init__.py  # Python package marker
│   │   ├── ai.md  # AI usage and historical execution provenance
│   │   ├── benchmark_k019_preprocessing_cache.py  # K019 cache benchmark, not production caching
│   │   ├── dataset.py  # MRI/mask discovery and frozen patient split
│   │   ├── experiment_prostate_fp.py  # Historical pre-E20 V4 draft runner
│   │   ├── experiment_prostate_fp_e20.py  # Frozen V4 paired experiment and decision
│   │   ├── improved_modules.py  # Shared residual/deep-supervised ImprovedUNet3D
│   │   ├── improved_v2_config.json  # V2 exposure counts and CE class weights
│   │   ├── improved_v3_config.json  # V3 continuation and selection thresholds
│   │   ├── intensity_normalization.py  # MRI nonzero percentile and z-score policy
│   │   ├── modules.py  # StandardUNet3D architecture
│   │   ├── normalization_config.json  # MRI percentile and standardisation settings
│   │   ├── patch_dataset.py  # Aligned deterministic patch extraction
│   │   ├── patch_dataset_config.json  # 64-cube crop and foreground-sampling policy
│   │   ├── predict.py  # E10 checkpoint prediction, metrics and panels
│   │   ├── predict_v3.py  # New V3 E12 protected inference adapter; remote acceptance pending
│   │   ├── profile_pipeline.py  # Preprocessing throughput and memory profiler
│   │   ├── prostate_fp_config.json  # Historical V4 draft settings
│   │   ├── prostate_fp_e20_config.json  # Frozen V4 treatment and acceptance gates
│   │   ├── README.md  # Five-version implementation and evidence guide
│   │   ├── resampling.py  # Physical-grid MRI/mask resampling
│   │   ├── resampling_config.json  # Target spacing/interpolation/allocation policy
│   │   ├── run_improved_v3_split.py  # V3 A/B training and C selection orchestration
│   │   ├── smoke_standard_unet.py  # Standard GPU smoke check
│   │   ├── submit_v3_split.sh  # V3 Slurm stage submission helper
│   │   ├── test_dataset.py  # Synthetic tests: dataset discovery/split
│   │   ├── test_improved_integration.py  # Synthetic tests: Improved integration
│   │   ├── test_improved_unet.py  # Synthetic tests: Improved network
│   │   ├── test_improved_v2.py  # Synthetic tests: V2 objective/training
│   │   ├── test_improved_v3.py  # Synthetic tests: V3 continuation
│   │   ├── test_improved_v3_split.py  # Synthetic tests: V3 stage orchestration
│   │   ├── test_intensity_normalization.py  # Synthetic tests: MRI normalisation
│   │   ├── test_patch_dataset.py  # Synthetic tests: crop sampling
│   │   ├── test_prediction.py  # Synthetic tests: E10 prediction
│   │   ├── test_prediction_v3.py  # Synthetic tests: V3 inference admission
│   │   ├── test_prostate_fp_e20.py  # Synthetic tests: V4 E20 controls
│   │   ├── test_resampling.py  # Synthetic tests: grid interpolation
│   │   ├── test_standard_unet.py  # Synthetic tests: Standard network
│   │   ├── test_training.py  # Synthetic tests: Standard training/loss
│   │   ├── test_validation.py  # Synthetic tests: full-volume metrics
│   │   ├── test_volume_io.py  # Synthetic tests: NIfTI loading/geometry
│   │   ├── train.py  # Standard E10 training and checkpointing
│   │   ├── train_improved.py  # Improved V1 training and auxiliary loss
│   │   ├── train_improved_v2.py  # V2 weighted-CE training and state checks
│   │   ├── train_improved_v3.py  # V3 restored-state continuation
│   │   ├── validate.py  # Sliding-window inference and confusion metrics
│   │   ├── validate_improved.py  # Improved V1 validation adapter
│   │   ├── validate_improved_v2.py  # V2 validation adapter and strict guards
│   │   └── volume_io.py  # Paired NIfTI array and geometry validation
│   ├── .gitattributes  # Recognition Git attribute rules
│   ├── FINAL-REPORT.md  # Technical report and Validation interpretation
│   ├── README.md  # Recognition project navigation and current decision
│   └── requirements.txt  # Python runtime dependency ranges
├── research_archive/
│   ├── 07_model_development_snapshots/
│   │   ├── P5-08_superseded_active_docs_2026-10-15/
│   │   │   ├── recognition/
│   │   │   │   ├── CLASS_COLLAPSE_DIAGNOSIS.md  # Archived development snapshot: Class Prediction Collapse Diagnosis
│   │   │   │   ├── Failiure-analysis.md  # Archived development snapshot: Failiure analysis
│   │   │   │   ├── FAILURE_CASE_EVIDENCE.md  # Archived development snapshot: Failure Case Evidence Manifest
│   │   │   │   ├── IMPROVED_V3_EXPERIMENT.md  # Archived development snapshot: Improved V3: bounded V2 continuation
│   │   │   │   ├── IMPROVED_V3_SPLIT_EXECUTION.md  # Archived development snapshot: V3 split execution: prepared, not remotely executed
│   │   │   │   └── REPORT_CODE_TRACEABILITY.md  # Archived development snapshot: HipMRI Report-to-Code Traceability
│   │   │   ├── PROVENANCE.md  # Archived development snapshot: P5-08 superseded active documentation originals
│   │   │   └── summary.md  # Archived development snapshot: project structure and file responsibilities
│   │   └── v4_e14_pre_e20_draft_2026-10-13/
│   │       ├── experiment_prostate_fp.py  # Archived development snapshot script/test: experiment prostate fp
│   │       ├── prostate_fp_config.json  # Archived development snapshot structured record/manifest: prostate fp config
│   │       └── PROVENANCE.md  # Archived development snapshot: V4 E14 pre-E20 draft snapshot ? provenance
│   ├── 08_validation_and_independent_audit/
│   │   ├── COMP3710_P4-04A_readiness_20261014/
│   │   │   └── READINESS_AUDIT.md  # Archived independent Validation audit: P4-04A E20 readiness audit — 14 October 2026
│   │   ├── COMP3710_P4-04C_audit_20261014/
│   │   │   ├── P4-04D_provenance_20261014/
│   │   │   │   └── P4-04D_PROVENANCE_ADDENDUM.md  # Archived independent Validation audit: P4-04D — Provenance closure addendum (14 October 2026)
│   │   │   ├── reconstructed_tests/
│   │   │   │   ├── analyze_e20.py  # Archived independent Validation audit script/test: analyze e20
│   │   │   │   ├── analyze_validation.py  # Archived independent Validation audit script/test: analyze validation
│   │   │   │   ├── export_results.py  # Archived independent Validation audit script/test: export results
│   │   │   │   ├── inventory.json  # Archived independent Validation audit structured record/manifest: inventory
│   │   │   │   ├── package_evidence.py  # Archived independent Validation audit script/test: package evidence
│   │   │   │   ├── run_twice.py  # Archived independent Validation audit script/test: run twice
│   │   │   │   ├── test_analysis.py  # Synthetic tests: analysis
│   │   │   │   ├── test_e20.py  # Synthetic tests: e20
│   │   │   │   └── test_export_results.py  # Synthetic tests: export results
│   │   │   ├── audit_e20_independent.py  # Archived independent Validation audit script/test: audit e20 independent
│   │   │   ├── audit_exports.py  # Archived independent Validation audit script/test: audit exports
│   │   │   ├── audit_provenance.py  # Archived independent Validation audit script/test: audit provenance
│   │   │   ├── export_audit.json  # Archived independent Validation audit structured record/manifest: export audit
│   │   │   ├── fix_exports.py  # Archived independent Validation audit script/test: fix exports
│   │   │   ├── fix_provenance.py  # Archived independent Validation audit script/test: fix provenance
│   │   │   ├── INDEPENDENT_AUDIT.md  # Archived independent Validation audit: P4-04C independent E20 evidence audit — 14 October 2026
│   │   │   ├── independent_results.json  # Archived independent Validation audit structured record/manifest: independent results
│   │   │   ├── provenance_audit.json  # Archived independent Validation audit structured record/manifest: provenance audit
│   │   │   ├── render_reports.py  # Archived independent Validation audit script/test: render reports
│   │   │   ├── SCIENTIFIC_EN.md  # Archived independent Validation audit: V4 E20 scientific interpretation — English (14 October 2026)
│   │   │   └── SCIENTIFIC_ZH.md  # Archived independent Validation audit: Chinese V4 E20 scientific interpretation
│   │   └── COMP3710_P4-04E_20261014/
│   │       ├── reproducibility_source/
│   │       │   ├── analyze_e20.py  # Archived independent Validation audit script/test: analyze e20
│   │       │   ├── analyze_validation.py  # Archived independent Validation audit script/test: analyze validation
│   │       │   ├── export_results.py  # Archived independent Validation audit script/test: export results
│   │       │   ├── inventory.json  # Archived independent Validation audit structured record/manifest: inventory
│   │       │   ├── README.md  # Archived independent Validation audit: P4-04B isolated E20 evidence audit (14 October 2026)
│   │       │   ├── run_twice.py  # Archived independent Validation audit script/test: run twice
│   │       │   ├── SHA256SUMS_LOCAL.txt  # Archived independent Validation audit log/checksum: SHA256SUMS LOCAL
│   │       │   ├── test_analysis.py  # Synthetic tests: analysis
│   │       │   ├── test_e20.py  # Synthetic tests: e20
│   │       │   └── test_export_results.py  # Synthetic tests: export results
│   │       ├── audit_e20_independent.py  # Archived independent Validation audit script/test: audit e20 independent
│   │       ├── audit_exports.py  # Archived independent Validation audit script/test: audit exports
│   │       ├── audit_provenance.py  # Archived independent Validation audit script/test: audit provenance
│   │       ├── build_closure.py  # Archived independent Validation audit script/test: build closure
│   │       ├── cpu_test_stdout.txt  # Archived independent Validation audit log/checksum: cpu test stdout
│   │       ├── export_audit.json  # Archived independent Validation audit structured record/manifest: export audit
│   │       ├── independent_results.json  # Archived independent Validation audit structured record/manifest: independent results
│   │       ├── provenance_audit.json  # Archived independent Validation audit structured record/manifest: provenance audit
│   │       ├── SHA256SUMS.txt  # Archived independent Validation audit log/checksum: SHA256SUMS
│   │       ├── SUPPLEMENT_SOURCE_MANIFEST.json  # Archived independent Validation audit structured record/manifest: SUPPLEMENT SOURCE MANIFEST
│   │       ├── V4_DEPENDENCY_AUDIT.md  # Archived independent Validation audit: V4 analysis dependency audit
│   │       ├── V4_EVIDENCE_CLOSURE.md  # Archived independent Validation audit: V4 evidence closure — P4-04E (14 October 2026)
│   │       ├── V4_P4-04E_SUPPLEMENT_source_only.zip  # Archived independent Validation audit evidence ZIP: V4 P4 04E SUPPLEMENT source only
│   │       ├── V4_PROVENANCE_MATRIX.md  # Archived independent Validation audit: V4 provenance matrix
│   │       ├── V4_REPRODUCIBILITY_MANIFEST.json  # Archived independent Validation audit structured record/manifest: V4 REPRODUCIBILITY MANIFEST
│   │       └── V4_TEST_RESULTS.md  # Archived independent Validation audit: P4-04E CPU-only verification
│   ├── 09_rangpur_original_evidence/
│   │   ├── P4-03B-evidence-20261013/
│   │   │   ├── P4-03B_INDEPENDENT_AUDIT.md  # Archived original Rangpur evidence: P4-03B independent E14/E16 Validation audit
│   │   │   └── p4_03b_evidence_20261013T134448164139Z.zip  # Archived original Rangpur evidence evidence ZIP: p4 03b evidence 20261013T134448164139Z
│   │   └── P4-04B-evidence-20261014/
│   │       └── p4_04b_e20_evidence_retry1.zip  # Archived original Rangpur evidence evidence ZIP: p4 04b e20 evidence retry1
│   ├── 10_scientific_documentation/
│   │   ├── COMP3710_P4-03C_docs_20261014/
│   │   │   ├── historical_snapshots/
│   │   │   │   ├── README.md  # Archived scientific document: Pattern Analysis
│   │   │   │   ├── recognition__CLASS_COLLAPSE_DIAGNOSIS.md  # Archived scientific document: Class Prediction Collapse Diagnosis
│   │   │   │   ├── recognition__FINAL-REPORT.md  # Archived scientific document: HipMRI 3D U-Net Final Report
│   │   │   │   ├── recognition__hipmri_3d_improved_unet_hard__ai.md  # Archived scientific document: Verified Command Reference and AI-Assisted Development Record
│   │   │   │   ├── recognition__hipmri_3d_improved_unet_hard__README.md  # Archived scientific document: HipMRI 3D Improved U-Net Segmentation
│   │   │   │   ├── recognition__IMPROVED_V3_EXPERIMENT.md  # Archived scientific document: Improved V3: bounded V2 continuation
│   │   │   │   ├── recognition__IMPROVED_V3_SPLIT_EXECUTION.md  # Archived scientific document: V3 split execution: prepared, not remotely executed
│   │   │   │   ├── recognition__README.md  # Archived scientific document: Recognition Project
│   │   │   │   └── recognition__REPORT_CODE_TRACEABILITY.md  # Archived scientific document: HipMRI Report-to-Code Traceability
│   │   │   ├── AI_USAGE_DISCLOSURE_ADDENDUM_20261014.md  # Archived scientific document: AI Usage Disclosure addendum — P4-03C, 14 October 2026
│   │   │   ├── BASELINE_V4_COMPARISON.md  # Archived scientific document: Baseline–V4 comparison (14 October 2026)
│   │   │   ├── CONSISTENCY_AUDIT.md  # Archived scientific document: P4-03C final consistency and provenance audit — 14 October 2026
│   │   │   ├── DOCUMENT_SYNC_MANIFEST.md  # Archived scientific document: P4-03C documentation synchronisation manifest — 14 October 2026
│   │   │   ├── EVIDENCE_DATA.json  # Archived scientific document structured record/manifest: EVIDENCE DATA
│   │   │   ├── FEASIBILITY_ADDENDUM_20261014.md  # Archived scientific document: Feasibility-report addendum — 14 October 2026
│   │   │   ├── finalize_p403c.py  # Archived scientific document script/test: finalize p403c
│   │   │   ├── GPU_RESOURCE_LIMITATIONS_ADDENDUM_20261014.md  # Archived scientific document: GPU and research limitations addendum — 14 October 2026
│   │   │   ├── P4-03B_AUDIT_SOURCE.md  # Archived scientific document: P4-03B independent E14/E16 Validation audit
│   │   │   ├── README.md  # Archived scientific document: P4-03C documentation workspace
│   │   │   ├── README_ADDENDUM_20261014.md  # Archived scientific document: Project README addendum — P4-03C, 14 October 2026
│   │   │   ├── render_addenda.py  # Archived scientific document script/test: render addenda
│   │   │   ├── render_p403c.py  # Archived scientific document script/test: render p403c
│   │   │   ├── render_reports.py  # Archived scientific document script/test: render reports
│   │   │   ├── RESEARCH_EXPERIMENT_LOG.md  # Archived scientific document: P4-03C research and experiment record — 14 October 2026
│   │   │   ├── RESEARCH_PROGRESS_EN.md  # Archived scientific document: HipMRI research progress — English (14 October 2026)
│   │   │   ├── RESEARCH_PROGRESS_ZH.md  # Archived scientific document: Chinese HipMRI research progress
│   │   │   ├── STATE_PACKET_20261014.md  # Archived scientific document: COMP3710 state packet — 14 October 2026
│   │   │   └── V4_INTERMEDIATE_VALIDATION.md  # Archived scientific document: V4 E14/E16 exploratory Validation (14 October 2026)
│   │   ├── COMP3710_V4_CLOSURE_20261014/
│   │   │   ├── audit_e20_independent.py  # Archived scientific document script/test: audit e20 independent
│   │   │   ├── audit_exports.py  # Archived scientific document script/test: audit exports
│   │   │   ├── audit_provenance.py  # Archived scientific document script/test: audit provenance
│   │   │   ├── build_v4_closure.py  # Archived scientific document script/test: build v4 closure
│   │   │   ├── COMP3710_STATE_PACKET_AFTER_V4.md  # Archived scientific document: COMP3710 HipMRI state packet after V4 — portable handover (14 October 2026)
│   │   │   ├── export_audit.json  # Archived scientific document structured record/manifest: export audit
│   │   │   ├── FINAL_DOCUMENT_QA.txt  # Archived scientific document log/checksum: FINAL DOCUMENT QA
│   │   │   ├── FINAL_QA_EXECUTION.txt  # Archived scientific document log/checksum: FINAL QA EXECUTION
│   │   │   ├── INDEPENDENT_AUDIT_RUN.txt  # Archived scientific document log/checksum: INDEPENDENT AUDIT RUN
│   │   │   ├── independent_results.json  # Archived scientific document structured record/manifest: independent results
│   │   │   ├── P4-03B_CPU_TESTS.txt  # Archived scientific document log/checksum: P4 03B CPU TESTS
│   │   │   ├── P4-04B_CPU_TESTS.txt  # Archived scientific document log/checksum: P4 04B CPU TESTS
│   │   │   ├── provenance_audit.json  # Archived scientific document structured record/manifest: provenance audit
│   │   │   ├── run_final_qa.py  # Archived scientific document script/test: run final qa
│   │   │   ├── SHA256SUMS.txt  # Archived scientific document log/checksum: SHA256SUMS
│   │   │   ├── V4_AI_USAGE_AND_REPRODUCIBILITY.md  # Archived scientific document: AI usage and reproducibility
│   │   │   ├── V4_CLOSURE_MANIFEST.json  # Archived scientific document structured record/manifest: V4 CLOSURE MANIFEST
│   │   │   ├── V4_COMPLETION_CHECKLIST.md  # Archived scientific document: V4 completion checklist
│   │   │   ├── V4_EVIDENCE_INDEX.md  # Archived scientific document: V4 evidence index
│   │   │   ├── V4_FINAL_QA_REPORT.md  # Archived scientific document: V4 final QA report — actual execution
│   │   │   ├── V4_FINAL_SCIENTIFIC_CONCLUSION_EN.md  # Archived scientific document: V4 final scientific conclusion — English
│   │   │   ├── V4_FINAL_SCIENTIFIC_CONCLUSION_ZH.md  # Archived scientific document: Chinese V4 final scientific conclusion
│   │   │   ├── V4_FINAL_STATUS.md  # Archived scientific document: V4 final stage status — CLOSED, 14 October 2026
│   │   │   └── V4_OPEN_ISSUES_AND_LIMITATIONS.md  # Archived scientific document: Open issues: V4 closure versus deferred work
│   │   └── COMP3710_V4_FINAL_DOCS_20261014/
│   │       ├── build_v4_final_docs.py  # Archived scientific document script/test: build v4 final docs
│   │       ├── V4_ACCEPTANCE_DECISION.md  # Archived scientific document: Frozen E20 acceptance decision
│   │       ├── V4_AI_USAGE_DISCLOSURE.md  # Archived scientific document: V4 AI-assisted development disclosure
│   │       ├── V4_CONTROL_TREATMENT_COMPARISON.md  # Archived scientific document: V4 paired Control–Treatment comparison
│   │       ├── V4_DOCUMENTATION_VALIDATION.txt  # Archived scientific document log/checksum: V4 DOCUMENTATION VALIDATION
│   │       ├── V4_EXPERIMENT_TIMELINE.md  # Archived scientific document: V4 experiment timeline and evidence status
│   │       ├── V4_FAILURE_ANALYSIS.md  # Archived scientific document: V4 E20 failure analysis
│   │       ├── V4_FINAL_REPORT_EN.md  # Archived scientific document: V4 paired Prostate false-positive experiment — final English report
│   │       ├── V4_FINAL_REPORT_ZH.md  # Archived scientific document: Chinese final V4 paired Prostate FP report
│   │       ├── V4_REPORT_INTEGRATION_GUIDE.md  # Archived scientific document: Versioned integration guide — no historical overwrite
│   │       ├── V4_REPRODUCIBILITY_AND_LIMITATIONS.md  # Archived scientific document: V4 reproducibility and limitations
│   │       ├── V4_RESEARCH_LOG_FINAL.md  # Archived scientific document: V4 final research log
│   │       └── validate_v4_docs.py  # Archived scientific document script/test: validate v4 docs
│   ├── 11_tests_debug_and_analysis/
│   │   ├── COMP3710_P4-04B_analysis_20261014/
│   │   │   ├── analyze_e20.py  # Archived historical analysis/test script/test: analyze e20
│   │   │   ├── analyze_validation.py  # Archived historical analysis/test script/test: analyze validation
│   │   │   ├── export_results.py  # Archived historical analysis/test script/test: export results
│   │   │   ├── extend_summary.py  # Archived historical analysis/test script/test: extend summary
│   │   │   ├── fix_optional.py  # Archived historical analysis/test script/test: fix optional
│   │   │   ├── IMPLEMENTATION_LOG.md  # Archived historical analysis/test: P4-04B local implementation log — 14 October 2026
│   │   │   ├── inventory.json  # Archived historical analysis/test structured record/manifest: inventory
│   │   │   ├── package_evidence.py  # Archived historical analysis/test script/test: package evidence
│   │   │   ├── READINESS_STATUS.md  # Archived historical analysis/test: P4-04B status — 14 October 2026
│   │   │   ├── README.md  # Archived historical analysis/test: P4-04B isolated E20 evidence audit (14 October 2026)
│   │   │   ├── run_twice.py  # Archived historical analysis/test script/test: run twice
│   │   │   ├── SHA256SUMS_LOCAL.txt  # Archived historical analysis/test log/checksum: SHA256SUMS LOCAL
│   │   │   ├── test_analysis.py  # Synthetic tests: analysis
│   │   │   ├── test_e20.py  # Synthetic tests: e20
│   │   │   ├── test_export_results.py  # Synthetic tests: export results
│   │   │   └── update_readme.py  # Archived historical analysis/test script/test: update readme
│   │   ├── hipmri_p4_03a_readonly_analysis_20261013/
│   │   │   ├── analyze_validation.py  # Archived historical analysis/test script/test: analyze validation
│   │   │   ├── IMPLEMENTATION_LOG.md  # Archived historical analysis/test: P4-03A implementation and verification log
│   │   │   ├── inventory.json  # Archived historical analysis/test structured record/manifest: inventory
│   │   │   ├── README.md  # Archived historical analysis/test: P4-03A: isolated intermediate Validation analysis
│   │   │   └── test_analysis.py  # Synthetic tests: analysis
│   │   └── hipmri_p4_03b_readonly_analysis_20261013/
│   │       ├── analyze_validation.py  # Archived historical analysis/test script/test: analyze validation
│   │       ├── export_results.py  # Archived historical analysis/test script/test: export results
│   │       ├── IMPLEMENTATION_LOG.md  # Archived historical analysis/test: P4-03A implementation and verification log
│   │       ├── inventory.json  # Archived historical analysis/test structured record/manifest: inventory
│   │       ├── README.md  # Archived historical analysis/test: P4-03A: isolated intermediate Validation analysis
│   │       ├── run_real_e14_e16.py  # Archived historical analysis/test script/test: run real e14 e16
│   │       ├── test_analysis.py  # Synthetic tests: analysis
│   │       └── test_export_results.py  # Synthetic tests: export results
│   ├── 12_handover_and_pretest_readiness/
│   │   └── COMP3710_P4-05A_20261014/
│   │       ├── build_p4_05a.py  # Archived pre-Test readiness script/test: build p4 05a
│   │       ├── INFERENCE_PROTOCOL_AUDIT.md  # Archived pre-Test readiness: V3 E12 inference protocol audit
│   │       ├── P4-05A_READINESS_AUDIT.md  # Archived pre-Test readiness: P4-05A read-only readiness audit
│   │       ├── PRETEST_FREEZE_CHECKLIST.md  # Archived pre-Test readiness: Pre-Test freeze readiness checklist
│   │       ├── PROPOSED_MODEL_FREEZE_RECEIPT.json  # Archived pre-Test readiness structured record/manifest: PROPOSED MODEL FREEZE RECEIPT
│   │       ├── STATE_PACKET_P4_05A.md  # Archived pre-Test readiness: COMP3710 state packet — P4-05A, 14 October 2026
│   │       └── V3_CANDIDATE_IDENTITY.md  # Archived pre-Test readiness: V3 E12 protected candidate identity
│   ├── 13_inventories/
│   │   ├── COMP3710_P5_01_AUDIT_20261014_211851_6c7e5f1f/
│   │   │   ├── build_inventory.py  # Archived file inventory script/test: build inventory
│   │   │   ├── COMP3710_DIRECTORY_TREE_C.md  # Archived file inventory: Directory tree — Source C
│   │   │   ├── COMP3710_DIRECTORY_TREE_D.md  # Archived file inventory: Directory tree — Source D
│   │   │   ├── COMP3710_GIT_REPOSITORY_AUDIT.md  # Archived file inventory: Git repository audit (read-only)
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.csv  # Archived file inventory table/export: COMP3710 MASTER FILE INVENTORY
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.json  # Archived file inventory structured record/manifest: COMP3710 MASTER FILE INVENTORY
│   │   │   ├── COMP3710_P5_01_AUDIT_SUMMARY.md  # Archived file inventory: P5-01 comprehensive read-only audit summary
│   │   │   ├── COMP3710_PROJECT_ASSET_MAP.md  # Archived file inventory: COMP3710 project asset map
│   │   │   ├── COMP3710_RANGPUR_EVIDENCE_INVENTORY.md  # Archived file inventory: Rangpur-derived evidence inventory
│   │   │   ├── COMP3710_TERMINAL_AND_EXPERIMENT_HISTORY.md  # Archived file inventory: Terminal and experiment history index
│   │   │   └── COMP3710_UNCLASSIFIED_AND_REVIEW_REQUIRED.md  # Archived file inventory: Unclassified and review-required
│   │   ├── COMP3710_P5_01_AUDIT_REFINED_20261014_212128_2fc0b2db/
│   │   │   ├── build_inventory.py  # Archived file inventory script/test: build inventory
│   │   │   ├── COMP3710_DIRECTORY_TREE_C.md  # Archived file inventory: Directory tree — Source C
│   │   │   ├── COMP3710_DIRECTORY_TREE_D.md  # Archived file inventory: Directory tree — Source D
│   │   │   ├── COMP3710_GIT_REPOSITORY_AUDIT.md  # Archived file inventory: Git repository audit (read-only)
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.csv  # Archived file inventory table/export: COMP3710 MASTER FILE INVENTORY
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.json  # Archived file inventory structured record/manifest: COMP3710 MASTER FILE INVENTORY
│   │   │   ├── COMP3710_P5_01_AUDIT_SUMMARY.md  # Archived file inventory: P5-01 comprehensive read-only audit summary
│   │   │   ├── COMP3710_PROJECT_ASSET_MAP.md  # Archived file inventory: COMP3710 project asset map
│   │   │   ├── COMP3710_RANGPUR_EVIDENCE_INVENTORY.md  # Archived file inventory: Rangpur-derived evidence inventory
│   │   │   ├── COMP3710_TERMINAL_AND_EXPERIMENT_HISTORY.md  # Archived file inventory: Terminal and experiment history index
│   │   │   ├── COMP3710_UNCLASSIFIED_AND_REVIEW_REQUIRED.md  # Archived file inventory: Unclassified and review-required
│   │   │   └── patch_builder.py  # Archived file inventory script/test: patch builder
│   │   ├── COMP3710_P5_01_DELTA_20261014_212842_f56b8873/
│   │   │   ├── AUDIT_SUMMARY_ZH.md  # Archived file inventory: Chinese P5-01 read-only delta summary
│   │   │   ├── build_delta.py  # Archived file inventory script/test: build delta
│   │   │   ├── DIRECTORY_TREE_C_PROJECT_ONLY.md  # Archived file inventory: Directory tree — Source C
│   │   │   ├── DIRECTORY_TREE_D.md  # Archived file inventory: Directory tree — Source D
│   │   │   ├── DUPLICATE_VERSION_ANALYSIS.md  # Archived file inventory: Duplicate and version analysis
│   │   │   ├── FILE_DEPENDENCY_MAP.md  # Archived file inventory: File dependency map
│   │   │   ├── GIT_READONLY_AUDIT.md  # Archived file inventory: Git repository audit (read-only)
│   │   │   ├── MASTER_FILE_INVENTORY.csv  # Archived file inventory table/export: MASTER FILE INVENTORY
│   │   │   ├── MASTER_FILE_INVENTORY.json  # Archived file inventory structured record/manifest: MASTER FILE INVENTORY
│   │   │   ├── P5_01_STATE_PACKET.md  # Archived file inventory: P5-01 state packet
│   │   │   ├── PROJECT_STAGE_MAP.md  # Archived file inventory: Project stage map
│   │   │   ├── RANGPUR_LOCAL_EVIDENCE_MAP.md  # Archived file inventory: Rangpur-derived evidence inventory
│   │   │   ├── TERMINAL_EXPERIMENT_HISTORY.md  # Archived file inventory: Terminal and experiment history index
│   │   │   └── UNKNOWN_AND_UNCLASSIFIED.md  # Archived file inventory: Unclassified and review-required
│   │   ├── COMP3710_P5_01_FINAL_20261014_212222_d215a281/
│   │   │   ├── build_inventory.py  # Archived file inventory script/test: build inventory
│   │   │   ├── COMP3710_DIRECTORY_TREE_C.md  # Archived file inventory: Directory tree — Source C
│   │   │   ├── COMP3710_DIRECTORY_TREE_D.md  # Archived file inventory: Directory tree — Source D
│   │   │   ├── COMP3710_GIT_REPOSITORY_AUDIT.md  # Archived file inventory: Git repository audit (read-only)
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.csv  # Archived file inventory table/export: COMP3710 MASTER FILE INVENTORY
│   │   │   ├── COMP3710_MASTER_FILE_INVENTORY.json  # Archived file inventory structured record/manifest: COMP3710 MASTER FILE INVENTORY
│   │   │   ├── COMP3710_P5_01_AUDIT_SUMMARY.md  # Archived file inventory: P5-01 comprehensive read-only audit summary
│   │   │   ├── COMP3710_PROJECT_ASSET_MAP.md  # Archived file inventory: COMP3710 project asset map
│   │   │   ├── COMP3710_RANGPUR_EVIDENCE_INVENTORY.md  # Archived file inventory: Rangpur-derived evidence inventory
│   │   │   ├── COMP3710_TERMINAL_AND_EXPERIMENT_HISTORY.md  # Archived file inventory: Terminal and experiment history index
│   │   │   ├── COMP3710_UNCLASSIFIED_AND_REVIEW_REQUIRED.md  # Archived file inventory: Unclassified and review-required
│   │   │   ├── verify_and_enrich.py  # Archived file inventory script/test: verify and enrich
│   │   │   └── verify_final_inventory.py  # Archived file inventory script/test: verify final inventory
│   │   └── COMP3710_P5_02_PLAN_20261014_213035_1323b32a/
│   │       ├── P5_02_APPROVAL_READY_SUMMARY_ZH.md  # Archived file inventory: Chinese P5-02 migration proposal summary
│   │       ├── P5_02_CONFLICT_AND_REVIEW_LIST.md  # Archived file inventory: Conflict and review list
│   │       ├── P5_02_DUPLICATE_PROVENANCE_MATRIX.md  # Archived file inventory: Duplicate and provenance matrix
│   │       ├── P5_02_GIT_TRACKING_POLICY.md  # Archived file inventory: Git tracking policy (proposal)
│   │       ├── P5_02_PER_FILE_MIGRATION_MANIFEST.csv  # Archived file inventory table/export: P5 02 PER FILE MIGRATION MANIFEST
│   │       ├── P5_02_PER_FILE_MIGRATION_MANIFEST.json  # Archived file inventory structured record/manifest: P5 02 PER FILE MIGRATION MANIFEST
│   │       ├── P5_02_RANGPUR_ASSET_TRACEABILITY.md  # Archived file inventory: Rangpur asset traceability
│   │       ├── P5_02_REFERENCE_BREAKAGE_RISKS.md  # Archived file inventory: Reference and provenance breakage risks
│   │       ├── P5_02_RESEARCH_DIRECTORY_DESIGN.md  # Archived file inventory: P5-02 research directory design
│   │       └── P5_02_TARGET_DIRECTORY_TREE.md  # Archived file inventory: Proposed target directory tree
│   ├── ARCHIVE_README.md  # Archived research archive: COMP3710 HipMRI local research archive
│   ├── MASTER_ARCHIVE_INDEX.md  # Archived research archive: Master archive index
│   ├── MIGRATION_VERIFICATION_REPORT.md  # Archived research archive: P5-03 migration verification report
│   ├── ORIGINAL_PATH_MAPPING.csv  # Archived research archive table/export: ORIGINAL PATH MAPPING
│   ├── RANGPUR_EVIDENCE_INDEX.md  # Archived research archive: Rangpur evidence index
│   ├── RESEARCH_STAGE_NAVIGATION.md  # Archived research archive: Research stage navigation
│   ├── TEST_AND_DEBUG_INDEX.md  # Archived research archive: Test and debug index
│   └── UNRESOLVED_DEPENDENCIES.md  # Archived research archive: Unresolved dependencies
├── .gitignore  # Git ignore rules for local evidence and generated files
├── LICENSE.txt  # Repository licence terms
├── README.md  # Main project entry and status
├── W11-EN.md  # Complete English Week 11 review and English file index
└── W11-ZH.md  # Complete Chinese Week 11 review and Chinese file index
```
