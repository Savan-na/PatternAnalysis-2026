# Failure Case Evidence Manifest

## Scope, provenance and figure permissions

This metadata-only register supports [Final Report Section 5](FINAL-REPORT.md#5-controlled-validation-comparison-and-failure-case-evidence). It records six locally inspected JSON/PNG pairs from student-run Rangpur job **650490** and two B040 PNGs from reported prediction jobs **650442** (Standard) and **650468** (Improved). No MRI arrays, prediction arrays, PNGs, checkpoints or complete experiment logs are copied into Git.

The local specification requests 3–5 multi-panel failure studies (Section 2.2, p. 5), permits README visual content while prohibiting dataset/model commits (Section 4.1, p. 13; Section 8, p. 16), and requires a comprehensive README/PDF (Section 5.3, p. 14; Section 8). No explicit HipMRI licence or course authorisation establishing public redistribution of derived MRI images was found locally. The repository Apache licence is not evidence of dataset image rights. **Figure insertion is deferred**; originals remain external. If course/dataset permissions allow local PDF use, insert the unmodified paired originals with A=Standard/B=Improved labels and the report captions, retaining all four columns, Z labels and legends. No such PDF or layout is generated in this batch.

Windows paths below are machine-local evidence locations, not portable submission assets. SHA-256 values identify the bytes inspected; they do not prove clinical annotation accuracy or independently certify remote execution.

## Common identity and validation checks

For all six JSONs, inspected `setup.model`, `setup.checkpoint`, `checkpoint_metadata`, `volumes[0].case` and `summary.status` agree with the directory/model/case assignment: Epoch **10**, **1,430** updates, frozen **Validation** split, one selected/successful volume, zero failures, status **PASS**, no Test opt-in. Both model arms use 64³ windows, stride 32 and `uniform_mean_softmax_probabilities_then_argmax`.

- Standard checkpoint: `/home/Student/s4906923/hipmri_training_runs/standard_unet_epoch10.pt`
- Standard checkpoint SHA-256 recorded in JSON: `f2a87f9c73e068719e2f1d878291116d26320be7c5176827f78537fc4fb4f9cf`
- Improved checkpoint: `/home/Student/s4906923/hipmri_training_runs/improved_unet_epoch10.pt`
- Improved checkpoint SHA-256 recorded in JSON: `4861c5d65a780af7c8ab67e59715ec37f70ca81dee314f3b80a6a26ad031c0d6`

These are JSON-recorded checkpoint identities; checkpoint binaries were not available for independent local hashing. All three pairs agree in affine, resampled XYZ shape, target spacing, preprocessing policies, mask-unit provenance, truth class counts and Z indices. MRI dtype=float32; truth/prediction dtype=uint8; predicted classes=[0,1,3]; all six have zero predicted Bones, Rectum and Prostate voxels, with those foreground classes present in truth. Coverage minimum is 1; windows=147 for M030/S022, 192 for K008.

Recorded training/runtime libraries: NumPy 2.4.6, NiBabel 5.4.2, SciPy 1.17.1, PyTorch 2.13.0+cu130, CUDA build 13.0; GPU NVIDIA A100-PCIE-40GB. Python version is not recorded in these setup fields. The subsequently downloaded original E10 logs confirm the same recorded library/GPU versions for both full-validation jobs and both training configurations; all recorded training source hashes match current file bytes. Python version and checkpoint binaries remain unavailable.

All stored per-class Dice and IoU, true/predicted voxel counts and intersections were independently recomputed locally from each 6×6 confusion matrix (truth rows, prediction columns), with agreement within 1e-12. Precision and recall below are count-derived. This is an arithmetic/content audit, not another inference execution.

## B040_Week0

**Figure 5.1.** Both PNGs were physically located and visually inspected: case title B040_Week0; identical Z=51,62,73; four columns MRI / Ground Truth / Prediction / Error Map. The PNG titles do not contain model identifiers; attribution relies on the student-supplied filenames and job records, not inspected JSON metadata.

| Model | External PNG | SHA-256 |
| --- | --- | --- |
| Standard | `C:\Users\ruiji\Downloads\B040_Week0_standard_E10.png` | `3f89a582a00cc5f0ce824b32dab5f149c0ad852a6403c3614b04f37cef168624` |
| Improved | `C:\Users\ruiji\Downloads\B040_Week0_improved_E10.png` | `98a9663b10d5acfddf36dd36cc2230fac89ed672fa9ecdc7fafa70cad3e4167b` |

B040 prediction-driver `results.json` files remain **unavailable locally**; their original Rangpur paths are not invented. The newly inspected full-validation logs independently provide B040 confusion counts: Standard/Improved Bladder DSC **0.5365188544298019 / 0.8694770976895014**, TP=44,071 / 96,525, GT=119,585, predicted=44,700 / 102,445. Count-derived recall is approximately **36.9% / 80.7%**, precision **98.6% / 94.2%**. GT Prostate=7,105; predicted Prostate=0 for both; Standard's Prostate truth row assigns 7,093 Body + 12 Bladder, Improved's 7,042 Body + 63 Bladder. These match the previously student-supplied numbers. This authenticates the full-validation numerical evidence, not the missing prediction-driver JSON or the B040 PNG model attribution. Preserve the [original B040 narrative](Failiure-analysis.md).

## M030_Week5

**Figure 5.2.** Locally authenticated paired JSONs and visually inspected original PNGs; shared Z=43,55,67; resampled XYZ=256×256×128. A=Standard, B=Improved. Model attribution is established by JSON setup and directory identity; original PNG titles name the case, not the architecture.

### Standard

- Local JSON: `C:\Users\ruiji\Downloads\failure_cases_650490\standard_M030_Week5\results.json`
- Local PNG: `C:\Users\ruiji\Downloads\failure_cases_650490\standard_M030_Week5\M030_Week5_panels.png`
- JSON SHA-256: `4711e2506e4bed6437132d5817129f43f6f0c82e46e17fde5c104f7a9d2cbf3d`
- PNG SHA-256: `6a4a4d80738d3fa6583ec78de6e375dda46ebb479b41ead3f08c08e9645de763`
- Original Rangpur output directory, recorded in `setup.output_dir`: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/standard_M030_Week5`. Driver output filename: `results.json`; original PNG path: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/standard_M030_Week5/M030_Week5_panels.png`.
- Bladder: DSC=0.42787390884948373, IoU=0.27216259004794985; TP=35929, GT=131783, predicted=36159; precision=0.9936392046240217, recall=0.2726375936198144.
- Prostate: GT=5932, predicted=0, TP=0, DSC=0, IoU=0; truth-to-prediction counts in class order 0–5: `[0,5928,0,4,0,0]`.
- Inference seconds=9.625767837045714; peak allocated GPU bytes=362597376; peak reserved GPU bytes=379584512. These are recorded single-case measurements, not GPU training costs.

### Improved

- Local JSON: `C:\Users\ruiji\Downloads\failure_cases_650490\improved_M030_Week5\results.json`
- Local PNG: `C:\Users\ruiji\Downloads\failure_cases_650490\improved_M030_Week5\M030_Week5_panels.png`
- JSON SHA-256: `806ae7c1401fd079bfcc31fb32cb56e4cb36203f563f391fcd1fb249a9e98e33`
- PNG SHA-256: `d2e80061d45c3688f5361c6b13d304b7505c6328698342905515d92ce468f183`
- Original Rangpur output directory, recorded in `setup.output_dir`: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/improved_M030_Week5`. Driver output filename: `results.json`; original PNG path: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/improved_M030_Week5/M030_Week5_panels.png`.
- Bladder: DSC=0.8515260361774891, IoU=0.7414413064649037; TP=98975, GT=131783, predicted=100682; precision=0.9830456288115055, recall=0.7510452789813558.
- Prostate: GT=5932, predicted=0, TP=0, DSC=0, IoU=0; truth-to-prediction counts in class order 0–5: `[5,5923,0,4,0,0]`.
- Inference seconds=9.464645315194502; peak allocated GPU bytes=362163200; peak reserved GPU bytes=396361728. These are recorded single-case measurements, not GPU training costs.

## S022_Week5

**Figure 5.3.** Locally authenticated paired JSONs and visually inspected original PNGs; shared Z=44,54,64; resampled XYZ=256×256×128. A=Standard, B=Improved. Model attribution is established by JSON setup and directory identity; original PNG titles name the case, not the architecture.

### Standard

- Local JSON: `C:\Users\ruiji\Downloads\failure_cases_650490\standard_S022_Week5\results.json`
- Local PNG: `C:\Users\ruiji\Downloads\failure_cases_650490\standard_S022_Week5\S022_Week5_panels.png`
- JSON SHA-256: `f6b83286ebc411caee30b1cbeea0673a62e30e203b1c456e041081262b06ac87`
- PNG SHA-256: `4e63100de538ef702085c19547f1f67fc82697efb75c560ad4f783da5c128e73`
- Original Rangpur output directory, recorded in `setup.output_dir`: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/standard_S022_Week5`. Driver output filename: `results.json`; original PNG path: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/standard_S022_Week5/S022_Week5_panels.png`.
- Bladder: DSC=0.624326737521242, IoU=0.45383359155814246; TP=10838, GT=20816, predicted=13903; precision=0.779543983312954, recall=0.5206571867794004.
- Prostate: GT=5966, predicted=0, TP=0, DSC=0, IoU=0; truth-to-prediction counts in class order 0–5: `[0,5929,0,37,0,0]`.
- Inference seconds=9.444015083834529; peak allocated GPU bytes=362597376; peak reserved GPU bytes=379584512. These are recorded single-case measurements, not GPU training costs.

### Improved

- Local JSON: `C:\Users\ruiji\Downloads\failure_cases_650490\improved_S022_Week5\results.json`
- Local PNG: `C:\Users\ruiji\Downloads\failure_cases_650490\improved_S022_Week5\S022_Week5_panels.png`
- JSON SHA-256: `b59b2d3185f1822e2d612accb0640c5f8ed4661dc49f3b1bad01f387c0e21a5d`
- PNG SHA-256: `297700c9127b176d6703b00f623637b01dd48525f0d02c4620d6b51e0bf03688`
- Original Rangpur output directory, recorded in `setup.output_dir`: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/improved_S022_Week5`. Driver output filename: `results.json`; original PNG path: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/improved_S022_Week5/S022_Week5_panels.png`.
- Bladder: DSC=0.2489808243998188, IoU=0.14219194619298095; TP=3298, GT=20816, predicted=5676; precision=0.5810429880197322, recall=0.15843581860107608.
- Prostate: GT=5966, predicted=0, TP=0, DSC=0, IoU=0; truth-to-prediction counts in class order 0–5: `[1,5951,0,14,0,0]`.
- Inference seconds=9.523250222904608; peak allocated GPU bytes=362163200; peak reserved GPU bytes=396361728. These are recorded single-case measurements, not GPU training costs.

## K008_Week0

**Figure 5.4.** Locally authenticated paired JSONs and visually inspected original PNGs; shared Z=56,70,84; resampled XYZ=268×268×128. A=Standard, B=Improved. Model attribution is established by JSON setup and directory identity; original PNG titles name the case, not the architecture.

### Standard

- Local JSON: `C:\Users\ruiji\Downloads\failure_cases_650490\standard_K008_Week0\results.json`
- Local PNG: `C:\Users\ruiji\Downloads\failure_cases_650490\standard_K008_Week0\K008_Week0_panels.png`
- JSON SHA-256: `a21ff408ea0e4e4c6099011723194236102c620acf8357d7478dd27ef65eb5f5`
- PNG SHA-256: `50cf4acf96ff6458c5e6e20a34dd4c2cd8f0cbafc242ff7c4f6fcf9ddba4b068`
- Original Rangpur output directory, recorded in `setup.output_dir`: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/standard_K008_Week0`. Driver output filename: `results.json`; original PNG path: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/standard_K008_Week0/K008_Week0_panels.png`.
- Bladder: DSC=0.5598804442317272, IoU=0.3887735861853796; TP=28660, GT=68308, predicted=34071; precision=0.8411845851310499, recall=0.41957018211629676.
- Prostate: GT=11114, predicted=0, TP=0, DSC=0, IoU=0; truth-to-prediction counts in class order 0–5: `[0,10957,0,157,0,0]`.
- Inference seconds=7.790201259078458; peak allocated GPU bytes=362597376; peak reserved GPU bytes=379584512. These are recorded single-case measurements, not GPU training costs.

### Improved

- Local JSON: `C:\Users\ruiji\Downloads\failure_cases_650490\improved_K008_Week0\results.json`
- Local PNG: `C:\Users\ruiji\Downloads\failure_cases_650490\improved_K008_Week0\K008_Week0_panels.png`
- JSON SHA-256: `5d401c38a2474302fbd93fec0520aa6d2f43bb024517f84af282449d8d6665b1`
- PNG SHA-256: `e3adce5c9ca89f00c8d4c04cab47500a5c0f840521e66c0bf6bcb6c4ede15412`
- Original Rangpur output directory, recorded in `setup.output_dir`: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/improved_K008_Week0`. Driver output filename: `results.json`; original PNG path: `/home/Student/s4906923/hipmri_prediction_runs/failure_cases_650490/improved_K008_Week0/K008_Week0_panels.png`.
- Bladder: DSC=0.6861527219583485, IoU=0.5222469410456062; TP=39438, GT=68308, predicted=46646; precision=0.8454744243879432, recall=0.5773555074076243.
- Prostate: GT=11114, predicted=0, TP=0, DSC=0, IoU=0; truth-to-prediction counts in class order 0–5: `[0,10527,0,587,0,0]`.
- Inference seconds=7.9319485030137; peak allocated GPU bytes=362163200; peak reserved GPU bytes=396361728. These are recorded single-case measurements, not GPU training costs.

## Original E10 log authentication

The student downloaded four original JSONL files to `C:\Users\ruiji\Downloads\hipmri_e10_evidence`. All bytes were read locally. The following hashes identify inspected files, not independent certification of remote execution. Job numbers come from filenames and student execution records; JSONL does not independently establish Slurm scheduler states. Earlier Section 5 integration relied on student-supplied full-run summaries because these files were then unavailable to that audit; this later audit resolves that local evidence gap.

| File | Bytes | JSON records / event counts | SHA-256 |
| --- | ---: | --- | --- |
| `standard_e10_648790.jsonl` | 176,842 | 1010: setup=1, step=1001, epoch=7, result=1 | `97f0ab29994ea07a333d4ea7f305f0d9d66507e6127128b94a2cfa66e018c206` |
| `improved_e10_649668.jsonl` | 219,622 | 1298: setup=1, step=1287, epoch=9, result=1 | `86b3c715776482d9378e1121ca419df92432268d86c407b83d9a6a8e9c02f09a` |
| `standard_val_e10_649669.jsonl` | 87,571 | 41: setup=1, volume=38, summary=1, result=1 | `55faf9ee2896b54804f1e1e6a369cb4f4c83f40e92fc4051535302e1d968e395` |
| `improved_val_fast_650393.jsonl` | 87,862 | 41: setup=1, volume=38, summary=1, result=1 | `25a917b3eff13bbbec3f104a4775558a76926c066dd7da51a73f6dc1e5550547` |

**Authentication PASS:** all nonempty lines parse as JSON objects; no blank/non-JSON lines, exact duplicate records, repeated epoch/case identities, missing required setup/final/validation-summary records or error records. Both final training records report ten completed epochs / 1,430 updates. Standard resumes from epoch 3 and records epochs 4–10 (1,001 new steps); Improved resumes from epoch 1 and records epochs 2–10 (1,287 new steps). Every recorded epoch has 143 sequential finite step losses, matching component means and total-step counters. The earlier updates are represented by resume counters, not newly recovered per-step records. Training metadata contains the same ordered 143 pairs / 26 approved patients, preprocessing policies and shared hyperparameters; differences are model identity and the documented auxiliary-loss settings. All seven Standard and nine Improved recorded source hashes match current local source bytes.

Both validation logs contain exactly one setup, 38 unique successful Validation records, one summary and one final PASS: zero failures. Patients/volume counts are B040=7, J026=8, K008=1, K018=8, M030=6, S022=8. Case order, affine, original/resampled shapes, spacing, mask-unit provenance, window counts and coverage match between architectures. Every output voxel is accounted for by its confusion matrix; all six reference classes occur in all 38 volumes. Both setups record 64³ DHW windows, stride 32, A100-PCIE-40GB, best-effort warn-only reproducibility and resampled XYZ metric coordinates. Every volume records uniform mean-softmax aggregation followed by argmax and complete window coverage. Both summaries use the same absent-class policy: both-absent=null/excluded, one-sided absence=0.

Validation checkpoint metadata identifies the correct architecture, completed epoch 10 / 1,430 updates, and training hyperparameters/versions matching its corresponding training log. Standard full validation names `/home/Student/s4906923/hipmri_training_runs/standard_unet_last.pt`; Improved names `/home/Student/s4906923/hipmri_training_runs/improved_unet_epoch10.pt`. The Standard paired-case JSONs above instead name `standard_unet_epoch10.pt`. The full-validation setup does **not** contain a checkpoint binary hash; binary equivalence across Standard names remains unverified. All six available paired-case confusion matrices match their corresponding full-validation records exactly. No checkpoint was opened or model inference repeated.

## Full-validation numerical provenance

Each pooled matrix below is independently summed from all 38 original per-volume matrices (truth rows, prediction columns, class order 0–5). Per-volume, per-patient, pooled and both macro summaries agree with independent arithmetic within 1e-12; the production `validate.metrics_from_confusion()` and `summarize()` also reproduce the stored summaries. This is an arithmetic audit, not another MRI evaluation.

**Standard pooled confusion counts:**

```text
[188109738, 7502026, 0, 262, 0, 0]
[15128929, 90562636, 0, 240159, 0, 0]
[803721, 9337945, 0, 2609, 0, 0]
[0, 842531, 0, 899520, 0, 0]
[8656, 488929, 0, 2480, 0, 0]
[0, 245764, 0, 5248, 0, 0]
```

**Improved pooled confusion counts:**

```text
[191633198, 3978585, 0, 243, 0, 0]
[5247677, 100401777, 1, 282269, 0, 0]
[180051, 9949699, 2, 14523, 0, 0]
[281, 461067, 0, 1280703, 0, 0]
[12550, 485822, 0, 1693, 0, 0]
[419, 237910, 0, 12683, 0, 0]
```

| Class | Standard DSC | Improved DSC | Standard IoU | Improved IoU |
| --- | ---: | ---: | ---: | ---: |
| Background | 0.94134160556791 | 0.9760118742343792 | 0.8891835274898908 | 0.95314764856734 |
| Body | 0.8427898257960118 | 0.9067809959985655 | 0.7282945177834638 | 0.8294595983783096 |
| Bones | 0.0 | 3.943109603265999e-07 | 0.0 | 1.9715551903359097e-07 |
| Bladder | 0.6220039283221238 | 0.7682301265834174 | 0.45138294738733115 | 0.6236799122652379 |
| Rectum | 0.0 | 0.0 | 0.0 | 0.0 |
| Prostate | 0.0 | 0.0 | 0.0 | 0.0 |

**Section 5.1 comparison:** every previous six-decimal table entry matches rounded recomputation. The correction is Improved Bones: DSC=3.943109603265999e-7 and IoU=1.9715551903359097e-7 are positive, not exact zero, although both round to 0.000000. There are two true positives and one false positive: K018_Week1 contributes one predicted/true-positive voxel; K018_Week4 contributes two predicted voxels, one true positive. This is negligible overlap, not useful Bones segmentation.

| Class | Reference voxels, both runs | Standard predicted cases / voxels / TP | Improved predicted cases / voxels / TP |
| --- | ---: | ---: | ---: |
| Background | 195,612,026 | 38/38 / 204,051,044 / 188,109,738 | 38/38 / 197,074,176 / 191,633,198 |
| Body | 105,931,724 | 38/38 / 108,979,831 / 90,562,636 | 38/38 / 115,514,860 / 100,401,777 |
| Bones | 10,144,275 | 0/38 / 0 / 0 | 2/38 / 3 / 2 |
| Bladder | 1,742,051 | 38/38 / 1,150,278 / 899,520 | 38/38 / 1,592,114 / 1,280,703 |
| Rectum | 500,065 | 0/38 / 0 / 0 | 0/38 / 0 / 0 |
| Prostate | 251,012 | 0/38 / 0 / 0 | 0/38 / 0 / 0 |

Both matrices account for 314,181,153 voxels. The comparison table uses **overall_pooled**, not macro scores. For example, Standard/Improved Bladder DSC is pooled=0.6220039283221238 / 0.7682301265834174, volume-macro=0.5451394775527015 / 0.6174746828100194, patient-macro=0.5839460841061425 / 0.7110247431543558. These aggregations answer different questions.

## Recorded training-loss trajectory

Only events present in the two original training files are listed. Means below are rounded to nine decimals and runtimes to six; source logs retain full precision. Every row contains 143 processed/expected steps and a completed-epoch marker. No absent epoch is interpolated or supplied from historical conversation values.

| Model | Completed epoch | Steps this epoch / cumulative | Mean CE | Mean Soft Dice loss | Mean total loss | Epoch runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 4 | 143 / 572 | 0.752480676 | 0.830941145 | 0.791710915 | 748.005741 |
| Standard | 5 | 143 / 715 | 0.667956288 | 0.823226513 | 0.745591402 | 737.666175 |
| Standard | 6 | 143 / 858 | 0.721922620 | 0.802125504 | 0.762024063 | 768.150368 |
| Standard | 7 | 143 / 1001 | 0.651949921 | 0.781117420 | 0.716533669 | 724.115506 |
| Standard | 8 | 143 / 1144 | 0.646269452 | 0.772173030 | 0.709221243 | 721.428806 |
| Standard | 9 | 143 / 1287 | 0.687572608 | 0.760931140 | 0.724251873 | 730.191612 |
| Standard | 10 | 143 / 1430 | 0.628552221 | 0.769902519 | 0.699227372 | 717.079013 |
| Improved | 2 | 143 / 286 | 0.802050968 | 0.818914382 | 0.810482680 | 781.576326 |
| Improved | 3 | 143 / 429 | 0.655317520 | 0.798339331 | 0.726828427 | 756.151234 |
| Improved | 4 | 143 / 572 | 0.618250502 | 0.775926834 | 0.697088673 | 752.753943 |
| Improved | 5 | 143 / 715 | 0.515768838 | 0.773786318 | 0.644777578 | 757.646131 |
| Improved | 6 | 143 / 858 | 0.622855341 | 0.764035690 | 0.693445515 | 761.757517 |
| Improved | 7 | 143 / 1001 | 0.513023533 | 0.735074085 | 0.624048807 | 762.226418 |
| Improved | 8 | 143 / 1144 | 0.504780532 | 0.724837740 | 0.614809138 | 772.456982 |
| Improved | 9 | 143 / 1287 | 0.516944142 | 0.705877416 | 0.611410784 | 782.145660 |
| Improved | 10 | 143 / 1430 | 0.507581607 | 0.714721878 | 0.611151745 | 779.711557 |

![Recorded epoch losses only; missing epochs are shaded](figures/e10_recorded_training_losses.png)

The plot contains numeric loss records only, no medical-image data. Standard epochs **1–3** and Improved epoch **1** require the original initial/resume stdout segments containing their setup, step and completed-epoch events. The existing `loss-history` auditor correctly returns **INCOMPLETE**, with precisely these missing epochs; all available epoch component means are verified against their 143 step records. This is incomplete historical coverage, not failure of the recorded E10 jobs.

Standard CE/Dice/total terms refer to the main head. Improved terms are `(main + 0.5 half + 0.25 quarter)/1.75`, after auxiliary logit upsampling. Total loss equals 0.5 CE + 0.5 Soft Dice loss in each run. Trends within a model are interpretable; cross-model absolute loss is not a like-for-like main-head quality comparison, and neither soft training loss equals hard validation Dice. Both total losses decline over their available segments but are non-monotonic; both Soft Dice losses increase from epoch 9 to 10. No per-class training loss, target exposure, probability distributions or gradient attribution is present in these logs. Loss reduction does not prove class-specific convergence.

Epoch runtimes cover `run_epoch()` (on-demand preprocessing and optimisation), excluding epoch-boundary checkpoint saving and setup. Available sums are Standard seven epochs=5,146.637220 s; Improved nine epochs=6,906.425768 s. These are neither complete ten-epoch job times nor matched compute budgets. Final recorded cumulative training GPU peaks are Standard allocated/reserved=487,463,424 / 813,694,976 bytes; Improved=487,471,616 / 836,763,648 bytes. They are not whole-process CPU RSS or validation memory.

Validation loop times are Standard=508.611324894242 s and Improved=458.3723906259984 s; summed inference-function times are 315.113957173191 s / 277.92773962492356 s. Loop timing includes preprocessing/metrics and differs from scheduler duration. The previously student-recorded Improved Slurm elapsed 8m04s is retained with that separate scope. No matched repeat profiling or total ten-epoch runtime is inferred.
