# HipMRI Data-Driven Model Development

This is the COMP3710 research workspace. G1–G4 audited the Train data and
model input; G5 trained Standard and Improved 3D U-Net for 50 epochs each and
a supplementary 2D U-Net for 40 epochs. G6 evaluated the two
Validation-selected 3D checkpoints once on the held-out Test set. The
[formal research report](results/reports/HipMRI_Research_Report.md) contains
the evidence, results, limits and conclusions.

## Where things belong

| Directory | Purpose |
|---|---|
| [`code/`](code/README.md) | Source, configuration, tests, methods, transfer tools |
| [`logs/`](logs/README.md) | Raw audit, profiling, training and validation records |
| [`results/`](results/README.md) | Historical analysis, interpreted figures and research report |

[`PROJECT_MAP.md`](PROJECT_MAP.md) is a bilingual, generated tree and
per-file content index. The scientific rules are in
[`code/docs/STUDY_PROTOCOL.md`](code/docs/STUDY_PROTOCOL.md). Historical Train
statistics in [`results/data_analysis/historical_train/`](results/data_analysis/historical_train/README.md)
are previous evidence; G1 figures are in separate image groups. The 3D
checkpoints and raw MRI remain on Rangpur; local JSON records, figures and
case panels are indexed under `logs/` and `results/`. Test was used once
after Validation selected the checkpoints; it must not guide later tuning.
