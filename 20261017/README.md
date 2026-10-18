# HipMRI Data-Driven Model Development

This is the COMP3710 research workspace. G1 raw Train integrity passed with
documented limits. G2 now measures Train-only distributions before changing
preprocessing, sampling, loss, or architecture. The primary later comparison
is standard versus improved 3D U-Net. The existing 2D implementation is
background work and does not require
a new training run.

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
are previous evidence; G1 figures are in separate image groups. No model in
this workspace has been trained on the restarted data. The held-out Test
arrays remain protected.
