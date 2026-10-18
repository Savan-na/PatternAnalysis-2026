# HipMRI Data-Driven Model Development

This is the COMP3710 research workspace. The immediate task is G1: audit the
current raw Train MRI/mask pairs before changing preprocessing, sampling, loss,
or architecture. The primary later comparison is standard versus improved 3D
U-Net. The existing 2D implementation is background work and does not require
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
statistics in `results/data_analysis/` are previous evidence, not a completed
audit of the current pipeline. No model in this workspace has been trained on
the restarted data. The held-out Test arrays remain protected.
