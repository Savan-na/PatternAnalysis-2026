# Recognition Project

This directory contains the COMP3710 HipMRI multi-class 3D MRI segmentation project for prostate radiotherapy planning. It compares a Standard 3D U-Net baseline with a residual, deeply supervised Improved 3D U-Net. Outputs cover six semantic classes; clinical use and held-out test performance have not been established.

- [Implementation and task-specific reproducibility instructions](hipmri_3d_improved_unet_hard/README.md)
- [Final Report: Sections 1–5, controlled Validation comparison and four paired failure studies](FINAL-REPORT.md)
- [Failure-case evidence manifest and external figure paths](FAILURE_CASE_EVIDENCE.md) — original medical PNGs remain outside Git pending permission
- [Report-to-code traceability and evidence register](REPORT_CODE_TRACEABILITY.md) — internal technical reference, not a course-mandated deliverable
- [AI contribution and historical command records](hipmri_3d_improved_unet_hard/ai.md)

The task README identifies current entry points and remaining submission gaps. The Final Report supplements it; the course specification requires the report in the task README and a PDF export of that README.

## V4 E20 - Current experiment status (13 October 2026)

The HipMRI project has prepared an exploratory, paired V4 experiment
targeting Prostate false positives. The Control continues the selected
V3 E12 objective; the Treatment changes only the Prostate Tversky
false-positive coefficient from alpha=0.5 to alpha=0.7.

The planned continuation is E13-E20, with 143 optimiser updates per
epoch and full-volume Validation at E14, E16, E18 and E20.

CPU numerical, training-loop, recovery, journal-integrity and
lock-lifecycle tests have passed within their stated test scopes.

**Status:** V4 GPU training, GPU smoke testing, E20 Validation and
Held-out Test evaluation have not been executed. No V4 performance
improvement is claimed.

See `hipmri_3d_improved_unet_hard/README.md` for experiment details and
`REPORT_CODE_TRACEABILITY.md` for evidence mapping.
