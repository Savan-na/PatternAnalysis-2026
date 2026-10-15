# Recognition Project

This directory contains the COMP3710 HipMRI multi-class 3D MRI segmentation project for prostate radiotherapy planning. It compares a Standard 3D U-Net baseline with a residual, deeply supervised Improved 3D U-Net. Outputs cover six semantic classes; clinical use and held-out test performance have not been established.

- [Implementation and task-specific reproducibility instructions](hipmri_3d_improved_unet_hard/README.md)
- [Python runtime and regression-test dependencies](requirements.txt)
- [Unified Baseline–V4 experiment history](docs/experiments.md) — protocols, numerical results, selection and evidence boundaries
- [Feasibility Report](docs/FEASIBILITY_REPORT.md) — research rationale, practical viability and remaining requirements
- [Class-prediction-collapse analysis](docs/analysis/class-prediction-collapse.md) — E10 diagnosis and selected-patch probe
- [Final Report: Sections 1–5, controlled Validation comparison and four paired failure studies](FINAL-REPORT.md)
- [Failure-case evidence manifest and external figure paths](docs/evidence/failure-cases.md) — original medical PNGs remain outside Git pending permission
- [Report-to-code traceability and evidence register](docs/evidence/traceability.md) — internal technical reference, not a course-mandated deliverable
- [AI contribution and historical command records](hipmri_3d_improved_unet_hard/ai.md)

The model-package README identifies implementation entry points and remaining submission gaps. The experiment history is the current research timeline; the detailed analysis and evidence register support the course-facing report. The course specification requires the report in the task README and a PDF export of that README.

An isolated [V3 E12 prediction adapter](hipmri_3d_improved_unet_hard/predict_v3.py) now checks the protected checkpoint, selection receipt and complete A/B/C lineage before admitting inference. Synthetic CPU tests cover admission and Test opt-in. The actual E12 files and HipMRI data are unavailable in this workspace, so real preflight, formal freeze and held-out Test evaluation remain pending. The original `predict.py` remains an E10 path.

## V4 E20 Validation decision (14 October 2026)

The frozen paired protocol compared Control (V3 objective, Prostate Tversky alpha=0.5) with Treatment (alpha=0.7), both continued from the selected V3 E12 parent through E20. E14, E16, E18 and E20 full-volume Validation completed. Treatment reduced **pooled Prostate false positives by 32.30%**, but failed three mandatory conjunctive gates: Prostate **patient-macro** Dice gain 0.027050 < 0.03, pooled Prostate Recall loss 0.076124 > 0.05, and maximum other-class **patient-macro** Dice drop 0.048440 > 0.02 (Rectum). The frozen rule rejected Treatment and retained V3 E12 as the protected candidate; Control E20 was not automatically selected. These are exploratory Validation findings, not Held-out Test results. The final model is not formally frozen.

See the frozen acceptance decision (local-only: `research_archive/10_scientific_documentation/COMP3710_V4_FINAL_DOCS_20261014/V4_ACCEPTANCE_DECISION.md`), independent P4-04C audit (local-only: `research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md`), and P4-05A freeze-readiness audit (local-only: `research_archive/12_handover_and_pretest_readiness/COMP3710_P4-05A_20261014/P4-05A_READINESS_AUDIT.md`). The [package README](hipmri_3d_improved_unet_hard/README.md) covers implementation; the [technical report](FINAL-REPORT.md) covers scientific interpretation; [traceability](docs/evidence/traceability.md) links claims to code and receipts.

## Local regression tests

Use Python 3.11 for the documented CPU regression environment. From the
repository root, create an isolated environment, install the bounded runtime
dependencies and run all `unittest` tests (including nested diagnostics tests):

```powershell
py -3.11 -m venv .venv-regression
.\.venv-regression\Scripts\Activate.ps1
python -m pip install -r recognition\requirements.txt
python -m unittest discover -v -s recognition\hipmri_3d_improved_unet_hard -t .
```

The test suite uses only synthetic data and does not require the HipMRI
dataset, a checkpoint, CUDA, or Slurm. Training and real-data inference still
require their documented external data and allocated CUDA environment. The
requirements file is a compatible-version specification, not a lockfile;
install a PyTorch build matching the target machine when CUDA is needed.

Latest local regression run: **279 tests passed** in **22.012 s** using
Windows/Python 3.11.16, CPU PyTorch 2.13.0, NumPy 2.4.6, NiBabel 5.4.2,
SciPy 1.17.1 and Matplotlib 3.11.2. CUDA was unavailable; this run verifies
the synthetic CPU suite only.
