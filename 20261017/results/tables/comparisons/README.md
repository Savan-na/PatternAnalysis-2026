# Held-out Test comparison

`g5_validation_body_audit.json` contains the complete 38-volume, six-patient
Validation Body boundary check for both fixed epoch-50 3D checkpoints. It
reproduced all saved confusion matrices before measuring HD95, false-positive
volume and distant Background-to-Body errors. The paired patient figure is in
`results/figures/comparisons/`; selected image panels are in
`results/case_studies/comparisons/`.

`g5_validation_aux_logits.json` is a read-only diagnostic of the same
selected Improved 3D weights on the same 38 Validation volumes. It compares
the trained summed logits with the main head alone; the latter was not trained
as a stand-alone model. It informs experiment planning, not checkpoint
selection or a held-out result.

`g6_test_3d.json` records the one-time evaluation of Standard and Improved 3D U-Net, using their Validation-selected epoch-50 checkpoints and the fixed 30-volume, six-patient Test split. It contains per-case and pooled six-class overlap, patient-level small-organ Dice, physical HD95, prostate axial-region Dice, inference time, peak GPU memory, model parameter count, checkpoint size and source hashes. Rangpur job 674125 completed successfully; the local SHA-256 is `21ce5e2eb53e3bf758478fd7c9fd95b887d82031b7a66ed77c7ef3168adc3e2f`.

The Test data did not choose a checkpoint, class weight, sampler or preprocessing setting. Empty-organ surface distances are recorded as unavailable, with counts, rather than assigned a misleading zero. The chart and interpretation are in `results/figures/comparisons/` and the formal research report, respectively.

`g6_test_case_class_metrics.csv` expands the same fixed result into 360 rows:
two models × 30 Test volumes × six classes. Each row gives GT and prediction
volume, TP/FP/FN, Dice, IoU, precision and recall; HD95 is included for
foreground classes when defined. It supports descriptive failure analysis and
must not be used to tune a model on Test.

`g5_validation_case_class_metrics.csv` contains 456 rows from the two selected
epoch-50 Validation records (two models × 38 volumes × six classes). It is
the appropriate per-case confusion and overlap table for future model
diagnosis on Validation; patient-level scores pool visits before calculating
Dice, as shown in the companion heatmap.
