# Held-out Test comparison

`g6_test_3d.json` records the one-time evaluation of Standard and Improved 3D U-Net, using their Validation-selected epoch-50 checkpoints and the fixed 30-volume, six-patient Test split. It contains per-case and pooled six-class overlap, patient-level small-organ Dice, physical HD95, prostate axial-region Dice, inference time, peak GPU memory, model parameter count, checkpoint size and source hashes. Rangpur job 674125 completed successfully; the local SHA-256 is `21ce5e2eb53e3bf758478fd7c9fd95b887d82031b7a66ed77c7ef3168adc3e2f`.

The Test data did not choose a checkpoint, class weight, sampler or preprocessing setting. Empty-organ surface distances are recorded as unavailable, with counts, rather than assigned a misleading zero. The chart and interpretation are in `results/figures/comparisons/` and the formal research report, respectively.
