# Held-out Test comparison

`g6_test_3d.json` is reserved for the one-time evaluation of Standard and Improved 3D U-Net after both requested 50-epoch continuations finish. The evaluation reads each arm's best Validation checkpoint and the fixed 30-volume, six-patient Test split. It records per-case and pooled six-class overlap, patient-level small-organ Dice, physical HD95, prostate axial-region Dice, inference time, peak GPU memory, model parameter count, checkpoint size and source hashes.

The Test data will not choose a checkpoint, class weight, sampler or preprocessing setting. Empty-organ surface distances are recorded as unavailable, with counts, rather than assigned a misleading zero. This directory has no Test result yet.
