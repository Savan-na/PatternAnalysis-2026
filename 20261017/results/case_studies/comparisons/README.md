# Selected segmentation overlays

The four `g5_validation_*.png` panels compare the two selected epoch-50
checkpoints against Validation ground truth. Cyan is ground truth; orange is
prediction. `J026_Week1` (Rectum) and `B040_Week1` (Prostate) have relatively
high overlap. `K018_Week3` shows extra Rectum predictions. `S022_Week6` was
chosen from the earlier epoch-30 failure review; at epoch 50 its full-volume
Prostate recall exceeds 0.97 in both models, while precision is below 0.50.
The picture now illustrates misplaced and extra predicted tissue more clearly
than organ-wide omission. `g5_validation_selected_cases.json` gives the exact
full-volume metrics and checkpoint hashes.

The planned `g6_test_*.png` panels will show descriptive held-out Test examples:
high-overlap `M020_Week1` Prostate, low-overlap `M015_Week7` Bladder and
`W029_Week6` Prostate. Their JSON will link each drawing to the one-time Test
confusion matrix. Reviewing these examples does not change the models.

A single axial panel cannot establish the full 3D error pattern. These PNGs
are medical-image-derived; review HipMRI data-use terms before distributing
them.
