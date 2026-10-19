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

The three `g6_test_*.png` panels show descriptive held-out Test examples:
`M020_Week1` Prostate has high overlap (full-volume Dice 0.905 Standard,
0.887 Improved). On `M015_Week7`, Standard largely predicts Bladder in the
wrong tissue (Dice 0.068); Improved partly overlaps the target (Dice 0.406)
and still produces false positives. `W029_Week6` Prostate has low full-volume
precision (0.305 Standard, 0.205 Improved) even though the selected axial
slice looks reasonably aligned. `g6_test_selected_cases.json` links every
drawing to the unmodified one-time Test confusion matrix and gives the exact
three-dimensional overlap metrics. These reviews did not change the models.

A single axial panel cannot establish the full 3D error pattern. These PNGs
are medical-image-derived; review HipMRI data-use terms before distributing
them.

The three `g6_body_*.png` panels add axial, coronal and sagittal slices through
the most distant Improved 3D Body false positive in `M015_Week3`,
`L011_Week5` and `W041_Week0`. The cyan line is GT Body; orange is predicted
Body. All six predictions reproduced their original one-time Test confusion
matrices before drawing. In each example, Improved predicts Body well outside
the main anatomical outline, including image-margin regions; Standard has far
fewer such voxels. This visually explains the large Improved Body HD95 without
asserting that the pictured single voxel itself defines the 95th percentile.
The matching `g6_body_selected_cases.json` records exact Body Dice, HD95 and
false-positive counts; it is tied to the unchanged Test JSON SHA-256
`21ce5e2eb53e3bf758478fd7c9fd95b887d82031b7a66ed77c7ef3168adc3e2f`.
This is descriptive Test review, not model tuning.
