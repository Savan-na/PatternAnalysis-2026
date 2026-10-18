# G5 Validation failure overlays

This folder is reserved for paired Standard and Improved 3D predictions on four frozen Validation cases. `J026_Week1` (Rectum) and `B040_Week1` (Prostate) had high overlap in both models at epoch 30; `K018_Week3` illustrates Rectum overprediction, and `S022_Week6` illustrates missed Prostate tissue in Standard 3D. The script `code/tools/review_g5_validation_failures.py` will reproduce each selected checkpoint's recorded confusion matrix before drawing MRI, ground-truth and prediction contours. Its JSON will also record case Dice, precision, recall and boundary distances in millimetres.

These examples were selected from Validation records to show both high-overlap and failure cases. They do not measure held-out Test performance, and a single axial panel does not establish the full 3D error pattern. The PNGs are medical-image-derived; review HipMRI data-use terms before distributing them.

No case-study output has been generated in this folder yet.
