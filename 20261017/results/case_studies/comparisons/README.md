# G5 Validation failure overlays

This folder is reserved for paired Standard and Improved 3D predictions on the frozen Validation cases `K018_Week3` (Rectum) and `S022_Week6` (Prostate). The script `code/tools/review_g5_validation_failures.py` will reproduce the recorded epoch-30 confusion matrices before drawing MRI, ground-truth and prediction contours. Its JSON will also record case Dice, precision, recall and boundary distances in millimetres.

These examples were selected from Validation records to inspect overprediction and missed tissue. They do not measure held-out Test performance, and a single axial panel does not establish the full 3D error pattern. The PNGs are medical-image-derived; review HipMRI data-use terms before distributing them.

No case-study output has been generated in this folder yet.
