# 3D model comparison figures

`g5_3d_validation.svg` uses all 50 epoch records for each frozen 3D run. It
shows Train patch loss, the predeclared patient-balanced Validation score every
five epochs, and pooled Validation Dice at each selected epoch 50.

`g6_test_3d.svg` and `.png` use the one-time held-out Test JSON. The left panel
shows pooled Dice for Bladder, Rectum and Prostate; the right panel gives each
of the six Test patients equal visual weight. Pooled Dice and the patient score
have different aggregation rules. These figures describe the fixed Test result;
they were not used to choose checkpoints or tune training.

`g5_3d_validation_all_classes.svg`/`.png` and
`g6_test_all_classes.svg`/`.png` show pooled Dice for all six classes,
including Background, Body and Bones. They supplement the small-organ
selection figures; pooled Dice is not the patient-balanced selection score.
`g5_validation_patient_class_dice.svg`/`.png` shows each Validation patient's
six class Dice after pooling that patient's visits, separately for both 3D
models. It supports later Train/Validation failure analysis.
`g6_test_hd95_all_foreground.svg`/`.png` compares median per-volume physical
HD95 for the five foreground classes. `g6_test_failure_patterns.svg`/`.png`
shows M015 Bladder Dice by week and each Test volume's Body HD95. All G6
figures use the unchanged one-time Test result and are descriptive only.
The Test JSON SHA-256 is
`21ce5e2eb53e3bf758478fd7c9fd95b887d82031b7a66ed77c7ef3168adc3e2f`;
the two selected G5 records are `epoch_050.json` in the Standard and Improved
training directories. Regenerate and verify the full-class figures with
`python -B code/tools/summarize_all_class_review.py` from the project root.
