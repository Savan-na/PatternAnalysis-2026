# Model comparison figures

`g5_three_arm_training_trajectories.svg`/`.png` shows Train objective trends
and the predeclared patient-balanced Validation score every five epochs for
the two 3D models and supplementary 2D model. Compare each loss trend within
its own arm because the 2D input and class weights differ.
`g5_three_arm_validation_all_classes.svg`/`.png` compares pooled Validation
Dice for all six classes at each arm's selected epoch 50. The 2D versus 3D
comparison combines architecture, spatial context, sampling and class-weight
differences, so it does not isolate one cause.

`g5_3d_validation_all_classes.svg`/`.png` and
`g6_test_all_classes.svg`/`.png` show pooled Dice for all six classes,
including Background, Body and Bones. They supplement the small-organ
selection figures; pooled Dice is not the patient-balanced selection score.
`g5_validation_patient_class_dice.svg`/`.png` shows each Validation patient's
six class Dice after pooling that patient's visits, separately for both 3D
models. It supports later Train/Validation failure analysis.
`g5_validation_body_boundaries.svg`/`.png` shows per-patient median Body HD95
and distant Background-to-Body errors across all 38 selected Validation
volumes; its source is `results/tables/comparisons/g5_validation_body_audit.json`.
`g6_test_hd95_all_foreground.svg`/`.png` compares median per-volume physical
HD95 for the five foreground classes. `g6_test_failure_patterns.svg`/`.png`
shows M015 Bladder Dice by week and each Test volume's Body HD95. All G6
figures use the unchanged one-time Test result and are descriptive only.
The Test JSON SHA-256 is
`21ce5e2eb53e3bf758478fd7c9fd95b887d82031b7a66ed77c7ef3168adc3e2f`;
the two selected G5 records are `epoch_050.json` in the Standard and Improved
training directories. Regenerate and verify the full-class figures with
`python -B code/tools/summarize_all_class_review.py` from the project root.
