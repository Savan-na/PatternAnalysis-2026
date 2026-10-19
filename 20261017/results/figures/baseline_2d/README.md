# Supplementary 2D Validation figure

The current six-class figure comes from 50 completed epoch JSON records: the
original 40-epoch Rangpur job 673908 and unchanged-configuration continuation
job 674592. Train loss and the frozen patient-balanced Validation score every
five epochs are shown in the [shared trajectory figure](../comparisons/training/g5_three_arm_training_trajectories.png).

The selected score rose from 0.7367 at epoch 40 to 0.7757 at epoch 50.
Pooled metrics aggregate voxels rather than patients, so they answer a
different question. This 2D arm uses axial crops and organ-aware sampling;
comparing its score with the 3D arms cannot isolate an architecture effect.
Held-out Test has not been used in this figure.

`g5_2d_validation_all_classes.svg`/`.png` shows pooled Dice for all six
classes at the selected epoch 50, including Background, Body and Bones.
Both figures were regenerated after checking the local manifest SHA-256
`f34deddbf59b4b688464033e4301e99b605637e5fae6e564fc0ba89c09fb5ff9`
and epoch-50 record SHA-256
`364c40ebb8c5e460ca99787e242653bc8a39e4868af439cf266be004ce2d6587`.
