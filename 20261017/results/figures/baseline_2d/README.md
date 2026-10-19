# Supplementary 2D Validation figure

`g5_2d_validation.svg` comes from 50 completed epoch JSON records: the original
40-epoch Rangpur job 673908 and unchanged-configuration continuation job 674592.
It plots Train crop loss, the frozen patient-balanced Validation selection score
every five epochs, and pooled Dice, precision and recall for Bladder, Rectum
and Prostate at the selected epoch 50.

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
