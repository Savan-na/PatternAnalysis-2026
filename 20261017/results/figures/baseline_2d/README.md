# Supplementary 2D Validation figure

`g5_2d_validation.svg` comes from the 40 epoch JSON records of Rangpur job 673908. It plots Train crop loss, the frozen patient-balanced Validation selection score every five epochs, and pooled Dice, precision and recall for Bladder, Rectum and Prostate at the selected epoch 40.

The selected score is 0.7367. Pooled metrics aggregate voxels rather than patients, so they answer a different question. This 2D arm uses axial crops and organ-aware sampling; comparing its score with the 3D arms cannot isolate an architecture effect. Held-out Test has not been used in this figure.
