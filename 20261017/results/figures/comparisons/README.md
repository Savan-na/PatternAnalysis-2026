# 3D model comparison figures

`g5_3d_validation.svg` uses all 50 epoch records for each frozen 3D run. It
shows Train patch loss, the predeclared patient-balanced Validation score every
five epochs, and pooled Validation Dice at each selected epoch 50.

`g6_test_3d.svg` and `.png` use the one-time held-out Test JSON. The left panel
shows pooled Dice for Bladder, Rectum and Prostate; the right panel gives each
of the six Test patients equal visual weight. Pooled Dice and the patient score
have different aggregation rules. These figures describe the fixed Test result;
they were not used to choose checkpoints or tune training.
