# P2 fixed-weight Validation diagnostic

Four predefined cases, two unchanged epoch-50 models. Body panels compare full-volume and fixed 64x64x32 XYZ sliding-window inference; cyan is ground truth, orange prediction. Head panels show raw Body-minus-Background logits on the same evaluation-selected slice; these are not probabilities. All full-volume confusion matrices reproduced G5 before comparison.

Raw measurements: logs/validation/inference_context/p2_four_cases_20261020/manifest.json. Interpret all six classes and per-case changes before proposing a larger experiment. No training or Test evaluation was performed.
