# Patient-Balanced Sampling Exposure

The [50-epoch geometry record](g5_train_patch_geometry.json) reconstructs
all 13,000 actual 3D uniform crop origins from the two paired training logs.
It measures spatial coverage, whereas the other files here measure class
exposure. Only 6 crops (0.046%) touched an outer 2-voxel corner; 423 (3.25%)
touched an outer 8-voxel corner. The result alone cannot establish that an
edge-specific training change would improve segmentation. Convolutional
padding occurs at every patch boundary, so this is specifically a measure of
physical volume-margin coverage.

The [class table](class_exposure.csv), [patient table](patient_exposure.csv), and [figure](patient_balanced_exposure.svg) describe one CPU dry-run epoch: one Train scan per patient, ten model-ready patches per policy and patient. They show whether patient balancing and crop placement change small-organ exposure before training. The uniform and organ-aware policies share their first four crops per patient.

The source files are `logs/audits/g4_patient_full_20261018/manifest.json` (SHA-256 `cafd3593efa5880a4a61b6281bac5d177f58cd713e734cf8dad422d191f0b90c`) and `patches.csv` (SHA-256 `04a6822f15940b658f0c3eb062528aa4fa17721552030dc5dc6d45ce5c441968`). The verification script checked each patch against the frozen G3 case grids and reconciled every class and patient total with the manifest. This is one deterministic epoch, without spatial augmentation or model optimization. The completed training logs separately record actual patch class exposure. Interpretation is in the [research report](../../reports/RESEARCH_REPORT.md).
