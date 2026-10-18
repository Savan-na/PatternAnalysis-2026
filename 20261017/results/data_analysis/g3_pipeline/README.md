# G3 existing-pipeline organ preservation

The [CSV](organ_preservation_by_volume.csv) compares raw and factor-two
downsampled label counts, physical volumes, and occupied axial slices for
Bladder, Rectum, and Prostate in all 143 Train volumes. The [figure](organ_physical_volume_ratio.svg)
shows each organ's per-volume physical-volume ratio; 1 means equal labelled
physical volume. These results check whether the *implemented* preprocessing
removes an organ or materially changes its size before any sampling decision.

Both outputs come from `code/tools/summarize_g3_pipeline.py`, which verifies
and reconciles the G1 and G3 audit JSON files before writing. The G3 source is
`logs/audits/g3_existing_pipeline_full_20261018_job671892.json` (SHA-256
`319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9`).
Physical-volume agreement does not establish boundary fidelity or training
patch exposure. Interpretation and remaining checks are in the
[research report](../../reports/HipMRI_Research_Report.md).
