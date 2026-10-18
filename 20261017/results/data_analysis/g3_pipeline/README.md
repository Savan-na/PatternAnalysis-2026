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

The [candidate exposure table](candidate_patch_class_exposure.csv),
[patient table](candidate_patch_patient_exposure.csv), and
[comparison figure](candidate_patch_exposure.svg) compare 1,430 uniform
random crops with 1,430 organ-aware candidate crops on the same 143 Train
volumes. They measure Patch Hit Rate, pooled sampled-voxel share, and complete
organ inclusion for fixed 64×64×32 crops. This helps decide whether a future
sampler needs to increase organ exposure; it does **not** describe actual past
or future model training. The case-weighted design gives 10 patches per scan,
so patients with more visits receive more patches.

`code/tools/summarize_g3_candidate_sampling.py` verifies every CSV record
against the G3 case grids and the returned manifest. Its Rangpur sources are
`logs/audits/g3_patch_candidate_full_20261018/manifest.json` (SHA-256
`e142be157e692572a7fd96084ec80c241edc6f165498dec25ea13919abb1889b`)
and `patches.csv` (SHA-256
`96ac1118b0109a21e3bc943df71094026dcf1852535d79124f200383d790dce2`).
