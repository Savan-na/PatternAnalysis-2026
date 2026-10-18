# G2 raw Train summary

These five CSV tables and four SVG plots summarize the 143 raw Train MRI/Mask
pairs from the verified G1 audit. They show global class frequency, patient and
volume variation, MRI intensity summaries, and voxel spacing. They help choose
which data questions to investigate before changing preprocessing or sampling.
Shares use **all raw voxels** as the denominator; each patient pools its visits,
so patients with more visits contribute more voxels to the global total.

Source: `logs/audits/train_full_20261018_job671208.json`, SHA-256
`6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1`.
Regenerate with `python -B code/tools/summarize_g2_train_audit.py` using the
existing Matplotlib environment. The full interpretation and G2 limits are in
the [research report](../../reports/HipMRI_Research_Report.md).
