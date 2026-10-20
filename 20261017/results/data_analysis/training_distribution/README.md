# Training Data Distribution

These eleven CSV tables and eight SVG plots summarize the 143 raw Train MRI/Mask
pairs. The first group uses the verified G1 audit for class, patient, volume,
longitudinal organ size, normalized organ-box location, intensity-summary and spacing statistics. The second group uses the full G2
Rangpur CPU run for axial class coverage and nonzero MRI intensity histograms.
They help check which data questions matter before changing preprocessing or
sampling; raw slice coverage is not training-patch exposure.
Shares use **all raw voxels** as the denominator; each patient pools its visits,
so patients with more visits contribute more voxels to the global total.

Source: `logs/audits/train_full_20261018_job671208.json`, SHA-256
`6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1`.
Regenerate with `python -B code/tools/summarize_g2_train_audit.py` using the
existing Matplotlib environment. The two detailed input CSVs are
`per_slice_class_coverage.csv` (SHA-256 `d8b0966e9965797991f2313858d8088c1e2c32a62bcc483090be1a0dede17e87`)
and `per_volume_intensity_histogram.csv` (SHA-256
`c1b39892873b1b61dca3f03f8abcb0ec69a48154609a017a420f0502523d9930`).
Regenerate their summaries and plots with
`python -B code/tools/summarize_g2_slice_detail.py`. The full interpretation
and G2 limits are in the [research report](../../reports/RESEARCH_REPORT.md).
