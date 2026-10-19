# Results and Report Artifacts

This folder is for **interpreted** outputs linked to immutable run IDs in
`../logs/`: per-patient and per-class metric tables, boundary analyses,
resource comparisons, selected failure-case panels, and the manager-facing
research report. Every new result should identify its source run, checkpoint hash,
data split, measurement definition, and any incomplete cases.

Current layout:

```text
results/
  data_analysis/          Train-only audits and historical evidence
  tables/comparisons/     Frozen 3D Test JSON and per-case/class CSV
  figures/comparisons/    3D Validation, Test and failure figures
  figures/baseline_2d/   Supplementary 2D Validation figures
  case_studies/comparisons/  Selected MRI/GT/prediction panels
  reports/
```

Only populated analysis groups are retained. Cross-model tables and visuals
are in `comparisons`; the supplementary 2D figure has its own folder. The one
cumulative formal research report
is [`reports/HipMRI_Research_Report.md`](reports/HipMRI_Research_Report.md).
The earlier mixed-source analysis is archived beside it. The
[`data_analysis/`](data_analysis/README.md) index separates historical figures,
three G1 image groups, a G2 Train summary and a G3 preprocessing group, each
with a short README beside its figures.

The restarted Standard and Improved 3D arms completed 50 Train epochs each;
the supplementary 2D arm completed 40. Their Validation figures are under
[`figures/comparisons/`](figures/comparisons/README.md). The one-time 3D
held-out [Test result](tables/comparisons/README.md) and selected
[Validation/Test overlays](case_studies/comparisons/README.md) are also local.
Generated artifacts and medical-derived PNGs stay in ignored subdirectories
unless course and data licence requirements allow inclusion in the final
submission. Historical
Baseline–V4 plots or scores are background evidence, not results of this
restart.
