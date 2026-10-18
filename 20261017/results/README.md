# Results and Report Artifacts

This folder is for **interpreted** outputs linked to immutable run IDs in
`../logs/`: per-patient and per-class metric tables, boundary analyses,
resource comparisons, selected failure-case panels, and the manager-facing
research report. Every new result should identify its source run, checkpoint hash,
data split, measurement definition, and any incomplete cases.

Suggested layout as results become available:

```text
results/
  tables/<arm_id>/
  figures/<arm_id>/
  case_studies/<arm_id>/
  reports/
```

The first three categories each have pre-created `baseline_2d`,
`baseline_3d`, `improved_3d`, and `comparisons` directories. Put cross-model
tables and visuals in `comparisons`. The one cumulative formal research report
belongs in `reports/`; existing `data_analysis/` files remain historical
Train-only source analysis and are not another current audit report.

No restarted-model result exists yet. Generated artifacts and
medical-derived images stay in ignored subdirectories unless course and data
licence requirements allow inclusion in the final submission. Historical
Baseline–V4 plots or scores are background evidence, not results of this
restart.
