# Seven-model fixed-checkpoint assessment audit

Models: supplementary 2D, Standard 3D, Improved v1/v2/v3, v4 A0/A1.
Selected measured Validation epochs: 50, 50, 50, 45, 50, 45, 45.
Each selection is independently checked against all 50 records, source hashes,
manifest, best.pt and the immutable selected epoch checkpoint. v2 epoch 45 was
retained remotely after verified archival of other epoch weights.

This replaces the *unexecuted* five-model audit as the next evaluation task;
historical audit scripts and records are retained, not silently rewritten.
No training, thresholds, postprocessing, or selection using Test is introduced.
Test has already been inspected and cannot be described as fresh blind evidence.

## Measurements

- 38 Validation and 30 existing Test volumes for each of seven models: 476 pairs.
- Six-class confusion; local aggregation must pool visits within each patient,
  calculate metrics, then average patients equally. Also report voxel-pooled values.
- Five foreground HD95/ASSD with explicit empty-mask status; physical millimetres.
- Prostate physical base/apex partitions, presence extent, OAR confusion,
  six-connected components, internal slice gaps, adjacent area transitions.
- Background-to-Body false predictions at least 20 mm from true Body.
- Matched reference-case inference: three warm-ups, five synchronized repetitions,
  peak allocated/reserved VRAM and parameters, one resident GPU model at a time.
  In addition, every evaluated volume records synchronized single-pass inference
  latency and peak allocated VRAM, enabling cohort distributions. These are not
  repeated per-volume microbenchmarks. Preprocessing, metrics and IO are excluded
  and must not be labelled end-to-end.
- Shared Validation/Test diagnostic slices with MRI, GT, prediction and FP/FN
  overlays; six classes represented. Six Validation patients have additional
  Body three-grid-plane panels, displayed at physical pixel aspect. These are
  purposeful diagnostic selections, not representative prevalence estimates.
- Exact-file cross-split MRI duplicate checks. No claim of excluding near duplicates.

All 266 Validation confusion matrices must reproduce their locked records;
historical Standard/v1 Test matrices must also reproduce. Any mismatch stops
the audit and preserves partial evidence. No output overwrite or automatic rerun.

## Execution and storage

`bash tools/submit_all_model_audit.sh` submits one CPU test gate followed by one
A100 job (one-hour allocation, not a runtime prediction). It checks at least
256 MiB free. No checkpoints are written. Do not submit the old five-model audit
at the same time. Frozen training source files are untouched.

Output is isolated under:
`results/experiments/improved_3d_v4/comprehensive_audit_20261021/`.
`metrics.json` with `complete=true` and `checksums.json` indicate completion.
Download the whole directory, verify checksums, then analyze; the current
five-model summarizer must not be used unmodified for this seven-model output.

## Assessment limitations

The locally supplied course PDF is a draft (filename v2.2, internal heading v2.0).
The latest official assessment announcement has not been independently checked.
Its Hard tier specifies 3D residual units and deep supervision, downsampled HipMRI,
and Test Dice >= 0.7 without an unambiguous aggregation definition. Report
prostate, each foreground class and several declared aggregates transparently;
do not infer threshold compliance from a three-organ Validation average.
Clinical safety, external generalization, formal consultation, PR submission and
future tutor review cannot be established by this numerical audit.
The supplementary 2D model uses slices derived from the controlled 3D cohort,
not the specified keras_slices_data route, and its training recipe differs.
