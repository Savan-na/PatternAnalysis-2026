# Experiment Logs

Current note (20 October 2026): original G5 local records remain under
`training/baseline_2d`, `baseline_3d`, and `improved_3d`. Downloaded v2/v3
records are retained in their experiment `training_records` directories;
do not move them merely to match the suggested layout below. V4 remote
run IDs are `improved_3d_v4_a0_20261020` and `improved_3d_v4_a1_20261020`;
the jobs have been submitted, with no completion evidence yet.
See the [documentation index](../DOCUMENTATION_INDEX.md) for current status.

Store original machine-readable evidence here, grouped by run ID. An audit
record is a measurement, not a scientific conclusion. Keep raw stdout/stderr,
environment and source hashes, Slurm allocation/job ID, exact configuration,
optimizer step events, checkpoints' SHA-256 references, and complete
per-case Validation records. Do not rewrite a completed run's record; create
a new run ID for a correction or rerun.

Suggested layout for future runs:

```text
logs/
  audits/        Train/Validation data and geometry preflight JSON
  profiles/      One-volume GPU capacity and throughput records
  training/<arm_id>/<run_id>/
  validation/<arm_id>/<run_id>/
```

Use `baseline_2d`, `baseline_3d`, or `improved_3d` as `<arm_id>`.
Existing training arm directories contain completed G5 records. See validation/README.md for future diagnostics. Keep audit/profile
records shared because they assess the common dataset or capacity gate.

The root `.gitignore` excludes generated log subdirectories. The historical
Baseline–V4 logs remain in the older project; they must not be presented as
new-model evidence. Test-derived logs are permitted only after the final
checkpoint and evaluation protocol are frozen.
