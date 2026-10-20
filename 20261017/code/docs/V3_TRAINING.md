# Improved v3 GN1 experiment

v1 control -> v3 treatment: replace exactly 14 InstanceNorm3d layers with
GroupNorm(1,C), keeping affine values, epsilon and module positions. The v1
constructor initializes all parameters first; normalization replacement preserves
the post-construction CPU RNG. Parameter count remains 2,366,946. The original
residual backbone, dropout .3, three-head logit sum, weighted CE/Dice, sampling,
optimizer and 50-epoch/6500-update budget remain unchanged. v2 is not the causal
control for this experiment. GN still depends on the input spatial extent.

## Identity and storage

- Config: `config/v3_gn1_frozen.json`, including predeclared decision criteria.
- Model: `hipmri_rewrite/improved_v3.py` (inherits unchanged v1 forward).
- Trainer: `tools/train_v3_cohort.py`, isolated from frozen G5 and v2 trainers.
- Remote run: `logs/training/improved_3d_v3_gn1_20261020/`.
- Local analysis: `results/experiments/improved_3d_v3/`.
- Tests: `tests/test_improved_v3.py`.

No original checkpoints/configs are edited. Shared source hashes, historical v1
manifest SHA, seed and Torch version are checked before training. Resume checks
config/source hashes, model, data root and Python/Torch/NumPy versions. Save
initial weights/digest, atomically replace latest after each completed batch,
and preserve every completed epoch's weights and JSON plus selected best.
Historical initial v1 weights are unavailable: matching construction/seed is
stronger evidence than a config match, but is not a recorded historical tensor
comparison. Initializing from trained v1/v2 checkpoints is prohibited here.

## Submission on Rangpur

```bash
bash -n "$HOME/20261018_g1_stage/20261017/code/tools/submit_v3_50.sh"
bash "$HOME/20261018_g1_stage/20261017/code/tools/submit_v3_50.sh"
```

One CPU test job gates one A100 six-hour allocation; expected time is roughly
two GPU hours based on earlier runs, not a guarantee. Six hours is a time limit.
The script checks package hashes, rejects an existing new-run directory and
blocks duplicate queued/running jobs; the trainer holds an exclusive writer lock.
Do not submit repeatedly. After inspecting a stopped job, explicitly resume:

```bash
bash "$HOME/20261018_g1_stage/20261017/code/tools/submit_v3_50.sh" resume
```

The current uncommitted batch can be lost on interruption; previous completed
epochs and latest committed batch remain. Old individual batch weights are not
all retained. Future epoch extension uses the trainer's explicit
`--resume --extend-to-epochs N` with the unchanged config, not a second new run.

## Evaluation

Every five epochs: 38 Validation volumes, six-class confusion/overlap, five
foreground HD95/ASSD (one-empty failures explicitly null/status), distant
Background-to-Body counts and GT Background denominators. Selection remains
patient-equal three-small-organ Dice; fixed epoch50 and best are reported
separately. Training records include class exposure, batch losses, gradient
norms, peak memory, load time and checkpoint time. Full confusion matrices
support Background/Body-to-small-organ error analysis. No Test inference is
submitted by this package.

## Local checks and limits

CPU tests check identical v1/v3 initial tensors and RNG, all 14 GN1 settings,
exact additive inference, gradients to every head, and optimizer/RNG recovery
matching uninterrupted dropout training. Python and Bash syntax also checked.
CUDA training/performance remain unverified until the formal job starts.
