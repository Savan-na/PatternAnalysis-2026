# Improved v2 independent-supervision run

The new files leave the frozen G5 trainer, network, losses and records unchanged.
The subclass inherits every parameter and initialization from Improved v1.
Training uses `(main_loss + .5*half_loss + .25*quarter_loss)/1.75`, with the
original weighted CE/foreground Dice in each term. All heads use upsampled logits
and full-resolution labels. Inference returns the newly trained main head.

## Frozen experiment

- 26 Train patients, 143 volumes; original patient/week/crop schedule and seed.
- 50 epochs, 130 batch updates per epoch, batch size 2; original Adam settings.
- Original normalization, InstanceNorm, dropout, channels and uniform sampling.
- Validation every five epochs: all six classes, foreground surfaces, distant
  Background-to-Body errors. No Test arrays are loaded.
- Retain original patient-equal small-organ checkpoint score and report epoch50.
- Decision criteria are recorded in `config/v2_independent_supervision_frozen.json`.

Historical source SHA checks reject changes in shared G5 dependencies. Seed and
Torch version are checked against the historical Improved manifest. Initial
weights are saved as `initial.pt` with a digest; exact identity to unavailable
historical initial weights cannot be established merely from a matching seed.

## Submission and recovery on Rangpur

From the login node after extracting the verified package:

```bash
bash -n "$HOME/20261018_g1_stage/20261017/code/tools/submit_v2_50.sh"
bash "$HOME/20261018_g1_stage/20261017/code/tools/submit_v2_50.sh"
```

One CPU test job gates one A100 six-hour allocation. Six hours is a requested
limit, not an expected runtime or a guarantee of uninterrupted resources.
The submission script rejects an existing new-run directory or a queued/running
job with the same name. The trainer holds an OS lock against concurrent writers.

New run: `logs/training/improved_3d_v2_ds_20261020/`.
Each completed batch atomically replaces `latest.pt` (model, optimizer, CPU/CUDA
and NumPy RNG, batch cursor and accumulated logs). Each completed epoch preserves
`epoch_NNN.pt` and `epoch_NNN.json`; `best.pt` follows Validation selection.
Individual historical batch checkpoints are not all retained. An interruption
can lose the current uncommitted batch, while completed epochs remain available.

After a timeout/failure, inspect the log, then explicitly submit:

```bash
bash "$HOME/20261018_g1_stage/20261017/code/tools/submit_v2_50.sh" resume
```

Resume rejects config/source/runtime/model/data-root changes and reloads the
optimizer and RNG states. A future epoch extension is supported by the trainer's
`--resume --extend-to-epochs N` arguments with the unchanged frozen config; it is
not submitted automatically. Never resume from the historical v1 directory.

## Local verification

Two CPU tests passed: same v1/v2 initialization, exact main-output dispatch and
finite gradients reaching all heads; optimizer/RNG checkpoint round-trip produces
the same next update with dropout. Python syntax and Bash syntax passed. CUDA
training and real Validation for v2 have not yet run. No performance improvement
is claimed before those records exist.
