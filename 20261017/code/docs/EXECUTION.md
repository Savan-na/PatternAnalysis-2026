# Execution and evidence checklist

## Evidence identity

Use the Python 3.11 `torch` environment. Record Python, PyTorch, CUDA, NumPy,
SciPy and NiBabel versions in every run manifest, along with the source ZIP or
Git revision and the exact patient-split/configuration hashes. The current
Rangpur G2 commands are maintained in [RANGPUR_COMMANDS.md](RANGPUR_COMMANDS.md).

## G1: raw integrity gate

- Confirm the current Rangpur dataset root by read-only inspection; the old
  path in previous instructions is not presumed valid. Never copy raw data
  into Git.
- Inventory all 211 filename pairs and confirm the frozen 143/38/30 volume
  split. This inventory lists Test filenames only; it must not decode Test
  arrays.
- Check the completed full Train audit and its case warnings in the
  [formal research report](../../results/reports/HipMRI_Research_Report.md).
  A two-case smoke run alone is not a full-audit pass.
- Keep Validation labels out of distribution analysis and tuning. Any later
  Validation integrity check must be limited to what is needed to confirm
  file readability and geometry; it must not drive preprocessing or model
  choices.
- Confirm affine, units, orientation, shape, six-class ID range, intensity
  finiteness, and file hashes on Train data. Record unknown units or
  conflicting header transforms for review rather than silently correcting
  them.

## G2: completed Train-only raw EDA gate

- Reconcile the derived class, patient, volume, intensity and spacing tables
  with the complete G1 JSON. Keep historical statistics distinct from the
  current raw-audit result.
- On Rangpur CPU, smoke-test then measure axial class coverage and nonzero MRI
  intensity histograms for all Train cases. Verify case identity and source
  hashes; transfer approved derived statistics only.
- Review organ-size and longitudinal differences, boundary contrast where
  measurable, and the limits of selected visual overlays. Record supported,
  excluded and unresolved hypotheses before entering G3.

## G3: current existing-pipeline diagnosis

- Audit the actual implemented Train path: `load_case`, factor-two
  `downsample`, `zscore_nonzero`, and whole-volume `to_3d_model_arrays`.
  Record label counts, physical organ volumes, axial-slice coverage, image
  finiteness, normalized nonzero mean/std and padded-background voxels.
- Use a two-case Rangpur CPU smoke run before the full 143-case audit. Preserve
  the G1 source hashes and frozen patient split; do not process Validation/Test
  label distributions for these decisions.
- The restart currently has no ROI crop, augmentation, Patch Sampler or formal
  training loop. Specify and freeze a candidate sampler before measuring its
  patch exposure. Do not present raw-slice presence as a patch-hit rate.

### G3 candidate crop comparison

The current model adapter supplies complete downsampled volumes; no existing
training sampler is available in this checkout. Before selecting a new policy,
audit two *candidate* patch streams on the same 143 Train cases with the
frozen `config/g3_patch_audit.json`: ten 64×64×32 XYZ crops per volume per
arm, using seed 371017. The uniform arm places all crops randomly. The
organ-aware candidate uses four identical uniform crops plus two crops each
anchored on a randomly selected Bladder, Rectum, and Prostate voxel. Case
frequency is held equal between arms to isolate crop-placement effects; this
design still weights patients with more visits more heavily. Record class
patch-hit rate, pooled voxel share, and the fraction of patches containing an
entire organ. Run a two-case smoke audit before the full CPU cohort. These
are hypothetical exposure measurements, not historical or actual training
exposure, and cannot establish model performance without later Validation.

## Later gates

- G4: dry-run one scan per Train patient per epoch through the same
  downsampling, normalization and paired patch conversion that GPU training
  will use. Compare uniform and organ-aware candidates, record class hits,
  voxel shares and patient counts from delivered masks. Repeat measurement
  during real training after any augmentation; a CPU dry run alone is not
  actual optimization exposure or evidence of improved Validation Dice.
- Confirm every foreground class present before downsampling remains present
  afterward, and quantify per-class physical-volume and slice-coverage
  changes, or stop and revise the spatial resolution before any model run.
- Benchmark a complete downsampled volume on allocated GPU. Fix the final
  shape, padding, batch size and training budget in a hashed run config.
- Check both primary 3D models' forward and backward passes, then a few-case
  Train-only overfit including prostate and rectum.

## Outputs still required

Full training checkpoints and per-step logs; complete Validation predictions;
frozen selection receipt; one-time Test metrics; 3–5 selected failure-case
panels; measured latency/VRAM; README report; final Git commit and Pull
Request. These are not present merely because this checklist exists.
