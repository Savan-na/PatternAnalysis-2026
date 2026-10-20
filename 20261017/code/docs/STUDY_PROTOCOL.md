# HipMRI research protocol — data-first revision

## Current execution scope — 20 October 2026

The G/P labels below describe the study's evidence history, not multiple current
training queues. The current controlled comparison is the [frozen v4 A0/A1
protocol](V4_Paired_Training_ZH.md): the same GN network, AdamW/cosine and budget,
with spatial augmentation as the only between-arm change. The two jobs were
last confirmed running. Use each model's best predeclared Validation candidate
for quality comparisons; do not force the final epoch or use Test to select it.
The earlier staged rationale remains below as historical methodology.

## Research question and evidence order

The project asks whether raw data integrity, class imbalance, patient
variation, preprocessing, or actual patch exposure explains failures on
Rectum and Prostate, and which controlled change improves them. The sequence
is G1 raw integrity, G2 Train-only statistical EDA, G3 pipeline and sampling
diagnosis, G4 evidence-based intervention, G5 controlled 3D model experiments,
then G6 final evaluation and report. G1 does not train models or change loss,
sampler, network, or raw files.

The primary baseline is standard 3D U-Net. Improved 3D U-Net is the primary
comparison. The existing 2D U-Net is background research and does not require
retraining. The historical Baseline–V4 outcomes remain historical evidence,
not results of the restarted study.

The user subsequently requested a supplementary 2D training arm. Its axial
slice and crop exposure was measured on Train only before freezing a separate
organ-aware sampling and checkpoint protocol. This optional arm does not
change the frozen Standard/Improved 3D comparison. A 2D-versus-3D result
cannot isolate architecture when their input context and sampling differ.

## Cohort and information boundaries

The frozen split has 26/6/6 patients and 143/38/30 Train/Validation/Test
volumes. All visits from one patient remain in one partition. Verify this
against current filenames before processing arrays. Train masks alone provide
class distributions, candidate class weights, sampling choices, and other
data-dependent settings. Validation is used for model selection and failure
analysis after decisions are set. The held-out Test is evaluated only after
the checkpoint, code, preprocessing, and metric rules are frozen.

G1 decodes all current Train MRI/mask pairs. It checks readability, hashes,
non-finite and constant images, label IDs, shape, affine, orientation, spatial
units, spacing, and suspicious anatomy geometry. Validation/Test filenames
may be inventoried for pair and patient isolation; their label distributions
do not inform G1–G4 decisions. A mechanical pass is not proof of correct
anatomical labels. Flagged cases need MRI/mask overlays and documented review.

## Controlled experiments and endpoints

After G1–G4, compare standard and improved 3D U-Net on the same split,
preprocessing, sampling, loss, training budget, and metrics when estimating
architecture effects. If loss or sampling differs, call the result a
comparison of training schemes. Change one main factor per targeted follow-up.
Record class exposure, per-class Dice and recall, prediction counts, loss,
learning rate, gradient norms, time, and memory during training.

Use Validation to select checkpoints and classify small-organ errors as
missing, displaced, oversized, or boundary failures. Report Dice, IoU,
precision, recall, HD95, patient variation, prostate apex/base behaviour,
qualitative overlays, model size, inference time, and memory. Surface
distances are measured in millimetres on a common physical grid; one-empty
predictions are explicitly counted. Report both segmentation quality and
compute cost on the final held-out Test without tuning on its results.

The current evidence status is recorded in the single formal research report
rather than this methods protocol. Both 3D runs completed a strictly checked
continuation from 40 to 50 epochs and selected epoch 50 by the frozen
patient-balanced Validation rule. The supplementary 2D arm completed its
separate protocol and a strictly checked continuation to epoch 50. The two
3D checkpoints then underwent one held-out Test evaluation; Test findings
must not drive model revisions.

## Assessment and independent-review update — 20 October 2026

The supplied local assessment draft explicitly requires volumetric residual
units and deep supervision for the selected Hard 3D Improved U-Net option.
Any final candidate must retain these mechanisms; an output-fusion ablation
can use separately supervised auxiliary losses with a main inference head.
The complete requirement mapping and source identity are in the
[P1 review](../../results/reports/DATA_AND_METHODS_REVIEW_ZH.md#p1).
G7 passed its Train exposure gates, but this does not establish a causal
explanation for the Body errors. First compare fixed-weight full-volume and
training-sized window inference on predefined Validation cases, together
with logit-margin diagnostics. Retain edge sampling as a candidate, revise
the sequence when evidence warrants, and keep course boundary/cost analysis
and final delivery progressing alongside bounded model development.
