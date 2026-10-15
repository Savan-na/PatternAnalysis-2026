# HipMRI experiment history: Baseline to V4

This is the current research timeline for the six-class HipMRI segmentation experiments. The [recognition overview](../README.md) is the navigation entry point; the [Final Report](../FINAL-REPORT.md) interprets the E10 and V4 Validation results; the [model package README](../hipmri_3d_improved_unet_hard/README.md) documents implementation. Evidence described as local-only is preserved in `research_archive/`, which is excluded from Git. The original prospective plans and commands remain there as historical records.

**Status (15 October 2026):** V4 E20 Control/Treatment Validation completed. Treatment was rejected after three mandatory gates failed despite a 32.30% reduction in *pooled* Prostate false positives. Control E20 was not automatically selected. V3 E12 remains the protected candidate; formal model freeze is incomplete. Held-out Test has not been evaluated, and the current `predict.py` has not been validated as the final V3 E12 inference path.

An isolated [V3 E12 prediction adapter](../hipmri_3d_improved_unet_hard/predict_v3.py) has since been implemented and synthetically tested. Its real checkpoint/receipt/lineage preflight and Validation equivalence remain unexecuted because the necessary remote artefacts and dataset are unavailable locally. This does not change the protected-candidate or Test status.

## Common evaluation and evidence boundaries

The patient-disjoint split contains 143 Train volumes from 26 patients, 38 Validation volumes from six patients, and 30 held-out Test volumes from six patients. All results below are **Validation**, not Test. Full-volume evaluation uses resampled XYZ volumes, 64³ sliding windows with stride 32, averaged softmax probabilities and six-class confusion matrices. *Pooled* metrics combine voxel counts before scoring; *volume-macro* averages volume scores; *patient-macro* first pools each patient's Weeks, then averages the six patients. They are distinct estimands. Both-absent classes are excluded from macro averages; one-sided absence scores zero. Sources and receipts have different assurance levels: code confirms the implemented protocol, original local logs support direct arithmetic checks, and some later remote outcomes are supported by authenticated student-supplied records without locally inspecting checkpoint payloads. See the [traceability register](evidence/traceability.md) and [case evidence manifest](evidence/failure-cases.md).

## Standard 3D U-Net Baseline — E10

1. **Objective / hypothesis:** establish the conventional architecture and ten-epoch comparison point for six-class segmentation.
2. **Architecture / relationship:** standard encoder–decoder 3D U-Net; the Improved variants change the architecture, so this is not a loss-only control for V2–V4.
3. **Training / changed variables:** ten completed epochs, 1,430 updates, same 143-volume Train split, 64³ patch policy and preprocessing used for the original E10 comparison. Original Standard loss is the reference objective.
4. **Execution:** training and full Validation completed remotely; Validation job 649669 recorded 38/38 successes. The inspected original log resumes at epoch 4; earlier step events are not reconstructed.
5. **Evaluation:** 38 volumes/six Validation patients, common full-volume evaluator and absent-class policy.
6. **Results:** pooled Dice: Background 0.941342, Body 0.842790, Bones 0, Bladder 0.622004, Rectum 0, Prostate 0. Patient-macro Dice: 0.935937, 0.835655, 0, 0.583946, 0, 0 in that class order. These values do not imply Test performance.
7. **Decision:** retained as the architectural baseline, not the protected final candidate.
8. **Evidence:** original E10 JSONL authentication and counts in the [manifest](evidence/failure-cases.md#original-e10-log-authentication); [Final Report Section 5.1](../FINAL-REPORT.md#51-controlled-validation-comparison). The local-only historical comparison is `research_archive/10_scientific_documentation/COMP3710_P4-03C_docs_20261014/BASELINE_V4_COMPARISON.md` (its E20 status is dated and superseded).
9. **Limits:** original Standard Validation-time checkpoint hash is absent, and early training-log segments are missing. No per-class causal explanation follows from the overlap scores.

## Improved U-Net V1 — E10

1. **Objective / hypothesis:** test whether residual units and deep supervision improve the original E10 segmentation.
2. **Architecture / relationship:** Improved 3D U-Net adds residual units and auxiliary supervised heads to Standard; these changes occurred jointly.
3. **Training / changed variables:** original E10 objective with the Improved architecture, ten completed epochs and 1,430 updates; same split, patch policy and full-volume evaluator as Standard.
4. **Execution:** training and full Validation completed remotely; successful replacement Validation job 650393 recorded 38/38 after job 649912 was cancelled. The inspected training log resumes at epoch 2.
5. **Evaluation:** the same 38 Validation volumes and six patients as Standard.
6. **Results:** pooled Dice: Background 0.976012, Body 0.906781, Bones 3.943109603265999e-7, Bladder 0.768230, Rectum 0, Prostate 0. Patient-macro Dice: 0.973940, 0.904107, 0, 0.711025, 0, 0. The tiny positive Bones score is not useful segmentation. Four purposively selected paired cases include both improvements and regression.
7. **Decision:** improved some large-organ overlap but did not resolve minority-class omission; not retained as the current protected candidate.
8. **Evidence:** [Final Report Sections 4–5](../FINAL-REPORT.md#5-controlled-validation-comparison-and-failure-case-evidence), authenticated E10 logs and [case manifest](evidence/failure-cases.md). Job 650736 reconstructed Train exposure; Job 650924 measured selected-patch probabilities, not parameter gradients or full-volume performance; see [class-collapse analysis](analysis/class-prediction-collapse.md).
9. **Limits:** residual units and deep supervision cannot be assigned separate causal effects. Selected patches and four cases do not identify the cause of failures or population-wide anatomy.

## Improved V2 — weighted-loss E10

1. **Objective / hypothesis:** recover minority-class overlap by changing the class weighting in the cross-entropy component.
2. **Architecture / relationship:** retains the Improved V1 architecture and deep supervision; changes the loss rather than the network design.
3. **Training / changed variables:** inverse-square-root class-weighted CE with the original foreground Dice, AdamW; E10/1,430-update endpoint. The common patient split, preprocessing, patch sampler and evaluator remain the comparison frame.
4. **Execution:** the student-recorded remote E10 training and Validation completed. Original per-volume V2 matrices and checkpoint payload are not independently present locally in this documentation repository.
5. **Evaluation:** recorded full Validation on the same 38 volumes/six patients; distinguish recorded remote results from independently inspected E10 original logs.
6. **Results:** recorded pooled Dice: Background 0.953976, Body 0.862141, Bones 0.362280, Bladder 0.826771, Rectum 0.461378, Prostate 0.445695. Recorded patient-macro Dice: 0.948549, 0.854755, 0.361053, 0.779908, 0.439592, 0.453977. The V2 record gives pooled Bones/Rectum/Prostate recall 0.275602/0.375633/0.646212 and precision 0.528495/0.597848/0.340148.
7. **Decision:** V2 E10 supplied the authenticated parent identity for the bounded V3 continuation; its minority-class overlap improved relative to V1 while some large-class scores declined.
8. **Evidence:** the historical local-only V2 comparison is `research_archive/10_scientific_documentation/COMP3710_P4-03C_docs_20261014/BASELINE_V4_COMPARISON.md`; student-recorded details remain in the [AI/provenance record](../hipmri_3d_improved_unet_hard/ai.md). V2 parent checkpoint SHA-256 is `3229bdb38afa7b83af04e79cd7f882059a3510963eff29957554a5438d490e9f`; its Validation log SHA-256 is `d77e0c53c13a53b8a0af0dd22cf45f6f64798c82a077f597ba9aaa29a80a215e` in the V3 preflight record.
9. **Limits:** missing original local V2 per-volume matrices/panels and early log records constrain independent verification and convergence claims. E10 alone does not establish a plateau or causal mechanism.

## Improved V3 — bounded E12 continuation

1. **Objective / hypothesis:** test whether two more V2 optimisation epochs improve minority-class Validation overlap without unacceptable regressions.
2. **Architecture / relationship:** same Improved architecture, weighted CE, foreground Dice and auxiliary weights as V2; V3 changes update budget and continues V2 state.
3. **Training / changed variables:** AdamW learning rate 0.001, weight decay 0.00001, same 64³ patches, sampler, preprocessing, split and evaluator. V2 E10/1,430 updates continued through E11 and E12 to 1,716 updates, with restored model, optimiser and RNG state. Stage A trained E11/143 updates to 1,573; B trained E12/143 to 1,716; C performed 38-volume Validation and selection with zero updates. A/B checkpoints carried `validation_pending=True`; C did not rewrite B. Split identity: `improved_v3_split_e12_v1`.
4. **Execution:** the earlier proposal and split-scheduling text are historical. Subsequent remote execution produced the protected E12 parent and C selection receipt; the recorded SHA-256 identities were verified. The checkpoint binary itself was not independently opened locally.
5. **Evaluation:** C used the original 38-volume/six-patient full Validation protocol, mean-softmax sliding windows and absent-class handling. Selection score was mean **patient-macro** Dice for Bones, Rectum and Prostate. An eligible candidate had to improve the best score by >1e-6, stay within V2 pooled-Dice drop allowances `[0.02,0.03,0.02,0.03,0.02,0.02]` for labels 0–5, and have TP-positive coverage for labels 2–5 in all 38 volumes. Ties and failed candidates retained V2. A separate meaningful-improvement flag required score and minimum foreground pooled-Dice gains of at least 0.03; this flag was not a course/clinical threshold.
6. **Results:** the authenticated historical comparison records V3 E12 pooled Dice 0.969837, 0.868735, 0.476785, 0.821240, 0.537022, 0.426721 and patient-macro Dice 0.966336, 0.866039, 0.476872, 0.785414, 0.529170, 0.415424 (Background through Prostate). These are Validation observations, with pooled Prostate Dice below V2 while other minority-class values improve.
7. **Decision:** E12 was selected and remains the protected continuation candidate after the V4 decision. This is not a formal final-model freeze.
8. **Evidence:** B checkpoint SHA-256 `bb0d03c7a047ed187fadb5e1bf6ac1776448b351621764e9b6f0ff1fb67660d4`; C `selection.json` SHA-256 `fffe08a6c6502989e1ad033a0b78ea5117377dbad2de762791a3a895b2e189da`. The complete original V3 experiment and three-stage execution plans are preserved byte-for-byte in the local-only `research_archive/07_model_development_snapshots/P5-08_superseded_active_docs_2026-10-15/` with a provenance manifest. The local-only P4-03C comparison and later V4 decision provide independent journal context.
9. **Limits:** stage restarts do not promise bitwise GPU equality; original checkpoint payload inspection and a verified V3 E12 `predict.py` final-inference adapter remain outstanding. No Held-out Test or clinical claim follows.

## V4 — paired E20 Control/Treatment

1. **Objective / hypothesis:** reduce Prostate false positives through a controlled change to its overlap loss while protecting recall and other classes.
2. **Architecture / relationship:** both arms continue the same selected V3 E12 parent and Improved architecture. Control retains original foreground Dice (equivalent Prostate alpha 0.5); Treatment substitutes Prostate soft Tversky alpha 0.7, beta 0.3. Other class terms and architecture are shared.
3. **Training / changed variables:** frozen paired E13–E20 protocol, same parent, split, preprocessing, patch policy, CE, AdamW and evaluator. The Prostate overlap term is the planned treatment variable. Scheduled full Validation at E14/E16/E18/E20.
4. **Execution:** both arms completed the scheduled Validation checkpoints. E20 is the frozen primary paired decision point; the E14 draft snapshot is historical, not the E20 implementation.
5. **Evaluation:** common 38-volume/six-patient exploratory Validation. The three mandatory acceptance gates use **patient-macro** Prostate Dice gain, **pooled** Prostate Recall loss and maximum other-class **patient-macro** Dice drop. Pooled, volume-macro and patient-macro results must not be interchanged.
6. **Results:** Treatment reduced **pooled** Prostate false positives by 32.30%. E20 pooled Prostate Dice was Control 0.617164 versus Treatment 0.650778; patient-macro Prostate Dice was 0.616010 versus 0.643060. Three gates failed: patient-macro gain 0.027050 < 0.03; pooled Recall loss 0.076124 > 0.05; maximum other-class patient-macro Dice drop 0.048440 > 0.02 (Rectum).
7. **Decision:** frozen conjunctive rule **rejected Treatment**. Control E20 was not automatically selected; V3 E12 remains protected. Formal freeze and Held-out Test are still pending.
8. **Evidence:** [Final Report V4 section](../FINAL-REPORT.md#v4-e20-completed-paired-validation-experiment). Local-only: `research_archive/10_scientific_documentation/COMP3710_V4_FINAL_DOCS_20261014/V4_ACCEPTANCE_DECISION.md`, `research_archive/08_validation_and_independent_audit/COMP3710_P4-04C_audit_20261014/INDEPENDENT_AUDIT.md`, and `research_archive/08_validation_and_independent_audit/COMP3710_P4-04E_20261014/V4_EVIDENCE_CLOSURE.md`.
9. **Limits:** the six Validation patients informed model development and parent selection, so this is exploratory comparison, not independent Test confirmation. Local checkpoint-payload inspection remains unknown, and existing `predict.py` is not an established V3/V4 final-inference path.
