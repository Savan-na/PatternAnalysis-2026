# Feasibility Review — HipMRI Segmentation

**COMP3710 | 20 October 2026 | Status: feasible, with outstanding validation and submission work**

## Objective and scope

Develop and evaluate a six-class HipMRI segmentation system using a 2D U-Net baseline, a standard 3D U-Net, and an improved residual 3D U-Net with deep supervision. The research objective is to improve segmentation accuracy and boundary reliability while documenting computational cost and failure cases. A clear improvement over the standard model remains an objective, not an established result.

## Evidence of feasibility

- **Data:** patient-disjoint Train/Validation/Test groups contain 26/6/6 patients and 143/38/30 volumes. The complete Train audit passed mechanical checks; image and mask affines match. Unknown mask spatial units remain a metadata limitation; mechanical agreement does not certify anatomical correctness.
- **Implementation:** all three models completed 50 training epochs on A100 resources. Training records and checkpoints support recovery; a weighted-loss diagnostic demonstrated that the standard model can learn small organs on a fixed batch.
- **Results:** the selected patient-balanced, three-small-organ Validation scores are 0.8502 (standard 3D), 0.8420 (improved 3D), and 0.7757 (2D). These are selection scores, not six-class Test Dice. Six-class results and representative segmentation failures are available locally.
- **Resources:** the initial 40-epoch 3D jobs completed in approximately 76 and 102 minutes. Additional ten-epoch runs completed successfully. Existing throughput measurements establish operability; a fair, isolated resource benchmark is still required.

## Main risk and next experiment

The improved model has not demonstrated overall superiority and produces distant Body false positives. Validation Body HD95 is substantially worse than the standard model. Training uses 64×64×32 patches, whereas current validation uses full volumes; input-dependent normalization and additive output heads are plausible contributors, not proven causes.

The four-case diagnostic completed with unchanged checkpoints and reproduced all original confusion matrices. Sliding windows reduced the improved model's distant Background-to-Body errors by 80.41% across these selected cases, but damaged small-organ Dice and did not improve every Body HD95. This supports context sensitivity, not a safe replacement inference protocol or an isolated normalization cause. Retain the original protocol and proceed toward the audited G7 sampling intervention as a controlled training experiment. Preserve residual learning and meaningful deep supervision; use Train/Validation evidence for subsequent decisions.

## Completion conditions

Complete Validation boundary/continuity analysis, standardized resource measurement, reproducible submission entry points, exact dependencies, and AI disclosure. Confirm the applicable assessment version and tutor check-off; neither is certified by this review. Previously inspected Test results remain descriptive and cannot provide a fresh blind evaluation for subsequent tuning. Raw medical data and model weights must remain outside the submitted repository.

**Decision:** the diagnostic is complete; prepare a single-factor G7 sampling comparison under the original inference protocol. See [the P1 research review](20261017/results/reports/HipMRI_P1_独立研究审查_ZH.md) for the evidence and decision gates.
