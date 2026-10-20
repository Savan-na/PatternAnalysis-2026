# Best-checkpoint assessment audit

This supersedes the unsubmitted four-model generalization package. It evaluates
five existing models (2D, Standard, v1, v2, v3), each at its own best recorded
Validation checkpoint within the 50-epoch training horizon. Selection is locked
from the existing 5-epoch Validation schedule, not Test or training loss.
Current selected epochs: 50, 50, 50, 45, 50. The evaluator independently rechecks
all 50 records and verifies best.pt against the selected immutable epoch weights.

One CPU test gate and one A100 task, with a 45-minute limit for 340 model/case
comparisons plus metrics and figures. Rough estimate 15–30 minutes, not a runtime
guarantee. No training, scheduler change, threshold tuning or old-file overwrite.
All Validation matrices and historical Standard/v1 Test matrices must reproduce.
Exact raw-MRI SHA cross-split duplicate checking is included, but does not exclude
near duplicates or identical images with different file encodings.

Measurements: six-class confusion (Dice/IoU/precision/recall derived locally),
five foreground HD95/ASSD with explicit empty statuses, prostate base/apex regions
and extent, prostate-in-OAR confusion, components, internal slice gaps, adjacent
area variation, distant Body FP, and one-resident-model inference VRAM/latency
with identical warm-up and repetition count. 2D and 3D recipes remain confounded;
do not attribute every difference to spatial dimension. Geometry proxies are
not clinical acceptability or dose-risk estimates.

Pictures: common GT peak-area and v3 largest-error slices across all five models,
plus physical-superior-based apex/central/base panels. No probability calibration
or surface Dice is invented from label masks. Test is already inspected and is
descriptive only. The updated Validation-only v4 plan is hash-locked beforehand;
new Test outcomes must not determine v4 choices.

Output:
`results/experiments/improved_3d_v3/best_checkpoint_audit_20261020/`.
Existing output blocks reruns. Partial metrics are kept; final metrics.json with
complete=true and checksums.json indicate success. Inspect failures, never delete
partial evidence to make a rerun appear new.

Submit once after verified extraction:

```bash
bash -n "$HOME/20261018_g1_stage/20261017/code/tools/submit_best_checkpoint_audit.sh"
bash "$HOME/20261018_g1_stage/20261017/code/tools/submit_best_checkpoint_audit.sh"
```

After downloading the entire output directory, run locally from project root:

```text
python -B code/tools/build_best_checkpoint_review.py
python -B code/tools/summarize_best_checkpoint_audit.py
```

The second command refuses missing/incomplete data and verifies downloaded
checksums. It populates the existing assessment_required and
supplementary_research sections; it does not replace historical raw metrics.
