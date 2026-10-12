# Improved V3: bounded V2 continuation

## Current status update - 13 October 2026

**This update supersedes the historical execution-status statements below.**

The V3 E12 continuation and selection have subsequently produced a
selected parent checkpoint and selection receipt on UQ Rangpur.
Their recorded SHA-256 values have been verified.

- Selected parent:
  `improved_v3_split_e12_v1/B/training_epoch12_pending_validation.pt`
- Parent SHA-256:
  `bb0d03c7a047ed187fadb5e1bf6ac1776448b351621764e9b6f0ff1fb67660d4`
- Selection receipt:
  `improved_v3_split_e12_v1/C/selection.json`
- Receipt SHA-256:
  `fffe08a6c6502989e1ad033a0b78ea5117377dbad2de762791a3a895b2e189da`

The historical proposals, estimated GPU budgets and earlier
NOT EXECUTED statements below are retained for provenance; they
must not be interpreted as the current V3 execution status.

The subsequent V4 E20 paired experiment has completed bounded CPU
verification but has not started GPU training or produced V4
Validation or Held-out Test results.


Status: local implementation; Rangpur preflight, training and Validation are **NOT EXECUTED**. GPU budget approval is required. V4/V5 are deferred.

Local verification (12 October 2026): WSL Python 3.11.15 / PyTorch 2.12.1+cu130, CPU. Final focused suite: **19 tests in 8.064 s, OK**. Relevant original 107 tests plus 19 V3 tests: **126 tests in 10.655 s, OK**. Synthetic/mocked I/O only; NiBabel is missing locally. No real NIfTI/CUDA acceptance or new segmentation result is claimed. New source AST and configuration JSON parsing passed.

Transfer fingerprints of the tested files:

~~~text
60c01eb058a73d270174717699234b7e9e83b2eac2006401d52c525ca00e0ad1  train_improved_v3.py
e860138e6f411c40c3cb843ff97cf0fc9d2a942025741ad37789bcc4f28702c8  test_improved_v3.py
806c1ce2a7e2e7019ac39c19008e9c221a47ac792da89459bc173ed87fd55315  improved_v3_config.json
~~~

## Evidence and decision

V2 E10 / 1,430 updates: Bones pooled recall 0.275602, precision 0.528495; Rectum recall 0.375633, precision 0.597848; Prostate recall 0.646212, precision 0.340148. These follow from student-supplied pooled counts. Bones appeared in 86.01% of reconstructed training patches; the existing sampler already uses a 0.5 prostate-aware branch. Missing class exposure is not a sufficient explanation.

V2 weighting recovered minority overlap while Body/Background Dice declined. Continued optimisation is plausible but unproven. Missing early V2 log records preclude a complete convergence curve; one endpoint cannot establish a plateau. V2 per-patient macro scores are available, but its original per-volume matrices and panels are not local. Preflight therefore authenticates and recomputes the original remote Validation JSONL. Older E10 panels cannot establish V2 boundary causes. Resolution, context, loss competition and aggregation remain hypotheses; none is changed here.

**Hypothesis:** two additional epochs with V2's continuing optimiser/RNG state improve minority-class Validation overlap without severe class regressions. Independent variable: update budget, initially 1,430 -> 1,716. Architecture, AdamW, LR 0.001, weight decay 0.00001, weighted CE, foreground Dice, auxiliary weights, patch shape, crop algorithm, split, preprocessing and evaluator remain fixed. Sampler epochs 10 and 11 correspond to human epochs 11 and 12.

The local assessment PDF one directory above Git, Section 2.2, printed page 6, specifies Hard 3D Improved U-Net with residual units/deep supervision and Test Dice >=0.7. That row does not specify aggregation. Filename v2.2 and cover Version 2.0 Draft differ. A public official-site search did not locate a newer 2026 specification; Blackboard/tutor clarification remains necessary. All-six-class Validation Dice >=0.70 is an aspiration, not a verified per-class marking rule. The retrieved public UQ profile concerned 2024 and was not treated as current authority.

## Implementation and selection

New code/config/tests: train_improved_v3.py, improved_v3_config.json, test_improved_v3.py. Original production files remain unchanged.

- Parent bytes are pinned to 3229bdb38afa7b83af04e79cd7f882059a3510963eff29957554a5438d490e9f. The existing protected V2 loader checks model, configuration/source hashes, strict weights and E10/1,430 identity.
- Baseline Validation log is pinned to d77e0c53c13a53b8a0af0dd22cf45f6f64798c82a077f597ba9aaa29a80a215e. Preflight requires setup + 38 successful distinct Validation volumes + summary + PASS; it recomputes counts/metrics and verifies baseline selection references.
- Initial continuation uses the existing V2 resume helper to restore model, AdamW and Python/NumPy/PyTorch/CUDA/DataLoader RNG states. Runtime versions, training path manifest, sampling, GPU model/count and source hashes must match. CUDA stays best-effort warn-only; bitwise GPU reproducibility is not promised.
- V3 has its own identity, parent lineage, selection/schedule, config digest and source hashes. Subsequent resume requires exact config equality and a complete saved boundary.
- Full Validation reuses _process_pair/summarize, 64-cube windows, stride 32, mean-softmax, resampled XYZ truth and original absent-class handling. Validation restores model mode and every RNG stream before training continues.
- Selection score: mean **patient-macro Dice for Bones, Rectum and Prostate**. A candidate must exceed the current best by >1e-6; pooled Dice drops from V2 must not exceed [0.02,0.03,0.02,0.03,0.02,0.02] for labels 0..5; classes 2..5 must have TP-positive coverage in all 38 volumes. Ties retain the earlier model. These are engineering tolerances, not clinical/course thresholds.
- A meaningful-improvement flag additionally requires score gain >=0.03 and minimum foreground pooled Dice gain >=0.03 over V2. Smaller eligible gains can be retained provisionally. V2 remains selected when no candidate qualifies.
- Report all six Dice/IoU, pooled precision/recall/FP/FN/coverage, volume macro, patient macro, foreground mean/minimum pooled Dice, per-volume timing and GPU peaks. Foreground means labels 1..5.
- Integrated Validation keeps the training optimiser and gradient buffers resident. Its measured GPU peaks therefore describe this integrated process and must not be presented as directly comparable to the original standalone Validation memory. Prediction mathematics remains shared.
- Default stop: E12. Later approved stops: E14/E16/E18/E20, with full Validation at each boundary. A rejected E12 candidate does not prove longer training cannot help; review trajectories and budget before any extension.

## Checkpoints and logging

Each saved epoch uses an immutable improved_unet_v3_epochNN.pt outside Git/data. Its epoch_result includes best-so-far selection and scheduled Validation. Atomic selection.json is a convenience pointer with the selected checkpoint hash; checkpoint state governs resume. V2 can remain selected and is never overwritten.

Scheduled Validation precedes that epoch's save. Failure/interruption requires resuming the previous saved boundary and replaying the unsaved epoch; report that extra compute. No partial epoch is completed. If saving succeeds before logging/pointer interruption, the complete checkpoint remains resumable. Existing epoch files cannot be overwritten.

An exclusive active.lock prevents concurrent writers. After abrupt termination, verify squeue/sacct and saved boundaries before manually removing a stale lock. Never remove a live job's lock.

Every invocation creates a unique program JSONL using exclusive creation, O_APPEND, fsync per record, sequence numbers and a SHA-256 chain. Size/inode checks detect common external truncation/replacement; final audit rejects NULs/partial lines/chain mismatch. Slurm stdout is a separate .out file. These safeguards prevent program log reuse and detect corruption; V2's NUL cause remains unknown and storage failures cannot be ruled out.

## Proposed budget — approval required

V2 took 01:55:48 for ten epochs: approximately 11m35s/epoch. Full Validation took 8m10s. Linear projection for two more epochs + Validation is **about 31m20s**, excluding startup, journalling, queues and filesystem variability. This is not measured V3 runtime.

Request one A100-PCIE-40GB and two CPUs:
1. One-step acceptance: 19 minutes on a100-test if test-only eligibility passes.
2. E11-E12 plus full Validation: 45 minutes on normal a100 after acceptance.
Maximum requested wall-time: **64 GPU-minutes**. No automatic retries or later stages.

Two checkpoints should occupy approximately 138 MB if comparable with V2. Allow >=1 GB free for the possible E11-E20 schedule (~690 MB), temporary saves and logs; check quota as well as filesystem capacity. No cache. Inspect current partitions/QoS/jobs before submission.

## Windows PowerShell — transfer

All remote commands below are **proposed, NOT EXECUTED**.

~~~powershell
Set-Location 'D:\Desktop\UQ26s2-COPMP3710-Pattern-Recognition-and-Analysis\Final project'
$login = 's4906923@rangpur.compute.eait.uq.edu.au'
$project = 'recognition/hipmri_3d_improved_unet_hard'
$files = @("$project/train_improved_v3.py", "$project/improved_v3_config.json", "$project/test_improved_v3.py")
$files | ForEach-Object { Get-FileHash -Algorithm SHA256 -LiteralPath $_ }
scp -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive $files "${login}:/home/Student/s4906923/comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard/"
if ($LASTEXITCODE -ne 0) { throw 'V3 transfer failed' }
~~~

Expected: three completed transfers and matching local/remote hashes. This establishes delivery only.

## Rangpur login Bash — CPU preflight, tests and resource inspection

~~~bash
cd "$HOME/comp3710_hipmri_verify"
PY="$HOME/miniconda3/envs/torch/bin/python"
V2_RUN="$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1"
V3_RUN="$HOME/hipmri_training_runs/improved_v3_v2_continuation_v1"
sha256sum recognition/hipmri_3d_improved_unet_hard/{train_improved_v3.py,improved_v3_config.json,test_improved_v3.py}
sha256sum "$V2_RUN/improved_unet_v2_last.pt" "$V2_RUN/validation_652948.jsonl"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m recognition.hipmri_3d_improved_unet_hard.train_improved_v3 \
  --parent-checkpoint "$V2_RUN/improved_unet_v2_last.pt" \
  --v2-validation-log "$V2_RUN/validation_652948.jsonl" --preflight-only
~~~

Expected: pinned hashes and protected CPU preflight PASS. Reads filenames, log records and checkpoint tensors; no MRI arrays or CUDA. Mismatches stop execution. Never substitute an unauthenticated parent or summary.

~~~bash
CUDA_VISIBLE_DEVICES='' "$PY" -B -m unittest \
  recognition.hipmri_3d_improved_unet_hard.test_dataset \
  recognition.hipmri_3d_improved_unet_hard.test_standard_unet \
  recognition.hipmri_3d_improved_unet_hard.test_improved_unet \
  recognition.hipmri_3d_improved_unet_hard.test_training \
  recognition.hipmri_3d_improved_unet_hard.test_improved_integration \
  recognition.hipmri_3d_improved_unet_hard.test_validation \
  recognition.hipmri_3d_improved_unet_hard.test_prediction \
  recognition.hipmri_3d_improved_unet_hard.test_improved_v2 \
  recognition.hipmri_3d_improved_unet_hard.test_improved_v3 -q
squeue -u "$USER"
scontrol show partition a100-test
scontrol show partition a100
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS,DefaultQOS
sacctmgr show qos format=Name,MaxWall
df -h "$HOME/hipmri_training_runs"
sbatch --test-only --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:19:00 --job-name=hipmri-v3-smoke --wrap='true'
sbatch --test-only --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:45:00 --job-name=hipmri-v3-e12 --wrap='true'
~~~

Expected: unittest OK and current eligibility/resource evidence. Test-only is not a submitted job. If a100-test is unavailable, seek approval for normal a100 with the same diagnostic limit. Missing remote test files require targeted transfer, not weakening tests.

## Prepare output directories and script after budget approval

~~~bash
(
set -euo pipefail
mkdir "$V3_RUN"
mkdir "$V3_RUN/diagnostic" "$V3_RUN/training"
cat > "$V3_RUN/run_v3.sh" <<'BASH'
#!/bin/bash
set -euo pipefail
cd "$HOME/comp3710_hipmri_verify"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY="$HOME/miniconda3/envs/torch/bin/python"
V2_RUN="$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1"
V3_RUN="$HOME/hipmri_training_runs/improved_v3_v2_continuation_v1"
case "${1:?diagnostic or training required}" in
  diagnostic) ARGS=(--run-dir "$V3_RUN/diagnostic" --diagnostic-one-step) ;;
  training) ARGS=(--run-dir "$V3_RUN/training" --stop-after-epoch 12) ;;
  *) exit 2 ;;
esac
exec "$PY" -u -B -m recognition.hipmri_3d_improved_unet_hard.train_improved_v3 \
  --parent-checkpoint "$V2_RUN/improved_unet_v2_last.pt" \
  --v2-validation-log "$V2_RUN/validation_652948.jsonl" "${ARGS[@]}"
BASH
)
~~~

Expected: a new isolated directory. If it already exists, inspect it; do not overwrite earlier runs.

## One-step A100 acceptance — only after approval and CPU PASS

~~~bash
DIAG_JOB=$(sbatch --parsable --partition=a100-test --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:19:00 \
  --job-name=hipmri-v3-smoke --output="$V3_RUN/diag_%j.out" --error="$V3_RUN/diag_%j.err" "$V3_RUN/run_v3.sh" diagnostic)
printf '%s\n' "$DIAG_JOB"
sacct -j "$DIAG_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
~~~

After Slurm completion:

~~~bash
cat "$V3_RUN/diag_${DIAG_JOB}.out"
cat "$V3_RUN/diag_${DIAG_JOB}.err"
"$PY" -B - "$V3_RUN/diagnostic" <<'PY'
from pathlib import Path
import sys
from recognition.hipmri_3d_improved_unet_hard.train_improved_v3 import audit_journal, sha256
root = Path(sys.argv[1])
logs = list(root.glob('events_*.jsonl'))
assert len(logs) == 1, 'Review multiple attempts individually'
rows = audit_journal(logs[0])
d = [r for r in rows if r['event'] == 'diagnostic']
assert len(d) == 1 and d[0]['processed_steps'] == 1
assert d[0]['epoch_completed'] is False and d[0]['checkpoint_saved'] is False
assert d[0]['finite_gradient_parameter_tensors'] == 77
assert d[0]['changed_parameter_tensors'] > 0
assert rows[-1]['event'] == 'result' and rows[-1]['status'] == 'PASS'
assert rows[-1]['parent_unchanged'] and rows[-1]['test_arrays_decoded'] == 0
assert not list(root.glob('*.pt')) and not (root/'selection.json').exists()
print(d[0])
print('PASS: one-step continuation acceptance', sha256(logs[0]))
PY
~~~

Expected: one finite step with restored optimiser state, finite gradients, changed tensors, no checkpoint writes and valid journal. Review before submitting the full stage. This establishes executability only.

## E11-E12 and full 38-volume Validation — after acceptance

~~~bash
TRAIN_JOB=$(sbatch --parsable --partition=a100 --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:45:00 \
  --job-name=hipmri-v3-e12 --output="$V3_RUN/train_%j.out" --error="$V3_RUN/train_%j.err" "$V3_RUN/run_v3.sh" training)
printf '%s\n' "$TRAIN_JOB"
squeue -j "$TRAIN_JOB"
sacct -j "$TRAIN_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
~~~

Expected: 286 new steps, 1,716 cumulative updates, E11/E12 checkpoints, 38 successful E12 Validation records, selection.json and preserved V2 hash. Failure stops scheduled checkpoint selection.

~~~bash
cat "$V3_RUN/train_${TRAIN_JOB}.out"
cat "$V3_RUN/train_${TRAIN_JOB}.err"
"$PY" -B - "$V3_RUN/training" <<'PY'
from pathlib import Path
import json, sys
from recognition.hipmri_3d_improved_unet_hard import train_improved_v3 as v3
from recognition.hipmri_3d_improved_unet_hard.dataset import VolumePair
root = Path(sys.argv[1])
logs = list(root.glob('events_*.jsonl'))
assert len(logs) == 1, 'Review interrupted/resumed journals by saved boundary'
rows = v3.audit_journal(logs[0])
final = rows[-1]
assert final['event'] == 'result' and final['status'] == 'PASS'
assert (final['completed_epochs'], final['total_steps']) == (12, 1716)
steps = [r for r in rows if r['event'] == 'step']
assert len(steps) == 286 and {r['global_step'] for r in steps} == set(range(1431,1717))
volumes = [r for r in rows if r['event'] == 'volume']
pairs = [VolumePair(p,w,Path(m),Path(s)) for p,w,m,s in rows[0]['config']['validation_pairs']]
v3.validate_records(volumes, pairs)
computed = v3.enrich_summary(volumes)
summaries = [r for r in rows if r['event'] == 'validation_summary']
assert len(summaries) == 1
for k,v in computed.items():
    assert summaries[0][k] == v, k
for r in rows:
    if r['event'] == 'epoch':
        assert v3.sha256(r['checkpoint']) == r['checkpoint_sha256']
selected = json.loads((root/'selection.json').read_text())
assert v3.sha256(selected['checkpoint']) == selected['checkpoint_sha256']
assert final['parent_unchanged'] and final['test_arrays_decoded'] == 0
print(json.dumps(summaries[0], indent=2))
print(json.dumps(selected, indent=2))
print('PASS: E12 arithmetic and checkpoint hashes', v3.sha256(logs[0]))
PY
sha256sum "$V2_RUN/improved_unet_v2_last.pt"
~~~

Inspect Slurm COMPLETED/0:0 separately; program PASS is not proof of scheduler termination. Compare all class metrics and selection guards before approving an extension.

Recovery example **inside a separately approved A100 allocation**, only if E11 is the latest saved boundary and E12 is absent:

~~~bash
"$PY" -u -B -m recognition.hipmri_3d_improved_unet_hard.train_improved_v3 \
  --parent-checkpoint "$V2_RUN/improved_unet_v2_last.pt" \
  --v2-validation-log "$V2_RUN/validation_652948.jsonl" \
  --run-dir "$V3_RUN/training" \
  --resume "$V3_RUN/training/improved_unet_v3_epoch11.pt" --stop-after-epoch 12
~~~

Expected: restore E11 model/AdamW/RNG and run E12 plus full Validation, without overwriting E11. Failed/repeated work consumes additional budget; do not invoke recovery automatically. If the interrupted job left active.lock, verify it is no longer running before removing only that stale lock. A later E14 stage would resume E12 with --stop-after-epoch 14 after separate approval.

## Windows PowerShell — download metadata-only evidence

~~~powershell
$download = Join-Path $env:USERPROFILE 'Downloads\hipmri_e10_evidence\v3_e12'
New-Item -ItemType Directory -Path $download -Force | Out-Null
$remoteRoot = '/home/Student/s4906923/hipmri_training_runs/improved_v3_v2_continuation_v1'
scp -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive "${login}:$remoteRoot/training/events_*.jsonl" "${login}:$remoteRoot/training/selection.json" "${login}:$remoteRoot/*.out" "${login}:$remoteRoot/*.err" $download
if ($LASTEXITCODE -ne 0) { throw 'Evidence transfer failed; inspect missing outputs' }
Get-ChildItem -LiteralPath $download -File | Get-FileHash -Algorithm SHA256
~~~

Expected: original metadata outside Git, with JSONL hashes matching the remote audit. Supply Slurm output too. No MRI or checkpoint is transferred here. No V4/V5, Test access or model freeze follows automatically.

The existing V2 validation/prediction loaders intentionally remain incompatible with V3 checkpoint identity. This batch evaluates V3 through the integrated protected runner. If a V3 candidate is selected for final use, a separately reviewed prediction/Test adapter will be needed before any explicitly approved Test run.
