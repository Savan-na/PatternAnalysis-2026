# V3 split execution: prepared, not remotely executed

## Current execution-status correction - 13 October 2026

**The document title and earlier prepared/not-executed statements
describe the original planning stage, not the latest project state.**

The subsequent V3 E12 selected-parent checkpoint and selection
receipt were produced on Rangpur and their SHA-256 identities verified.

- Parent checkpoint SHA-256:
  `bb0d03c7a047ed187fadb5e1bf6ac1776448b351621764e9b6f0ff1fb67660d4`
- Selection receipt SHA-256:
  `fffe08a6c6502989e1ad033a0b78ea5117377dbad2de762791a3a895b2e189da`

The original instructions, proposed scheduling steps and approval
gates below remain historical documentation. They are not
instructions to repeat completed V3 work.

The V4 E20 paired Control/Treatment continuation is a separate
planned experiment. CPU tests passed within their defined scopes;
V4 GPU training, scheduled Validation and Held-out Test evaluation
remain unexecuted.


## Decision and evidence boundary

The split is technically feasible as three independent 15-minute allocations
(45 minutes total requested wall time), each below the reported 20-minute test
QoS limit. Earlier V2 training averaged approximately 11.6 minutes/epoch and
Validation took approximately 8.1 minutes. These are planning references, not
measurements of these jobs. Checkpoint authentication, process startup, storage
contention and GPU variability consume the remaining margin. A timeout is possible.

Queue advantage is UNVERIFIED. A previously idle GPU is not proof of present
availability or priority. Test-only estimates are advisory and have already failed
to predict the earlier start. Three queue waits may be worse than one. Inspect
current scheduling before approving replacement. Do not cancel job 654465 here.
If the original starts or finishes, review its evidence before duplicating work.

## Isolated implementation

- `run_improved_v3_split.py`: stages A, B and C, protected preflight and CPU evidence audit.
- `test_improved_v3_split.py`: eight synthetic tests.
- `submit_v3_split.sh`: one guarded submission per stage; no automatic chaining or retries.

Original source/configuration, original V3 scripts and historical evidence remain
unchanged. Split identity is `improved_v3_split_e12_v1`; its config includes the
original protected V3 config plus fingerprints of the new Python runner/tests.

| Stage | Input | Operation | Output |
|---|---|---|---|
| A | authenticated V2 E10 / 1430 | sampler epoch 10; 143 updates | E11 / 1573 training-only checkpoint |
| B | authenticated A boundary | sampler epoch 11; 143 updates | E12 / 1716 pre-validation checkpoint |
| C | authenticated B boundary | 38-volume full Validation; no updates | frozen selection receipt and metrics |

Both boundaries contain model, AdamW and Python/NumPy/Torch CPU/CUDA/DataLoader
RNG state. Source and full configuration compatibility are checked before restore.
The model and optimiser are instantiated before state restoration; no iterator is
created until restoration. DataLoader settings and per-epoch sampling are unchanged.
The final Validation uses the same production function, stride, aggregation and
metrics, under the original RNG-preservation context.

Training-only checkpoints explicitly contain `validation_pending=True`. The
original V3 loader is not weakened to accept them. Job C never rewrites B: a
separate authenticated journal and `selection.json` prove completed Validation.
These split checkpoints need a separately reviewed inference adapter if selected
for subsequent deployment/Test; they must not be renamed into original V3 format.

The selection objective remains patient-macro mean Dice for classes 2/4/5,
improvement >1e-6, maximum pooled Dice drops [0.02,0.03,0.02,0.03,0.02,0.02], and
TP-positive coverage in all 38 volumes for each class 2/3/4/5. V2 remains selected
unless all guards and score improvement pass. No E11 Validation or E14+ exists.

## Local verification

Eight new tests plus 76 relevant existing tests: **84 tests in 19.531s, OK**.
Environment: existing WSL PyTorch, with an in-memory NiBabel import shim because
NiBabel is absent locally. No real NIfTI decoding or CUDA acceptance was performed.
The first shell launch failed due to quoting a path containing spaces; the corrected
stdin-based test launch succeeded. This was not a test failure.

The parity test performs 286 small CPU AdamW updates using shuffled 143-sample
epochs. Serializing/restoring after the first epoch reproduces second-epoch order,
weights, AdamW states and subsequent RNG draws exactly. This is synthetic CPU
evidence, not proof of bitwise GPU equivalence. Existing warn-only MaxPool3D CUDA
nondeterminism remains; process restarts may also select different backend kernels.
The protocol and effective updates are equivalent; exact floating-point equality
on different GPU jobs is not promised.

Other tests cover invalid boundaries/configs, immutable files, no training-time
Validation, journal/checkpoint corruption, original-loader rejection of pending
E12, recomputed C selection/metrics, and bounded CLI arguments. Python AST and Bash
syntax checks passed. No remote tests or jobs have been executed by Codex.

## 1. Current scheduling inspection (read-only, run before deciding)

Environment: Rangpur login Bash. Purpose: check current queue and limits. Expected:
actual state, available nodes/QoS and an advisory estimate. Impact: no allocation.

```bash
squeue -j 654465 -o '%.18i %.12P %.12a %.10T %.10Q %.30S %.30R'
scontrol show job 654465
sinfo -p a100-test -N -l
scontrol show partition a100-test
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS,DefaultQOS
sacctmgr show qos format=Name,MaxWall
sbatch --test-only --partition=a100-test --gres=gpu:a100:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:15:00 \
  --job-name=hipmri-v3-split-check --wrap='true'
```

Do not claim a faster finish from this estimate alone. If replacement is approved,
the student must separately cancel the original and confirm CANCELLED/no running
work. No cancellation command is included or executed by this task.

## 2. Transfer only new files

Environment: Windows PowerShell. Purpose: copy the isolated runner/tests/guard.
Expected: three 100% transfers. Impact: adds new source files to the verification
workspace; does not change original files or submit jobs.

```powershell
Set-Location 'D:\Desktop\UQ26s2-COPMP3710-Pattern-Recognition-and-Analysis\Final project'
$login = 's4906923@rangpur.compute.eait.uq.edu.au'
$p = 'recognition/hipmri_3d_improved_unet_hard'
$files = @("$p/run_improved_v3_split.py", "$p/test_improved_v3_split.py", "$p/submit_v3_split.sh")
$files | ForEach-Object { Get-FileHash -Algorithm SHA256 -LiteralPath $_ }
scp -o PubkeyAuthentication=no -o PreferredAuthentications=password,keyboard-interactive $files "${login}:/home/Student/s4906923/comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard/"
if ($LASTEXITCODE -ne 0) { throw 'Transfer failed; stop' }
```

## 3. New external directory and pinned manifest

Environment: Rangpur login Bash. Purpose: verify all new files, original V3 inputs,
V2 checkpoint/evidence and Stage 1 journal. Expected: all checks OK. Impact: creates
only a new independent external directory; refuses an existing directory. No GPU.

```bash
(
set -euo pipefail
umask 077
cd "$HOME/comp3710_hipmri_verify"
ROOT="$HOME/hipmri_training_runs/improved_v3_split_e12_v1"
mkdir "$ROOT"
cat > "$ROOT/input_sha256.txt" <<'HASHES'
98c0be498eed8d55c6b8b9839f8f7061c7525b11dcfec883f8fbcb7533d158bf  recognition/hipmri_3d_improved_unet_hard/run_improved_v3_split.py
877ea98efd8176ed4943dac18d24020929334ba7d0807c6f2b601b0ae58fcbf0  recognition/hipmri_3d_improved_unet_hard/test_improved_v3_split.py
50b2348657af18bd149ae4fd7c6a1dab68d06c9f99fea6cff0d55ecc527cde7e  recognition/hipmri_3d_improved_unet_hard/submit_v3_split.sh
60c01eb058a73d270174717699234b7e9e83b2eac2006401d52c525ca00e0ad1  recognition/hipmri_3d_improved_unet_hard/train_improved_v3.py
806c1ce2a7e2e7019ac39c19008e9c221a47ac792da89459bc173ed87fd55315  recognition/hipmri_3d_improved_unet_hard/improved_v3_config.json
e860138e6f411c40c3cb843ff97cf0fc9d2a942025741ad37789bcc4f28702c8  recognition/hipmri_3d_improved_unet_hard/test_improved_v3.py
HASHES
V2="$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1"
printf '%s  %s\n' \
 '3229bdb38afa7b83af04e79cd7f882059a3510963eff29957554a5438d490e9f' "$V2/improved_unet_v2_last.pt" \
 'd77e0c53c13a53b8a0af0dd22cf45f6f64798c82a077f597ba9aaa29a80a215e' "$V2/validation_652948.jsonl" >> "$ROOT/input_sha256.txt"
shopt -s nullglob
DIAG=("$HOME/hipmri_training_runs/improved_v3_v2_continuation_v1/diagnostic"/events_654393_*.jsonl)
test "${#DIAG[@]}" -eq 1
printf '%s  %s\n' 'c484f87ddac5195789887dffa1b01206f005d500537c5ba70c2a82bb342ce472' "${DIAG[0]}" >> "$ROOT/input_sha256.txt"
sha256sum --check "$ROOT/input_sha256.txt"
)
```

## 4. Remote CPU preflight and focused regression (NOT EXECUTED)

Environment: Rangpur login Bash, existing torch environment. Purpose: verify new
code with real dependencies and authenticate the original protected inputs.
Expected: eight tests OK and protected preflight PASS. Impact: checkpoint/filename
reads only; no MRI arrays or CUDA. The already completed 126-test suite is not repeated.

```bash
(
set -euo pipefail
cd "$HOME/comp3710_hipmri_verify"
PY="$HOME/miniconda3/envs/torch/bin/python"
ROOT="$HOME/hipmri_training_runs/improved_v3_split_e12_v1"
V2="$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m unittest \
 recognition.hipmri_3d_improved_unet_hard.test_improved_v3_split -v \
 2>&1 | tee "$ROOT/cpu_tests.txt"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m \
 recognition.hipmri_3d_improved_unet_hard.run_improved_v3_split \
 --stage A --root "$ROOT" --parent-checkpoint "$V2/improved_unet_v2_last.pt" \
 --v2-validation-log "$V2/validation_652948.jsonl" --preflight-only \
 | tee "$ROOT/preflight.txt"
)
```

## 5. Submission gates — ONLY after separate approval

Environment: Rangpur login Bash. Purpose: submit one 15-minute stage. Expected:
SUBMITTED_STAGE_A_JOB (then B/C respectively). Impact: one real GPU allocation per
invocation, 45 minutes total requested. Each guard requires original job 654465
CANCELLED and no active user jobs; B/C require the previous scheduler COMPLETED/0:0
plus successful numerical/journal/checkpoint audit. No automatic dependency chain.
Existing claim, output, lock or ambiguous submission blocks retries.

Run A only after approval, cancellation verification and CPU PASS:

```bash
bash "$HOME/comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard/submit_v3_split.sh" A
```

Run B only after A acceptance and review:

```bash
bash "$HOME/comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard/submit_v3_split.sh" B
```

Run C only after B acceptance and review:

```bash
bash "$HOME/comp3710_hipmri_verify/recognition/hipmri_3d_improved_unet_hard/submit_v3_split.sh" C
```

## 6. Monitor and accept each stage

Environment: Rangpur login Bash. Purpose: scheduler/log inspection. Expected:
COMPLETED/0:0 within 15 minutes. Impact: read-only. Set STAGE to A, B or C as appropriate.

```bash
STAGE=A
ROOT="$HOME/hipmri_training_runs/improved_v3_split_e12_v1"
JOB=$(cat "$ROOT/job_${STAGE}.txt")
squeue -j "$JOB"
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed,Timelimit,NodeList
tail -n 20 "$ROOT/${STAGE}_${JOB}.out"
cat "$ROOT/${STAGE}_${JOB}.err"
find "$ROOT" -maxdepth 2 -type f -printf '%p %s bytes\n'
```

Environment: Rangpur login Bash. Purpose: authenticate completed-stage numerical
evidence independently of exit code. Expected: JSON status PASS. Impact: CPU-only
checkpoint/log reads, with an external acceptance report. After C it recomputes all
38 confusion matrices, pooled/macro metrics and frozen selection, verifies lineage
and both checkpoint hashes. It also rechecks unchanged Stage 1/V2 inputs.

```bash
(
set -euo pipefail
cd "$HOME/comp3710_hipmri_verify"
STAGE=A  # change to B or C for the corresponding completed stage
ROOT="$HOME/hipmri_training_runs/improved_v3_split_e12_v1"
V2="$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1"
PY="$HOME/miniconda3/envs/torch/bin/python"
JOB=$(cat "$ROOT/job_${STAGE}.txt")
STATE=$(sacct -n -P -j "$JOB" --format=JobID,State,ExitCode | awk -F'|' -v id="$JOB" '$1==id {print $2"|"$3}')
test "$STATE" = 'COMPLETED|0:0'
sha256sum --check "$ROOT/input_sha256.txt"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m \
 recognition.hipmri_3d_improved_unet_hard.run_improved_v3_split \
 --stage "$STAGE" --root "$ROOT" --parent-checkpoint "$V2/improved_unet_v2_last.pt" \
 --v2-validation-log "$V2/validation_652948.jsonl" --audit-only \
 | tee "$ROOT/acceptance_${STAGE}.json"
)
```

## Failure and interpretation

Timeout, failure or interrupted journal means STOP; no remaining stage automatically
runs and no retry is authorised. Preserve all evidence and stale locks. Do not delete
claims or change hashes to bypass a failed gate. A complete immutable checkpoint
without a final PASS journal is not automatically accepted by this protocol. Recovery
requires review and a separately approved budget. Original scripts and V2 are untouched.

The 15/15/15 allocation split is a provisional feasibility choice, not a guarantee.
Do not replace it with 19/19/19: that exceeds 45 minutes. No measured queue speedup or
segmentation improvement exists yet. GPU memory for separate Validation is not directly
comparable to integrated Validation: restored optimiser state remains, but training
gradient buffers need not be present. Scientific comparison uses identical weights,
full-volume inference and metrics, not identical process-memory layout.
