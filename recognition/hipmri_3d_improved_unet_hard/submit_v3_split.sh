#!/bin/bash
# Preparation only until the student explicitly approves and invokes A/B/C.
# One 15-minute submission per stage. Never cancels, retries or auto-chains jobs.
set -euo pipefail
umask 077
cd "$HOME/comp3710_hipmri_verify"
ROOT="$HOME/hipmri_training_runs/improved_v3_split_e12_v1"
V2="$HOME/hipmri_training_runs/improved_v2_wce_sqrt_v1"
PY="$HOME/miniconda3/envs/torch/bin/python"
MODULE=recognition.hipmri_3d_improved_unet_hard.run_improved_v3_split
SCRIPT="$PWD/recognition/hipmri_3d_improved_unet_hard/submit_v3_split.sh"
STAGE="${1:?Specify A, B or C}"
case "$STAGE" in A|B|C) ;; *) echo 'Invalid stage' >&2; exit 2;; esac
test -d "$ROOT"
sha256sum --check "$ROOT/input_sha256.txt"
ARGS=(--stage "$STAGE" --root "$ROOT" --parent-checkpoint "$V2/improved_unet_v2_last.pt"
      --v2-validation-log "$V2/validation_652948.jsonl")

if [ "${2:-}" = '--worker' ]; then
    : "${SLURM_JOB_ID:?Slurm allocation required}"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    exec "$PY" -u -B -m "$MODULE" "${ARGS[@]}"
fi
test "$#" -eq 1
grep -qx 'OK' "$ROOT/cpu_tests.txt"
CUDA_VISIBLE_DEVICES='' "$PY" -B -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "PASS"' "$ROOT/preflight.txt"

# The pending original must be explicitly cancelled by the student first.
# COMPLETED original work must instead be reviewed, not duplicated here.
OLD=$(sacct -n -P -j 654465 --format=JobID,State | awk -F'|' '$1=="654465" {print $2}')
[[ "$OLD" == CANCELLED* ]] || { echo 'STOP: Job 654465 must be confirmed CANCELLED first.' >&2; exit 1; }
ACTIVE=$(squeue -h -u "$USER" -o '%i|%j|%T')
test -z "$ACTIVE" || { printf 'STOP: active jobs exist:\n%s\n' "$ACTIVE" >&2; exit 1; }
test ! -e "$ROOT/active.lock"
test ! -e "$ROOT/$STAGE"
test ! -L "$ROOT/$STAGE"
test ! -e "$ROOT/submission_${STAGE}.claim"

if [ "$STAGE" != A ]; then
    PREVIOUS=A
    [ "$STAGE" != C ] || PREVIOUS=B
    JOB=$(cat "$ROOT/job_${PREVIOUS}.txt")
    STATE=$(sacct -n -P -j "$JOB" --format=JobID,State,ExitCode |
        awk -F'|' -v id="$JOB" '$1==id {print $2"|"$3}')
    test "$STATE" = 'COMPLETED|0:0'
    CUDA_VISIBLE_DEVICES='' "$PY" -B -m "$MODULE" \
        --stage "$PREVIOUS" --root "$ROOT" \
        --parent-checkpoint "$V2/improved_unet_v2_last.pt" \
        --v2-validation-log "$V2/validation_652948.jsonl" --audit-only
fi

CUDA_VISIBLE_DEVICES='' "$PY" -B -m "$MODULE" "${ARGS[@]}" --preflight-only
sbatch --test-only --partition=a100-test --gres=gpu:a100:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:15:00 \
    --job-name="hipmri-v3-split-$STAGE" --wrap='true'

# Claim before real submission. An ambiguous failure requires human review.
(set -o noclobber; echo 'One submission attempt; no automatic retry' > "$ROOT/submission_${STAGE}.claim")
sbatch --parsable --partition=a100-test --gres=gpu:a100:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --time=00:15:00 \
    --job-name="hipmri-v3-split-$STAGE" \
    --output="$ROOT/${STAGE}_%j.out" --error="$ROOT/${STAGE}_%j.err" \
    "$SCRIPT" "$STAGE" --worker \
    > "$ROOT/submission_${STAGE}.stdout" 2> "$ROOT/submission_${STAGE}.stderr"
RESPONSE=$(cat "$ROOT/submission_${STAGE}.stdout")
JOB="${RESPONSE%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Ambiguous submission: inspect Slurm; do not retry' >&2; exit 1; }
printf '%s\n' "$JOB" > "$ROOT/job_${STAGE}.txt"
printf 'SUBMITTED_STAGE_%s_JOB=%s\n' "$STAGE" "$JOB"
