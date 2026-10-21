#!/usr/bin/env bash
set -euo pipefail
root="$HOME/20261018_g1_stage/20261017"
cd "$root/code"
exec "$HOME/miniconda3/envs/torch/bin/python" -B tools/submit_v5_screen.py "${1:-new}"
