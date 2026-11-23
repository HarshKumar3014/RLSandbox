#!/usr/bin/env bash
# Everything after setup, in order. Usage: bash scripts/run_all_milestone.sh <RATE_USD_PER_HOUR>
set -euo pipefail
RATE=${1:-1.20}
[ -f checkpoints/warmup_mem/adapter_model.safetensors ] || bash scripts/run_sft.sh
bash scripts/smoke_gpu.sh
bash scripts/run_eval.sh
bash scripts/run_pilot.sh "$RATE" learned full
source .venv/bin/activate && python -m markov_rl.report
echo ALL_DONE
