#!/usr/bin/env bash
# Milestone item 2: timing pilot (20 GRPO steps each, N=32 fixed). ~20-45 min per mode.
# Usage: bash scripts/run_pilot.sh [RATE_USD_PER_HOUR] [modes...]   e.g. bash scripts/run_pilot.sh 1.20 learned full tail
set -euo pipefail
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/workspace/hf}
RATE=${1:-1.20}; shift || true
MODES=${@:-learned full}
python -m markov_rl.pilot_timing --modes $MODES 2>&1 | tee -a logs/pilot.log
python -m markov_rl.run_table --rate "$RATE"
python -m markov_rl.report
echo PILOT_DONE
