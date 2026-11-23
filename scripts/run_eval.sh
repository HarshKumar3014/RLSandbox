#!/usr/bin/env bash
# Milestone item 1: warm-up eval at full context + untrained tail baseline. ~10-20 min.
set -euo pipefail
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/workspace/hf}
python -m markov_rl.eval --milestone --split test --Ns 8,16,32 --n_per_cell 200 2>&1 | tee logs/eval_milestone.log
echo EVAL_DONE
