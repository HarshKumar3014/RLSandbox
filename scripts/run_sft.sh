#!/usr/bin/env bash
# Both warm-up checkpoints, sequentially. Expected: ~5-15 min each on an A100.
set -euo pipefail
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p logs
python -m markov_rl.sft --config markov_rl/configs/sft_plain.yaml 2>&1 | tee logs/sft_plain.log
python -m markov_rl.sft --config markov_rl/configs/sft_mem.yaml 2>&1 | tee logs/sft_mem.log
echo SFT_DONE
