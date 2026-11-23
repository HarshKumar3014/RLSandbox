#!/usr/bin/env bash
# ~3-5 min. Catches vLLM / LoRA-reload / training issues before the long jobs. Needs checkpoints/warmup_mem.
set -euo pipefail
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/workspace/hf}
python -m markov_rl.grpo --config markov_rl/configs/rl_learned_m100_s0.yaml \
  --set name=smoke_gpu steps=3 programs_per_step=4 group_size=4 fixed_N=8 train_on_zero_adv=true save_every=3 \
  2>&1 | tee logs/smoke_gpu.log
echo SMOKE_OK
