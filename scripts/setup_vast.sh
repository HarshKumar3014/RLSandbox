#!/usr/bin/env bash
# One-time setup on the vast.ai box. Run from the project root (e.g. /workspace/markov).
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
pip install -q uv 2>/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
mkdir -p logs
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0)); print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
python -m pytest -q markov_rl/tests
# Data is shipped in data/; regenerate only if missing (deterministic, same seeds).
[ -f data/test.jsonl ] || { python -m markov_rl.tokstats && python -m markov_rl.data; }
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo SETUP_OK
