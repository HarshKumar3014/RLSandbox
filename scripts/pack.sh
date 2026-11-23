#!/usr/bin/env bash
# Local: bundle code + data for upload (no venv, no checkpoints).
set -euo pipefail
cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar czf /tmp/markov.tgz --exclude='.venv' --exclude='__pycache__' --exclude='checkpoints' \
  --exclude='.pytest_cache' --exclude='*.pdf' markov_rl scripts data results requirements.txt requirements-cpu.txt
ls -lh /tmp/markov.tgz
