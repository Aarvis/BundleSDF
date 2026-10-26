#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
mkdir -p native/logs
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.10.22 .venv
fi
uv pip install --python .venv/bin/python torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
task_requirements=native/requirements.in
if [[ -f native/requirements.lock ]]; then task_requirements=native/requirements.lock; fi
uv pip install --python .venv/bin/python -r "$task_requirements"
uv pip install --python .venv/bin/python 'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124/kaolin-0.18.0-cp310-cp310-linux_x86_64.whl'
