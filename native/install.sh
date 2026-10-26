#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if [[ $EUID -eq 0 ]]; then
  echo 'Run install_system.sh as root, then run this script as your normal WSL user.' >&2
  exit 1
fi
command -v uv >/dev/null
nvidia-smi
mkdir -p native/logs
python3 native/fetch_cuda.py 2>&1 | tee native/logs/cuda-download.log
bash native/install_python.sh 2>&1 | tee native/logs/python-install.log
bash native/build_opencv.sh 2>&1 | tee native/logs/opencv-build.log
bash native/build_extensions.sh 2>&1 | tee native/logs/extensions-build.log
source native/env.sh
python native/check_install.py 2>&1 | tee native/logs/check-install.log
uv pip check --python "$VIRTUAL_ENV/bin/python"
