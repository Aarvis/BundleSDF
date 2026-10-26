#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$VIRTUAL_ENV/bin/python" -u "$BUNDLESDF_ROOT/native/wmam_pose_worker.py" --fd "$1" --run-dir "$2"
