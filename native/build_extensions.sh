#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BUNDLESDF_ROOT"
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-build-isolation --no-deps -e mycuda
if ! python -c 'from pytorch3d.transforms import so3_log_map' >/dev/null 2>&1; then
  uv pip install --python "$VIRTUAL_ENV/bin/python" --no-build-isolation --no-deps 'git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'
fi
bash native/configure_bundletrack.sh
cmake --build BundleTrack/build --parallel "$MAX_JOBS"
mkdir -p .native/bin
g++-11 native/check_opencv.cpp -o .native/bin/check_opencv \
  -I.native/opencv/include/opencv4 -L.native/opencv/lib \
  -lopencv_cudafeatures2d -lopencv_features2d -lopencv_core
.native/bin/check_opencv
