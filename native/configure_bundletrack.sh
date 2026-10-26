#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BUNDLESDF_ROOT"
cmake --fresh -S BundleTrack -B BundleTrack/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -DCMAKE_CUDA_ARCHITECTURES="$BUNDLESDF_CUDA_ARCH" \
  -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" -DPYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DOpenCV_DIR="$BUNDLESDF_ROOT/.native/opencv/lib/cmake/opencv4"
