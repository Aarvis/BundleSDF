#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$BUNDLESDF_ROOT/.native/src" "$BUNDLESDF_ROOT/native/logs"
cd "$BUNDLESDF_ROOT/.native/src"
for task_project in opencv opencv_contrib; do
  if [[ ! -d "$task_project-4.11.0" ]]; then
    curl --fail --location --retry 3 "https://github.com/opencv/$task_project/archive/refs/tags/4.11.0.tar.gz" -o "$task_project-4.11.0.tar.gz"
    tar -xzf "$task_project-4.11.0.tar.gz"
  fi
done
cmake -S opencv-4.11.0 -B opencv-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$BUNDLESDF_ROOT/.native/opencv" \
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 \
  -DOPENCV_EXTRA_MODULES_PATH="$PWD/opencv_contrib-4.11.0/modules" \
  -DBUILD_LIST=core,imgproc,imgcodecs,highgui,calib3d,features2d,flann,rgbd,cudev,cudaarithm,cudafeatures2d \
  -DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" -DCUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCUDA_ARCH_BIN="$TORCH_CUDA_ARCH_LIST" -DCUDA_ARCH_PTX= \
  -DWITH_CUBLAS=OFF -DWITH_CUFFT=OFF -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF \
  -DWITH_CUDNN=OFF -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_OPENCL=OFF -DWITH_VTK=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_IPP=OFF -DWITH_OPENEXR=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF
cmake --build opencv-build --parallel "$MAX_JOBS"
cmake --install opencv-build
