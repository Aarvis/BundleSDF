#!/usr/bin/env bash
set -euo pipefail
if [[ $EUID -ne 0 ]]; then
  exec sudo bash "$0" "$@"
fi
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential gcc-11 g++-11 cmake ninja-build pkg-config curl git xz-utils \
  libpcl-dev libopencv-dev libboost-system-dev libboost-program-options-dev \
  libboost-serialization-dev libyaml-cpp-dev libblas-dev liblapack-dev \
  libgl1-mesa-dev freeglut3-dev libglew-dev pybind11-dev libzmq3-dev cppzmq-dev \
  libeigen3-dev python3-dev
