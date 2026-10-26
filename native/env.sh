# Source this file from bash. All changes apply only to the current shell.
BUNDLESDF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BUNDLESDF_ROOT
export VIRTUAL_ENV="$BUNDLESDF_ROOT/.venv"
export CUDA_HOME="$BUNDLESDF_ROOT/.native/cuda-12.4"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=g++-11
# CMake's package searches are extremely slow over WSL's Windows filesystem.
# Keep Linux tools and exclude Windows drive entries in this shell only.
task_linux_path=""
IFS=: read -r -a task_path_entries <<< "$PATH"
for task_path_entry in "${task_path_entries[@]}"; do
  case "$task_path_entry" in /mnt/[a-z]/*) continue ;; esac
  task_linux_path="${task_linux_path:+$task_linux_path:}$task_path_entry"
done
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$task_linux_path"
unset task_linux_path task_path_entries task_path_entry
export LD_LIBRARY_PATH="$BUNDLESDF_ROOT/.native/opencv/lib:$CUDA_HOME/lib64:$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$BUNDLESDF_ROOT:$BUNDLESDF_ROOT/BundleTrack/build${PYTHONPATH:+:$PYTHONPATH}"
export CMAKE_PREFIX_PATH="$BUNDLESDF_ROOT/.native/opencv${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OPENCV_IO_ENABLE_OPENEXR=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')}"
export BUNDLESDF_CUDA_ARCH="${BUNDLESDF_CUDA_ARCH:-${TORCH_CUDA_ARCH_LIST//./}}"
