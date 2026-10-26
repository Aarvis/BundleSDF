# BundleSDF: native WSL2 installation with uv

This is a separate native installation of NVlabs/BundleSDF, based on upstream
commit `ffa67d4`. It does not install Docker or change WMAM's Python environment.
Use the patched checkout containing `native/` and the accompanying source fixes;
an untouched upstream clone does not contain these installation scripts.

## Target environment

- Ubuntu 24.04 x86-64 under WSL2; an NVIDIA Windows driver with WSL CUDA support.
- Initial machine: RTX 4000 Ada Laptop, 12 GB VRAM; Windows driver 596.71.
- Python 3.10.22 managed by uv; PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124.
- Kaolin 0.18.0 wheel built for **that exact PyTorch/CUDA combination**.
- CUDA 12.4.1 redistributable build components, GCC/G++ 11, CMake 3.28.
- OpenCV 4.11.0 with CUDA matching, Ubuntu PCL 1.14, PyTorch3D v0.7.9
  (commit `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`).

These versions passed the installation and bounded sample checks below.
Other Linux releases/GPU architectures need separate
validation. CUDA 12.4 does not support every newer GPU architecture.

## 1. Locate the patched checkout

Run the following commands **in Ubuntu bash**, not PowerShell. This machine has
a nested clone at `/home/aarvis/WMAM/BundleSDF/BundleSDF`. Other machines can use
any Linux filesystem location. Set your own path:

```bash
cd ~/WMAM/BundleSDF/BundleSDF
git rev-parse --short HEAD
nvidia-smi
uv --version
```

For another machine, use the small `native-wsl-installation.zip` accompanying
this document. It contains scripts, dependency pins, checksums, this README, and
the source patch, but no environments, CUDA downloads, model weights, or sample
data. In a new Linux directory:

```bash
git clone https://github.com/NVlabs/BundleSDF.git BundleSDF
cd BundleSDF
git checkout ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1
# Copy the installation ZIP here, then extract it with system Python:
python3 -m zipfile -e /path/to/native-wsl-installation.zip .
git apply --check native/compatibility.patch
git apply native/compatibility.patch
```

Do not apply the patch again to this machine's already patched checkout.

Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/)
if it is unavailable. Docker, Conda, and a Linux NVIDIA display driver are not
required. WSL uses the Windows NVIDIA driver.

## 2. Install Ubuntu development libraries

```bash
bash native/install_system.sh
```

This step needs root and uses sudo. If sudo requires a password, enter it in your
own terminal. Alternatively, from PowerShell, replace the path below as needed:

```powershell
wsl -d Ubuntu -u root -- bash /home/aarvis/WMAM/BundleSDF/BundleSDF/native/install_system.sh
```

This installs development libraries through apt. PCL pulls in substantial VTK,
Boost, and other dependencies even with `--no-install-recommends`. It does not
run apt autoremove or replace the Windows GPU driver.

## 3. Install and compile as your normal Linux user

```bash
cd ~/WMAM/BundleSDF/BundleSDF
bash native/install.sh
```

The installer downloads NVIDIA redistributable components to `.native/`, checks
their SHA-256 values against NVIDIA's 12.4.1 manifest, and assembles a local CUDA
build toolkit. It leaves `/usr/local/cuda` and system compiler alternatives alone.
Python packages are installed into this checkout's `.venv` using uv.

OpenCV is built locally with only the modules needed by BundleTrack, including
CUDA descriptor matching. Its Python binding is disabled: Python uses its own
pinned opencv-python wheel, while BundleTrack links against the local C++ build.
No GUI backend is built for C++ highgui; validation runs without GUI windows.

Builds default to four jobs. On machines with less RAM:

```bash
MAX_JOBS=2 bash native/install.sh
```

`native/env.sh` detects the active NVIDIA GPU's compute capability. The initial
Ada machine builds for `8.9` / CMake `89`. For cross-compilation or multiple GPUs,
explicitly configure both architecture variables and validate the result.

Build logs are under `native/logs/`. Failed builds can be resumed by repeating
the relevant script; avoid deleting the entire checkout or environment.

## 4. Activate and verify

```bash
source native/env.sh
python native/check_install.py
uv pip check --python .venv/bin/python
python run_custom.py --help
```

Source `native/env.sh` in each new bash shell. It sets the local CUDA, compiler,
library and Python paths only for that shell. Do not run upstream `build.sh`:
it hardcodes Docker's Python paths and builds a broad set of GPU architectures.
The script also excludes Windows drive entries from this shell's PATH: CMake's
PCL/VTK discovery otherwise spends minutes probing Windows directories over WSL's
filesystem bridge. It does not change Windows PATH or WSL configuration.

The checker exercises GPU tensor operations, Kaolin octree construction,
PyTorch3D transformations, the grid encoder's forward/backward CUDA kernels,
and OpenCV CUDA descriptor matching, and imports BundleTrack and BundleSDF.
It writes `native/logs/installation-check.json`. Passing this checker is not an
end-to-end reconstruction result.

## 5. Official milk sample

The sample and LoFTR weights are external downloads referenced by the upstream
[README](https://github.com/NVlabs/BundleSDF#run-on-your-custom-data).
Online XMem segmentation is unnecessary when the sample's masks are supplied.

```bash
source native/env.sh
python native/fetch_sample.py
python native/smoke_sample.py --output .native/runs/my-milk-check \
  --frames 120 --stride 1 --train-steps 50
```

Choose a new output directory each time; the helper refuses to overwrite existing
results. Downloads are verified against `native/sample-assets.json`. The sample
ZIP is about 1.28 GB (decimal), plus about 1.32 GB extracted. LoFTR weights are
about 46 MB. The dataset contains RGB, metric depth, masks, and intrinsics.

This bounded helper uses the real tracker and neural reconstruction, with 50
training steps per update, three startup keyframes, a two-degree keyframe
rotation threshold, supplied masks, and no final global refinement. It exports
`online_mesh.obj` and `summary.json` in the requested output directory. It is an
installation test, not a replication of the paper's accuracy benchmark.

Keep `--stride 1` for this test: skipping frames can exceed the tracker's allowed
inter-frame motion and cause failures unrelated to installation.

For the full upstream pipeline, use a **new** output path:

```bash
python run_custom.py --mode run_video \
  --video_dir .native/data/milk/2022-11-18-15-10-24_milk \
  --out_folder .native/runs/milk-full --use_segmenter 0 --use_gui 0 --debug_level 2
```

The upstream script deletes/recreates its output directory and runs global
refinement after tracking. The full sequence/refinement was not run in this
installation validation and takes more time and storage.

## Source compatibility changes

- Use standard shared pointers for PCL 1.14 cloud types instead of legacy Boost pointers.
- Respect the GPU architecture passed to CMake and link Boost filesystem explicitly.
- Remove unused CUDA optical-flow/image-processing and cuSolver includes, while
  retaining the actual CUDA matching and RANSAC implementations.
- Build against the Python interpreter and OpenCV selected by the local scripts.
- Include PCL's geometry declarations explicitly.
- Remove an unconditional upstream debugger breakpoint from reference-frame
  recovery so unattended runs can continue through that branch.

The complete source patch is saved as `native/compatibility.patch`.

## Troubleshooting

- **CUDA/compiler mismatch:** source `native/env.sh`; check `which nvcc`,
  `nvcc --version`, and `echo "$CUDA_HOME"`. They should select this checkout's
  CUDA 12.4 compiler. Do not change `/usr/local/cuda` or use an unsupported-compiler
  override. `nvidia-smi` reports driver capability, not the selected build toolkit.
- **Missing my_cpp/common/gridencoder:** source the environment and rerun
  `bash native/build_extensions.sh`. Do not reuse binaries from a different
  Python, PyTorch version, GPU architecture, or checkout path.
- **Slow configuration:** ensure the shell was initialized through `native/env.sh`.
  It excludes inherited Windows tool directories during discovery.
- **Compilation uses too much RAM:** lower `MAX_JOBS`; the provided scripts do
  not change system-wide resource settings.
- **Blank GUI/highgui errors:** validation is headless; C++ highgui has no window
  backend. The optional DearPyGui interface and full rendering paths were not
  validated here.
- **Download failure:** Google Drive downloads depend on file permissions/quota.
  Do not substitute an unverified checkpoint; use the upstream-linked assets and
  compare against the provided SHA-256 checksums.
- **AMP deprecation warnings:** upstream uses older PyTorch AMP calls. These
  emitted warnings but passed the tested inference/reconstruction path.

## Storage and isolation

`.native/` holds CUDA build tools, downloads, OpenCV sources/build outputs, and
sample/output files. `.venv/` holds Python packages; uv may share downloaded
packages through its user cache. Ubuntu libraries occupy WSL system storage.
Measured checkout footprint after validation: about **7.9 GiB in `.venv`** and
**4.6 GiB in `.native`**, excluding apt libraries and uv's shared cache. This is
the observed local footprint, not a guaranteed installation budget.
This is smaller in scope than the supplied DeepStream/Triton Docker base, but
CUDA packages, native compilation, and sample data still require multiple GB.
Check actual free space on the Windows drive hosting WSL, not only `df -h /`:
WSL's virtual disk capacity can exceed the physical space available on Windows.

The installation itself does not command the robot. The separate, optional WMAM
single-target tracking integration is documented in
`../../docs/notes/wmam-target-pose-tracking.md`. It uses this native environment in
a persistent worker, without reconstruction. When transferring that integration,
also preserve `native/wmam_worker.sh`, `native/wmam_pose_worker.py`, its changes
to `native/tracking_only.py` and `loftr_wrapper.py`, and the `YamlNode.set_float`
binding in `BundleTrack/pybind_interface/pybind_api.cpp`; rebuild `my_cpp` after
applying the binding change.

For the separate milk pose-tracking-only benchmark and its input/output videos,
see [POSE_ONLY_EXPERIMENT.md](POSE_ONLY_EXPERIMENT.md).

## Validation record

Validated on 2026-10-21 on the machine specified above:

- Native CUDA extensions, PyTorch3D, local CUDA OpenCV, and BundleTrack compiled.
- `native/check_install.py` passed, including CUDA grid encoding and gradients,
  Kaolin octree construction, PyTorch3D transforms, and C++ OpenCV CUDA matching.
- `uv pip check` passed with 170 installed packages.
- Replaying the documented `bash native/install.sh` completed successfully.
- Official milk sample: **120 consecutive frames**, **zero FAIL statuses**,
  **18 keyframes**, **8,673 mesh vertices / 17,310 faces**, approximately
  **51.6 seconds** total. Mesh vertices were checked for finite coordinates.
- Result: `.native/runs/milk-smoke-03/online_mesh.obj`; detailed per-frame poses,
  statuses and timings: `.native/runs/milk-smoke-03/summary.json`.
- Compact evidence: `native/validation.json`; detailed logs: `native/logs/`.

These results validate a bounded single-object run. They do not establish pose
accuracy against ground truth, complete unseen-surface reconstruction, full-video
global-refinement quality, simultaneous WMAM/GraspGenX operation, or multi-object
performance. The pose outputs use BundleSDF's object frame; integration must
explicitly account for object-to-camera and camera-to-world transforms.
