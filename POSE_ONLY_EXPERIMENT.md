# Milk sample: tracking only, with latency and videos

This experiment is isolated inside this BundleSDF checkout. It does not modify
WMAM, start a robot controller, or integrate with the WMAM interface.

## Run

Use the native environment described in `NATIVE_WSL_INSTALL.md`:

```bash
cd ~/WMAM/BundleSDF/BundleSDF
source native/env.sh
python native/benchmark_pose_only.py \
  --output .native/runs/my-pose-only-run --frames 120
```

Use a new output directory for each run. The script refuses to overwrite results.
The dataset defaults to `.native/data/milk/2022-11-18-15-10-24_milk`; use `--video`
to override it. Missing sample files can be fetched with `python native/fetch_sample.py`.

The default warm-up is three LoFTR image pairs without adding any tracker frames
or changing the pose state. Its time is reported separately. Use
`--warmup-pairs 0` to include cold CNN execution in the per-frame measurements;
do not compare different warm-up settings as though they were identical trials.

## What runs

`native/tracking_only.py` reuses the installed BundleSDF implementation's frame
construction, RGB feature matching, depth correspondence filtering, GPU RANSAC,
rigid alignment, local GPU pose optimization, and keyframe selection.

Its constructor does **not** invoke the normal BundleSDF constructor, and its
tracking call does **not** invoke the normal reconstruction-aware `run` method.
There is no multiprocessing manager, neural reconstruction worker, SDF training,
mesh extraction, reconstruction feedback, or waiting for that worker. Both
reconstruction-dependent optimization weights are explicitly zero. The normal
BundleSDF tracking-plus-reconstruction demo remains unchanged by this experiment.
Importing the upstream Python module still imports its dependencies, including
Kaolin; imports alone do not start reconstruction.

The tracker retains partial point clouds in reference/keyframes for tracking.
It does not fuse them into a reconstructed object model or mesh.

Inputs for each consecutive frame:

- Recorded RGB, depth PNG converted from millimetres to metres, supplied object mask.
- Recorded intrinsics scaled consistently to the 640 x 480 tracking resolution.
- No CAD model, annotated object poses, or precomputed object mesh.

The first object-frame origin is the midpoint of filtered visible-surface bounds.
Its axes initially align with the camera: X right, Y down, Z forward. This is a
reference-frame convention, not a canonical physical orientation or centre of mass.
Later frames estimate motion relative to this frame. Returned matrices are
**object-to-camera**, in metres: `p_camera = R @ p_object + t`.
They are the inverse of the upstream internal `_pose_in_model` representation.

## Outputs

| File | Contents |
| --- | --- |
| `input_rgb.mp4` | Clean RGB frames, resized to the tracking resolution |
| `output_pose.mp4` | Same frames with pose axes, a reference box, frame index, latency and status |
| `poses.csv` | One row per frame: timing, status, timestamps and 4 x 4 object-to-camera matrix |
| `poses.json` | Equivalent structured per-frame evidence |
| `benchmark.json` | Aggregate latency, startup, throughput, settings and environment |
| `tracking_config.yml` | Exact tracking configuration for this run |
| `preview_input.png`, `preview_output.png` | First input and final overlay |

Axes are red X, green Y, blue Z. The yellow box is the first observed cloud's
axis-aligned bounds transformed by the tracked pose. It is not a fitted mesh,
does not grow per frame, and does not claim to describe hidden surfaces exactly.
If tracking fails, the frame is labelled failed and no current pose is drawn;
the CSV leaves its pose fields empty instead of silently reusing an old pose.

Both videos use approximately 30.03 FPS, estimated from the median interval of
the dataset's nanosecond frame IDs. The original intervals vary slightly and
are preserved in CSV/JSON. **Playback speed is not inference throughput.** A
four-second output video may require much more than four seconds to process.

## Timing definitions

- `imports_ms`: Python dependency imports.
- `model_initialization_ms`: tracker/LoFTR construction and checkpoint loading,
  with CUDA synchronization at completion.
- `warmup_ms`: separately reported image-pair warm-up.
- `load_ms`: disk input decoding and image/depth/mask resizing.
- `frame_preparation_ms`: mask erosion, depth filtering, GPU cloud/normal creation.
- `matching_and_pose_ms`: matching, outlier rejection, alignment, local pose
  optimization and keyframe bookkeeping. This is not just the optimization solver.
- `pose_readout_ms`: obtaining the current transform and converting to
  object-to-camera coordinates.
- `tracking_ms`: full pose-producing call, including preparation and readout.
  CUDA is synchronized at the start and after GPU stages, so queued GPU work
  cannot escape the measurement. Synchronization overhead is included.
- `pose_available_ms`: `load_ms + tracking_ms`.
- `overlay_ms` and `video_submit_ms`: drawing and submitting frames to encoders,
  excluded from tracking. Final encoder drain is reported separately.

`subsequent_frames_tracking` excludes only frame 1 (object-frame initialization);
warm-up is separately controlled by the command. Total wall time includes input
loading and both video encoders. Video encoding can still contend for CPU resources;
these are measurements of this concrete experiment, not a universal tracker limit.
The environment fixes OMP and Python Torch/OpenCV thread counts to one.

## Measured run

Run: `.native/runs/milk-pose-only-01`, 2026-10-21, RTX 4000 Ada Laptop GPU,
Python 3.10.22, PyTorch 2.6.0+cu124; first 120 consecutive frames, three warm-up pairs.

| Metric | Result |
| --- | ---: |
| Frames without a reported tracking failure | 120 / 120 |
| Reference/keyframes | 17 |
| First pose after model warm-up | 179.0 ms |
| Subsequent-frame mean tracking latency | 183.0 ms |
| Subsequent-frame median tracking latency | 187.8 ms |
| Subsequent-frame P95 tracking latency | 312.6 ms |
| Subsequent-frame mean including input loading | 193.4 ms |
| Tracking-only throughput across all frames | 5.47 FPS |
| Wall time including both videos, excluding startup | 24.42 s |
| Wall throughput including both videos | 4.91 FPS |
| Dependency imports | 4.01 s |
| Tracker/model initialization | 0.365 s |
| Three-pair warm-up | 0.625 s |

Average stages after frame 1: preparation **11.8 ms**, matching plus pose
optimization **171.1 ms**, pose readout **0.073 ms**. Latency grows as the clip
adds reference frames, so reporting only early-frame timing would be misleading.
No reconstruction-related multi-second spikes occurred in this run, but this
existing RGBD tracking stack still did not reach the source camera's 30 FPS.

Both MP4 files were decoded successfully: 120 frames each, 640 x 480, about
four seconds, same playback rate. All 120 returned transforms had finite entries,
orthonormal rotations and determinant approximately +1. No mesh files were created.
`artifact_checks.json` records these checks; `preview_grid.png` shows four sampled
output frames reviewed for axis/box placement and readable overlays.

Zero failure statuses and valid rigid matrices are not proof of pose accuracy.
No ground-truth pose evaluation was performed, and this is one short single-object
clip. The prior reconstruction benchmark had different warm-up/logging/thread
settings and reconstruction feedback, so it is not a controlled speedup trial.
