"""Benchmark milk RGBD pose tracking and save matched input/output videos."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

PROGRAM_STARTED = time.perf_counter()


def draw_overlay(image, pose, intrinsics, bounds, result, frame_number, total):
    import cv2
    import numpy as np
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 112), (20, 20, 20), -1)
    if pose is not None:
        def project(points):
            camera = points @ pose[:3, :3].T + pose[:3, 3]
            if np.any(camera[:, 2] <= 0.01):
                return None
            pixels = camera @ intrinsics.T
            return np.rint(pixels[:, :2] / pixels[:, 2:]).astype(int)
        axes = project(np.array([[0, 0, 0], [.06, 0, 0], [0, .06, 0], [0, 0, .06]]))
        if axes is not None:
            for end, color, label in zip(axes[1:], [(0, 0, 255), (0, 255, 0), (255, 100, 0)], 'XYZ'):
                cv2.arrowedLine(output, tuple(axes[0]), tuple(end), color, 3, cv2.LINE_AA, tipLength=.15)
                cv2.putText(output, label, tuple(end + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        if bounds is not None:
            import itertools
            corners = np.array(list(itertools.product(*zip(bounds[0], bounds[1]))))
            pixels = project(corners)
            if pixels is not None:
                for i in range(8):
                    for bit in (1, 2, 4):
                        j = i ^ bit
                        if j > i:
                            cv2.line(output, tuple(pixels[i]), tuple(pixels[j]), (0, 200, 255), 1, cv2.LINE_AA)
    status = 'TRACKED' if result['valid'] else 'FAILED - no current pose'
    lines = [f"POSE ONLY | {frame_number}/{total} | {status}",
             f"Pose latency {result['tracking_ms']:.1f} ms | {result['keyframes']} keyframes | no reconstruction",
             'XYZ axes: red / green / blue. Box: initial observed bounds.',
             'Playback follows source time; video speed is not inference throughput.']
    for y, text in zip((24, 49, 74, 99), lines):
        cv2.putText(output, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
    return output


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return {'count': len(a), 'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p90_ms': float(np.percentile(a, 90)), 'p95_ms': float(np.percentile(a, 95)),
            'min_ms': float(a.min()), 'max_ms': float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--video', type=Path, default=root / '.native/data/milk/2022-11-18-15-10-24_milk')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--warmup-pairs', type=int, default=3)
    args = parser.parse_args()
    if args.frames < 2 or args.warmup_pairs < 0:
        parser.error('frames >= 2 and warmup-pairs >= 0 required')
    if args.output.exists():
        parser.error('Choose a new output directory; existing results are preserved')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    tick = time.perf_counter()
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from tracking_only import PoseOnlyTracker
    from BundleTrack.scripts.data_reader import YcbineoatReader
    from Utils import yaml, set_seed
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    imports_ms = (time.perf_counter() - tick) * 1000
    set_seed(0)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    reader = YcbineoatReader(video_dir=str(args.video.resolve()), shorter_side=480)
    count = min(args.frames, len(reader.color_files))
    ids = reader.id_strs[:count]
    timestamps = np.array([int(x) for x in ids], dtype=np.int64)
    intervals = np.diff(timestamps).astype(float) / 1e9
    fps = 1 / float(np.median(intervals))
    if not (1 <= fps <= 240) or np.any(intervals <= 0):
        raise ValueError('Sample frame IDs must be increasing nanosecond timestamps')
    config = yaml.load((root / 'BundleTrack/config_ho3d.yml').read_text())
    config['SPDLOG'] = 0
    config['debug_dir'] = str(args.output / 'tracker') + '/'
    Path(config['debug_dir']).mkdir()
    config['depth_processing']['percentile'] = 95
    config['erode_mask'] = 3
    config['feature_corres'].update({'resize': 400, 'rematch_after_nerf': False,
        'max_dist_neighbor': .02, 'max_normal_neighbor': 30,
        'max_dist_no_neighbor': .01, 'max_normal_no_neighbor': 20, 'map_points': True})
    config['keyframe']['min_rot'] = 2
    config['ransac'].update({'inlier_dist': .01, 'inlier_normal_angle': 20,
        'max_trans_neighbor': .02, 'max_rot_deg_neighbor': 30,
        'max_trans_no_neighbor': .01, 'max_rot_no_neighbor': 10})
    config['p2p'].update({'max_dist': .02, 'max_normal_angle': 45})
    config['bundle']['w_sdf'] = 0
    config['bundle']['w_pm'] = 0
    configuration = args.output / 'tracking_config.yml'
    with configuration.open('w') as f:
        yaml.dump(config, f)
    tick = time.perf_counter()
    tracker = PoseOnlyTracker(configuration)
    torch.cuda.synchronize()
    model_initialization_ms = (time.perf_counter() - tick) * 1000
    tick = time.perf_counter()
    # Exercise CNN/CUDA without adding frames or changing the tracking state.
    warm = cv2.cvtColor(cv2.resize(cv2.imread(reader.color_files[0]), (400, 400)), cv2.COLOR_BGR2GRAY)
    warm = warm[None, :, :, None]
    for _ in range(args.warmup_pairs):
        tracker.loftr.predict(warm, warm)
    torch.cuda.synchronize()
    warmup_ms = (time.perf_counter() - tick) * 1000
    torch.cuda.reset_peak_memory_stats()

    rows, bounds = [], None
    first_rgb, last_overlay = None, None
    width, height = reader.W + reader.W % 2, reader.H + reader.H % 2
    writer_options = dict(fps=fps, codec='libx264', quality=8, macro_block_size=1,
                          ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    input_writer = imageio.get_writer(str(args.output / 'input_rgb.mp4'), **writer_options)
    output_writer = imageio.get_writer(str(args.output / 'output_pose.mp4'), **writer_options)
    benchmark_started = time.perf_counter()
    try:
        for i in range(count):
            tick = time.perf_counter()
            color = cv2.resize(cv2.imread(reader.color_files[i]), (reader.W, reader.H), interpolation=cv2.INTER_NEAREST)
            depth = reader.get_depth(i)
            mask = reader.get_mask(i)
            load_ms = (time.perf_counter() - tick) * 1000
            result = tracker.track(color, depth, reader.K.copy(), ids[i], mask)
            pose = result['object_to_camera']
            tick = time.perf_counter()
            if i == 0 and pose is not None:
                # Use the exact filtered initialization cloud, already centered
                # by upstream setNewInitCoordinate; never fit a new box per frame.
                cloud_path = Path(config['debug_dir']) / 'cloud_for_init_coord.ply'
                import open3d as o3d
                points = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
                inverse = np.linalg.inv(pose)
                local = points @ inverse[:3, :3].T + inverse[:3, 3]
                bounds = np.array([local.min(axis=0), local.max(axis=0)])
            overlay = draw_overlay(color, pose, reader.K, bounds, result, i + 1, count)
            overlay_ms = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            padded_input = cv2.copyMakeBorder(color, 0, height-reader.H, 0, width-reader.W, cv2.BORDER_REPLICATE)
            padded_output = cv2.copyMakeBorder(overlay, 0, height-reader.H, 0, width-reader.W, cv2.BORDER_REPLICATE)
            input_writer.append_data(cv2.cvtColor(padded_input, cv2.COLOR_BGR2RGB))
            output_writer.append_data(cv2.cvtColor(padded_output, cv2.COLOR_BGR2RGB))
            video_submit_ms = (time.perf_counter() - tick) * 1000
            row = {'frame_index': i, 'frame_id': ids[i], 'source_time_s': float((timestamps[i]-timestamps[0])/1e9),
                   **result, 'load_ms': load_ms, 'pose_available_ms': load_ms + result['tracking_ms'],
                   'overlay_ms': overlay_ms, 'video_submit_ms': video_submit_ms}
            row['object_to_camera'] = pose.tolist() if pose is not None else None
            rows.append(row)
            if first_rgb is None:
                first_rgb = color.copy()
            last_overlay = overlay
            print(f"FRAME {i+1}/{count} {result['status']} pose_ms={result['tracking_ms']:.3f} keyframes={result['keyframes']}", flush=True)
    finally:
        close_started = time.perf_counter()
        input_writer.close()
        output_writer.close()
        encoder_finalize_ms = (time.perf_counter() - close_started) * 1000
    benchmark_seconds = time.perf_counter() - benchmark_started
    cv2.imwrite(str(args.output / 'preview_input.png'), first_rgb)
    cv2.imwrite(str(args.output / 'preview_output.png'), last_overlay)
    scalar_fields = ['frame_index', 'frame_id', 'source_time_s', 'status', 'valid', 'keyframes',
                     'load_ms', 'frame_preparation_ms', 'matching_and_pose_ms', 'pose_readout_ms',
                     'tracking_ms', 'pose_available_ms', 'overlay_ms', 'video_submit_ms']
    matrix_fields = [f'T_object_to_camera_{a}{b}' for a in range(4) for b in range(4)]
    with (args.output / 'poses.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields + matrix_fields)
        writer.writeheader()
        for row in rows:
            data = {key: row[key] for key in scalar_fields}
            for a in range(4):
                for b in range(4):
                    data[matrix_fields[a*4+b]] = row['object_to_camera'][a][b] if row['valid'] else ''
            writer.writerow(data)
    (args.output / 'poses.json').write_text(json.dumps(rows, indent=2) + '\n')
    steady = rows[1:]
    report = {
        'mode': 'BundleSDF tracking only; no reconstruction process, SDF training, mesh extraction or reconstruction feedback',
        'dataset': str(args.video.resolve()), 'frames': len(rows),
        'successful_frames': sum(r['valid'] for r in rows), 'failed_frames': sum(not r['valid'] for r in rows),
        'source_fps': fps, 'video_fps': fps, 'source_frame_intervals_ms': stats(intervals*1000),
        'video_dimensions': [width, height], 'tracking_dimensions': [reader.W, reader.H],
        'imports_ms': imports_ms, 'model_initialization_ms': model_initialization_ms,
        'warmup_pairs': args.warmup_pairs, 'warmup_ms': warmup_ms,
        'first_pose_ms_after_model_warmup': rows[0]['tracking_ms'],
        'all_frames_tracking': stats([r['tracking_ms'] for r in rows]),
        'subsequent_frames_tracking': stats([r['tracking_ms'] for r in steady]),
        'subsequent_frames_pose_available_including_load': stats([r['pose_available_ms'] for r in steady]),
        'stage_statistics': {key: stats([r[key] for r in steady]) for key in
            ['load_ms', 'frame_preparation_ms', 'matching_and_pose_ms', 'pose_readout_ms', 'overlay_ms', 'video_submit_ms']},
        'tracking_only_throughput_fps': len(rows)/(sum(r['tracking_ms'] for r in rows)/1000),
        'benchmark_wall_seconds_including_videos': benchmark_seconds,
        'benchmark_wall_throughput_fps': len(rows)/benchmark_seconds,
        'encoder_finalize_ms': encoder_finalize_ms,
        'process_elapsed_seconds_before_report': time.perf_counter()-PROGRAM_STARTED,
        'keyframes': len(tracker.bundler._keyframes),
        'pose_convention': 'T_camera_from_object: p_camera = R @ p_object + t; metres',
        'initialization': 'Filtered visible-surface AABB midpoint; axes aligned with first camera. No PCA/CAD/GT pose.',
        'initial_observed_bounds_in_object_frame': bounds.tolist() if bounds is not None else None,
        'segmentation': 'Dataset-provided mask for each frame; no online segmentation model',
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'parent_peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(),
        'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'limitations': ['Single sequential clip; not an accuracy benchmark or latency distribution across trials',
            'Video constant FPS uses median source interval; original timestamps retained in CSV/JSON',
            'No reconstruction refinement; timing can differ with growing keyframe cache',
            'GPU-synchronized wall timings; video encoding remains separate but can contend for CPU',
            'Stage timings include synchronization overhead; subsequent frames exclude object initialization only'],
    }
    (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if report['failed_frames']:
        raise SystemExit('Tracking failures recorded; inspect statuses and video')


if __name__ == '__main__':
    main()
