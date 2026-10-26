"""Bounded official-milk-sequence validation; no GUI or final global refinement."""
import argparse
import json
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--video', type=Path, default=root / '.native/data/milk/2022-11-18-15-10-24_milk')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--train-steps', type=int, default=50)
    args = parser.parse_args()
    if args.frames < 2 or args.stride < 1 or args.train_steps < 1:
        parser.error('frames >= 2, stride >= 1, train-steps >= 1 required')
    if args.output.exists():
        parser.error('Choose a new output directory; existing results are never deleted')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)

    import cv2
    import numpy as np
    import torch
    from bundlesdf import BundleSdf
    from BundleTrack.scripts.data_reader import YcbineoatReader
    from Utils import yaml, set_seed
    set_seed(0)
    track = yaml.load((root / 'BundleTrack/config_ho3d.yml').read_text())
    track['SPDLOG'] = 0
    track['debug_dir'] = str(args.output) + '/'
    track['depth_processing']['percentile'] = 95
    track['erode_mask'] = 3
    track['feature_corres']['resize'] = 400
    track['feature_corres']['rematch_after_nerf'] = True
    track['feature_corres']['max_dist_neighbor'] = 0.02
    track['feature_corres']['max_normal_neighbor'] = 30
    track['feature_corres']['max_dist_no_neighbor'] = 0.01
    track['feature_corres']['max_normal_no_neighbor'] = 20
    track['feature_corres']['map_points'] = True
    track['keyframe']['min_rot'] = 2
    track['ransac']['inlier_dist'] = 0.01
    track['ransac']['inlier_normal_angle'] = 20
    track['ransac']['max_trans_neighbor'] = 0.02
    track['ransac']['max_rot_deg_neighbor'] = 30
    track['ransac']['max_trans_no_neighbor'] = 0.01
    track['ransac']['max_rot_no_neighbor'] = 10
    track['p2p']['max_dist'] = 0.02
    track['p2p']['max_normal_angle'] = 45
    nerf = yaml.load((root / 'config.yml').read_text())
    nerf.update(dict(continual=True, n_step=args.train_steps, trunc_start=0.01, trunc=0.01,
                mesh_resolution=0.005, fs_sdf=0.1, down_scale_ratio=1,
                far=track['depth_processing']['zfar'], notes='',
                datadir=str(args.output / 'nerf'), save_dir=str(args.output / 'nerf'),
                expname='native_smoke'))
    track_path, nerf_path = args.output / 'track.yml', args.output / 'nerf.yml'
    with track_path.open('w') as f: yaml.dump(track, f)
    with nerf_path.open('w') as f: yaml.dump(nerf, f)
    reader = YcbineoatReader(video_dir=str(args.video), shorter_side=480)
    tracker = None
    started = time.monotonic()
    timings, frame_results = [], []
    try:
        tracker = BundleSdf(cfg_track_dir=str(track_path), cfg_nerf_dir=str(nerf_path),
                            start_nerf_keyframes=3, use_gui=False)
        for i in list(range(0, len(reader.color_files), args.stride))[:args.frames]:
            color = cv2.imread(reader.color_files[i])
            depth = reader.get_depth(i)
            h, w = depth.shape
            color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(reader.get_mask(i), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            tick = time.monotonic()
            tracker.run(color, depth, reader.K.copy(), reader.id_strs[i], mask=mask,
                        occ_mask=None, pose_in_model=np.eye(4))
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            timings.append(elapsed)
            frame = tracker.bundler._newframe
            # The enum is bound to Python; keep both status and pose in evidence.
            pose = np.array(frame._pose_in_model)
            frame_results.append({'id': reader.id_strs[i], 'status': str(frame._status),
                                  'pose_in_model': pose.tolist(), 'seconds': elapsed})
            print('FRAME', len(frame_results), reader.id_strs[i], str(frame._status),
                  'keyframes', len(tracker.bundler._keyframes), 'seconds', round(elapsed, 3), flush=True)
        tracker.on_finish()
        mesh = tracker.p_dict.get('mesh')
        mesh_file = None
        if mesh is not None and len(mesh.vertices):
            mesh_file = str(args.output / 'online_mesh.obj')
            mesh.export(mesh_file)
        report = {'frames_processed': len(frame_results), 'stride': args.stride,
                  'training_steps_per_update': args.train_steps, 'start_nerf_keyframes': 3,
                  'keyframe_min_rotation_deg': 2, 'global_refinement': False,
                  'elapsed_seconds': time.monotonic() - started,
                  'keyframes': len(tracker.bundler._keyframes), 'mesh': mesh_file,
                  'mesh_vertices': len(mesh.vertices) if mesh is not None else 0,
                  'tracking_results': frame_results,
                  'parent_peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'note': 'Parent allocator excludes reconstruction child and other GPU processes.'}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'tracking_results'}, indent=2), flush=True)
        if len(frame_results) < 2 or any('FAIL' in f['status'] for f in frame_results):
            raise RuntimeError('Tracking failed for one or more sample frames')
        if not mesh_file:
            raise RuntimeError('Tracking ran, but no reconstructed mesh was produced')
    finally:
        if tracker is not None:
            if tracker.p_nerf.is_alive():
                tracker.p_nerf.terminate()
                tracker.p_nerf.join(timeout=10)
            tracker.manager.shutdown()

if __name__ == '__main__':
    main()
