"""Trusted local IPC worker for WMAM's single manipulation target. No simulator."""
import argparse
import json
import logging
from multiprocessing.connection import Connection
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from tracking_only import PoseOnlyTracker
from tracking_timing import StageProgress
from loftr_wrapper import LoftrRunner
from Utils import yaml, set_seed


def configuration(root, run_dir, token, depth, mask):
    cfg = yaml.load((root / 'BundleTrack/config_ho3d.yml').read_text())
    directory = run_dir / token
    directory.mkdir(parents=True, exist_ok=True)
    cfg['debug_dir'] = str(directory) + '/'
    cfg['SPDLOG'] = 0
    cfg['depth_processing']['percentile'] = 100
    # The milk example assumed depth < 1 m. LIBERO's workspace camera is farther.
    valid = depth[(mask > 0) & (depth > .1) & np.isfinite(depth)]
    cfg['depth_processing']['zfar'] = max(1., float(np.percentile(valid, 99.5)) + .2)
    cfg['bundle']['w_sdf'] = cfg['bundle']['w_pm'] = 0
    cfg['feature_corres'].update({'resize': 400, 'rematch_after_nerf': False, 'map_points': True,
        'max_dist_neighbor': .02, 'max_normal_neighbor': 45,
        'max_dist_no_neighbor': .02, 'max_normal_no_neighbor': 45})
    # PoseOnlyTracker selects the last two successful calls explicitly.
    cfg['bundle'].update({'window_size':2,'max_BA_frames':3})
    cfg['ransac'].update({'inlier_dist': .01, 'inlier_normal_angle': 30})
    cfg['p2p'].update({'max_dist': .02, 'max_normal_angle': 45})
    path = directory / 'tracking.yml'
    with path.open('w') as f:
        yaml.dump(cfg, f)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    connection = Connection(args.fd)
    logging.getLogger().setLevel(logging.WARNING)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    set_seed(0)
    model = LoftrRunner()
    torch.cuda.synchronize()
    connection.send({'status': 'ready', 'gpu': torch.cuda.get_device_name()})
    tracker, token, last_sim_time, last_capture = None, None, None, None
    root = Path(__file__).resolve().parents[1]
    progress = StageProgress(args.run_dir)
    with (args.run_dir / 'poses.jsonl').open('a', buffering=1) as evidence:
        try:
            while True:
                packet = connection.recv()
                started = time.perf_counter()
                progress.begin(packet)
                progress.emit('worker_preparation', 'started')
                result = {k: packet[k] for k in ('token', 'command_id', 'target', 'frame_id', 'captured_at', 'sim_time')}
                try:
                    depth = np.nan_to_num(np.asarray(packet['depth'], dtype=np.float32), nan=0., posinf=0., neginf=0.)
                    mask = np.asarray(packet['mask'], dtype=np.uint8)
                    world_from_camera = np.asarray(packet['world_from_camera'], dtype=float)
                    if token != packet['token']:
                        tracker = None
                        torch.cuda.empty_cache()
                        path = configuration(root, args.run_dir, packet['token'], depth, mask)
                        initial = np.linalg.inv(world_from_camera) @ packet['initial_world_from_object']
                        tracker = PoseOnlyTracker(path, loftr=model, initial_object_to_camera=initial)
                        token, last_sim_time, last_capture = packet['token'], None, None
                    # Motion rejection must account for sampling gaps. Keep a
                    # bounded 0.6 m/s translation budget, rather than applying
                    # the milk video's 2 cm gate to five simulator captures.
                    gap = (packet['sim_time'] - last_sim_time) if packet['sim_time'] is not None and last_sim_time is not None else (
                        packet['captured_at'] - last_capture if last_capture is not None else 1/30)
                    translation_budget = max(.02, min(.25, .6 * max(0., gap)))
                    rotation_budget = max(30., min(90., 180. * max(0., gap)))
                    for cfg in (tracker.cfg_track,):
                        cfg['ransac']['max_trans_neighbor'] = translation_budget
                        cfg['ransac']['max_rot_deg_neighbor'] = rotation_budget
                    # The C++ YAML node is independent of the Python mapping;
                    # update it through the binding used by bundle configuration.
                    tracker.bundler.yml['ransac'].set_float('max_trans_neighbor', translation_budget)
                    tracker.bundler.yml['ransac'].set_float('max_rot_deg_neighbor', rotation_budget)
                    color = cv2.cvtColor(np.asarray(packet['rgb'], dtype=np.uint8), cv2.COLOR_RGB2BGR)
                    preparation_ms = (time.perf_counter()-started)*1000
                    progress.emit('worker_preparation', 'completed', elapsed_ms=preparation_ms)
                    tracked = tracker.track(color, depth, packet['intrinsics'], str(packet['frame_id']), mask,
                                            stage_callback=progress.emit)
                    world_pose = (world_from_camera @ tracked['object_to_camera']) if tracked['valid'] else None
                    result.update(valid=tracked['valid'], reason=tracked['status'],
                                  world_from_object=world_pose.tolist() if world_pose is not None else None,
                                  tracking_ms=tracked['tracking_ms'], keyframes=tracked['keyframes'],
                                  worker_preparation_ms=preparation_ms,
                                  **{key: tracked[key] for key in ('frame_preparation_ms','matching_ms','ransac_ms',
                                      'optimization_ms','pose_readout_ms','matching_and_pose_ms',
                                      'stage_ms','stage_total_ms','stage_counts','timing_method',
                                      'reference_policy','history_limit','retained_frames','reference_frame_ids',
                                      'matching_pair_count','matching_pairs')},
                                  source_camera=packet['camera'], motion_budget_m=translation_budget)
                    if tracked['valid']:
                        last_sim_time, last_capture = packet['sim_time'], packet['captured_at']
                except Exception as exc:
                    result.update(valid=False, reason=f'{type(exc).__name__}: {exc}')
                    if tracker is not None and hasattr(tracker,'timings'):
                        result.update(tracker.timings.report())
                    progress.emit('worker', 'failed', error=result['reason'])
                result['worker_ms'] = (time.perf_counter() - started) * 1000
                result['finished_at'] = time.time()
                evidence.write(json.dumps(result) + '\n')
                progress.emit('worker', 'completed', elapsed_ms=result['worker_ms'], valid=result['valid'])
                connection.send(result)
        except (EOFError, BrokenPipeError, ConnectionResetError):
            pass
        finally:
            progress.close()
            connection.close()


if __name__ == '__main__':
    main()
