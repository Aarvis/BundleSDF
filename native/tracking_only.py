"""Pose-only adapter around the installed BundleSDF tracking implementation.

Deliberately does not call BundleSdf.__init__/run: those start and synchronize
with a reconstruction process. The upstream frame, matching, RANSAC and local
GPU bundle-adjustment methods are reused without modifying the normal demo.
"""
from pathlib import Path
import time

import numpy as np
import torch

from bundlesdf import BundleSdf, set_bundlesdf_precision
from loftr_wrapper import LoftrRunner
from Utils import yaml
import my_cpp


class PoseOnlyTracker(BundleSdf):
    def __init__(self, configuration: Path, *, loftr=None, initial_object_to_camera=None):
        self.tensor_dtype = set_bundlesdf_precision(use_full_precision=True)
        self.cfg_track = yaml.load(configuration.read_text())
        if self.cfg_track['bundle']['w_sdf'] != 0 or self.cfg_track['bundle']['w_pm'] != 0:
            raise ValueError('Pose-only mode requires reconstruction-dependent losses disabled')
        self.debug_dir = self.cfg_track['debug_dir']
        self.SPDLOG = self.cfg_track['SPDLOG']
        self.use_gui = False
        self.bundler = my_cpp.Bundler(my_cpp.YamlLoadFile(str(configuration)))
        self.loftr = loftr if loftr is not None else LoftrRunner()
        self.initial_camera_to_object = (np.linalg.inv(initial_object_to_camera)
                                        if initial_object_to_camera is not None else np.eye(4))
        self.cnt = -1
        self.K = None
        self.mesh = None

    def track(self, color, depth, intrinsics, frame_id, mask):
        """Return current online object-to-camera pose and synchronized timings.

        Input decoding/resizing and video writing are intentionally outside this
        method. Mask erosion is included in frame preparation. No ground-truth
        pose, CAD model, SDF, or fused point cloud is supplied.
        """
        import cv2
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.cnt += 1
        if self.K is None:
            self.K = intrinsics.copy()
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
        percentile = self.cfg_track['depth_processing']['percentile']
        if percentile < 100:
            valid = (depth >= 0.1) & (mask > 0) & np.isfinite(depth)
            if not np.any(valid):
                raise ValueError(f'No valid masked depth in frame {frame_id}')
            threshold = np.percentile(depth[valid], percentile)
            depth[depth >= threshold] = 0
        initial = self.initial_camera_to_object if self.cnt == 0 else np.eye(4)
        frame = self.make_frame(color, depth, intrinsics, frame_id, mask,
                                occ_mask=None, pose_in_model=initial)
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        self.process_new_frame(frame)
        torch.cuda.synchronize()
        estimated = time.perf_counter()
        camera_to_object = np.array(frame._pose_in_model, dtype=np.float64, copy=True)
        object_to_camera = np.linalg.inv(camera_to_object)
        valid_pose = (frame._status != my_cpp.Frame.Status.FAIL and
                      np.isfinite(object_to_camera).all())
        ready = time.perf_counter()
        return {
            'status': str(frame._status), 'valid': bool(valid_pose),
            'object_to_camera': object_to_camera if valid_pose else None,
            'keyframes': len(self.bundler._keyframes),
            'frame_preparation_ms': (prepared - started) * 1000,
            'matching_and_pose_ms': (estimated - prepared) * 1000,
            'pose_readout_ms': (ready - estimated) * 1000,
            'tracking_ms': (ready - started) * 1000,
        }
