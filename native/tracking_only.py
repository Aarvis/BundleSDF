"""Pose-only adapter around the installed BundleSDF tracking implementation.

Deliberately does not call BundleSdf.__init__/run: those start and synchronize
with a reconstruction process. The upstream frame, matching, RANSAC and local
GPU bundle-adjustment primitives are reused with a two-reference window without modifying the normal demo.
"""
from pathlib import Path
import time

import numpy as np
import torch

from bundlesdf import BundleSdf, set_bundlesdf_precision
from loftr_wrapper import LoftrRunner
from Utils import yaml
import my_cpp
from tracking_timing import FrameTimings, TimedProxy
from tracking_window import RecentPoseWindow


class PoseOnlyTracker(BundleSdf):
    def __init__(self, configuration: Path, *, loftr=None, initial_object_to_camera=None):
        self.tensor_dtype = set_bundlesdf_precision(use_full_precision=True)
        self.cfg_track = yaml.load(configuration.read_text())
        if self.cfg_track['bundle']['w_sdf'] != 0 or self.cfg_track['bundle']['w_pm'] != 0:
            raise ValueError('Pose-only mode requires reconstruction-dependent losses disabled')
        self.debug_dir = self.cfg_track['debug_dir']
        self.SPDLOG = self.cfg_track['SPDLOG']
        self.use_gui = False
        self.timings = FrameTimings(torch.cuda.synchronize)
        native = my_cpp.Bundler(my_cpp.YamlLoadFile(str(configuration)))
        features = TimedProxy(native._fm, self.timings, {
            'getProcessedImagePairs': 'matching_preparation',
            'rawMatchesToCorres': 'correspondence_filtering',
            'runRansacMultiPairGPU': 'ransac',
            'procrustesByCorrespondence': 'pose_initialization',
        })
        self.bundler = TimedProxy(native, self.timings, {
            'selectKeyFramesForBA': 'keyframe_selection',
            'getFeatureMatchPairs': 'pair_selection',
            'optimizeGPU': 'optimization',
            'checkAndAddKeyframe': 'keyframe_update',
            'forgetFrame': 'frame_cleanup',
        }, children={'_fm': features})
        self.loftr = TimedProxy(loftr if loftr is not None else LoftrRunner(),
                               self.timings, {'predict': 'matching'})
        self.initial_camera_to_object = (np.linalg.inv(initial_object_to_camera)
                                        if initial_object_to_camera is not None else np.eye(4))
        self.recent = RecentPoseWindow()
        self.cnt = -1
        self.K = None
        self.mesh = None

    def track(self, color, depth, intrinsics, frame_id, mask, *, stage_callback=None):
        """Track with synchronized per-stage wall time; preserve upstream logic.

        frame_processing is an inclusive aggregate. stage_ms records exclusive
        times, so nested matching/RANSAC/optimization are not counted twice.
        """
        import cv2
        self.timings.begin(frame_id, stage_callback)
        started = time.perf_counter()
        try:
            with self.timings.stage('frame_preparation'):
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
                initial = self.initial_camera_to_object if not self.recent.frames else np.eye(4)
                frame = self.make_frame(color, depth, intrinsics, frame_id, mask,
                                        occ_mask=None, pose_in_model=initial)
            with self.timings.stage('frame_processing'):
                self.recent.process(self,frame,my_cpp.Frame.Status.FAIL)
            with self.timings.stage('pose_readout'):
                camera_to_object = np.array(frame._pose_in_model, dtype=np.float64, copy=True)
                object_to_camera = np.linalg.inv(camera_to_object)
                valid_pose = (frame._status != my_cpp.Frame.Status.FAIL and
                              np.isfinite(object_to_camera).all())
            ready = time.perf_counter()
        finally:
            details = self.timings.finish()
        totals = details['stage_total_ms']
        return {
            'status': str(frame._status), 'valid': bool(valid_pose),
            'object_to_camera': object_to_camera if valid_pose else None,
            'keyframes': len(self.bundler._keyframes),
            'frame_preparation_ms': totals.get('frame_preparation', 0.),
            'matching_and_pose_ms': totals.get('frame_processing', 0.),
            'matching_ms': totals.get('matching', 0.),
            'ransac_ms': totals.get('ransac', 0.),
            'optimization_ms': totals.get('optimization', 0.),
            'pose_readout_ms': totals.get('pose_readout', 0.),
            'tracking_ms': (ready-started)*1000,
            **details,
            **self.recent.report(),
        }
