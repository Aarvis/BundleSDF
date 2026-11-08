"""Pose tracking against only the last two successful observations."""
import numpy as np


class RecentPoseWindow:
    limit = 2

    def __init__(self):
        self.frames = []
        self.initial_valid_points = None
        self.reference_ids = []
        self.pair_ids = []

    def _retain(self, tracker, frame):
        bundler = tracker.bundler
        self.frames = (self.frames+[frame])[-self.limit:]
        retained = {f._id for f in self.frames}
        # No persistent keyframe pool: every retained valid call is a reference.
        bundler._keyframes = list(self.frames)
        bundler._local_frames = list(self.frames)
        bundler._frames[frame._id] = frame
        for index in list(bundler._frames.keys()):
            if index not in retained:
                bundler.forgetFrame(bundler._frames[index])
        bundler._firstframe = self.frames[0]

    def process(self, tracker, frame, fail_status):
        b = tracker.bundler
        b._newframe = frame
        self.reference_ids = [f._id_str for f in reversed(self.frames)]
        self.pair_ids = [[frame._id_str,f._id_str] for f in reversed(self.frames)]
        tracker.timings.emit("reference_window", "completed",
                             reference_frame_ids=self.reference_ids, image_pairs=self.pair_ids,
                             history_limit=self.limit)

        def reject():
            frame._status = fail_status
            b.forgetFrame(frame)
            b._local_frames = list(self.frames)

        frame.invalidatePixelsByMask(frame._fg_mask)
        if np.count_nonzero(np.asarray(frame._fg_mask)) < 100:
            reject()
            return
        if tracker.cfg_track["depth_processing"]["denoise_cloud"]:
            frame.pointCloudDenoise()
        points = frame.countValidPoints()
        if points <= 0 or (self.initial_valid_points is not None and
                           points < self.initial_valid_points/40.):
            reject()
            return
        if not self.frames:
            # Keep the initial object's coordinate system, not its old image.
            if np.max(np.abs(np.asarray(frame._pose_in_model)-np.eye(4))) <= 1e-4:
                frame.setNewInitCoordinate()
            self.initial_valid_points = points
            self._retain(tracker,frame)
            return

        newest = self.frames[-1]
        frame._ref_frame_id = newest._id
        frame._pose_in_model = np.array(newest._pose_in_model,copy=True)
        # One LoFTR call with <=2 image pairs. No historical/historical rematching
        # and no recovery against a growing pool of older keyframes.
        pairs = [(frame,ref) for ref in reversed(self.frames)]
        tracker.find_corres(pairs)
        if frame._status == fail_status:
            reject()
            return
        minimum = tracker.cfg_track["feature_corres"]["min_match_with_ref"]
        matches = b._fm._matches
        valid_refs = [ref for ref in reversed(self.frames)
                      if (frame,ref) in matches and len(matches[(frame,ref)]) >= minimum]
        if not valid_refs:
            reject()
            return
        # Prefer the latest valid reference; fall back only to the other retained
        # frame. Both already have poses in the same fixed object coordinates.
        reference = valid_refs[0]
        frame._ref_frame_id = reference._id
        offset = b._fm.procrustesByCorrespondence(frame,reference)
        frame._pose_in_model = offset @ np.asarray(frame._pose_in_model)
        b._frames[frame._id] = frame
        local = list(self.frames)+[frame]
        b._local_frames = local
        frozen = [(ref,ref._nerfed) for ref in self.frames]
        try:
            # The native solver treats _nerfed references as fixed poses.
            # This flag freezes known reference poses; no SDF is reconstructed.
            for ref,_ in frozen:
                ref._nerfed = True
            b.optimizeGPU(local,False)
        finally:
            for ref,old in frozen:
                ref._nerfed = old
        if frame._status == fail_status or not np.isfinite(np.asarray(frame._pose_in_model)).all():
            reject()
            return
        self._retain(tracker,frame)

    def report(self):
        return dict(reference_policy="last_two_successful_calls", history_limit=self.limit,
                    retained_frames=len(self.frames), reference_frame_ids=list(self.reference_ids),
                    matching_pair_count=len(self.pair_ids), matching_pairs=list(self.pair_ids))
