"""Frame-to-frame visual odometry: how far did the aircraft move between two consecutive camera frames?

Bridges the gaps of map matching (water, uniform fields, banked turns): map fixes have no drift but are
intermittent; odometry is continuous but drifts. The node uses odometry only between map fixes and reports an
accuracy that grows with the distance flown since the last fix.

Method: track corners between the two frames (pyramidal Lucas-Kanade), project every tracked point to the
ground with each frame's own attitude (roll/pitch/yaw from the autopilot) and height (barometer), and take the
robust median of the ground-point offsets. Because each frame is projected with its own attitude, rotation of
the aircraft (pitch/roll changes, turns) does not show up as false translation.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geolocate import camera_pose_from_attitude, pixel_to_ground


@dataclass
class OdomStep:
    d_east: float
    d_north: float
    n_tracks: int
    spread_m: float          # robust spread of the per-point offsets (quality)


class VisualOdometry:
    def __init__(self, K: np.ndarray, work_width: int = 640, max_corners: int = 400, min_tracks: int = 30):
        self.K, self.work_width, self.max_corners, self.min_tracks = K, work_width, max_corners, min_tracks
        self.prev = None          # (gray, pts_full_res, roll, pitch, yaw, height)

    def reset(self):
        self.prev = None

    def step(self, frame_rgb: np.ndarray, height_m: float, roll_deg: float, pitch_deg: float,
             yaw_deg: float) -> OdomStep | None:
        """Aircraft displacement (east, north) in meters since the previous call, or None (first frame / lost)."""
        s = self.work_width / frame_rgb.shape[1]
        gray = cv2.cvtColor(cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        out = None
        if self.prev is not None:
            g0, p0, r0, pi0, y0, h0 = self.prev
            if p0 is not None and len(p0) >= self.min_tracks:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, gray, p0, None, winSize=(21, 21), maxLevel=3)
                # forward-backward check removes bad tracks
                p0b, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, g0, p1, None, winSize=(21, 21), maxLevel=3)
                ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (np.linalg.norm(p0 - p0b, axis=2).ravel() < 1.0)
                if ok.sum() >= self.min_tracks:
                    a, b = p0[ok, 0] / s, p1[ok, 0] / s                      # full-resolution pixels
                    ga = pixel_to_ground(a, self.K, camera_pose_from_attitude(r0, pi0, y0), np.array([0.0, 0.0, h0]))
                    gb = pixel_to_ground(b, self.K, camera_pose_from_attitude(roll_deg, pitch_deg, yaw_deg),
                                         np.array([0.0, 0.0, height_m]))
                    d = ga - gb                                               # = camera displacement, per point
                    med = np.median(d, axis=0)
                    dev = np.linalg.norm(d - med, axis=1)
                    mad = float(np.median(dev))
                    keep = dev < max(3.0 * mad, 1.0)
                    if keep.sum() >= self.min_tracks:
                        med = np.median(d[keep], axis=0)
                        out = OdomStep(float(med[0]), float(med[1]), int(keep.sum()), mad)
        pts = cv2.goodFeaturesToTrack(gray, self.max_corners, 0.01, 12)
        self.prev = (gray, pts, roll_deg, pitch_deg, yaw_deg, height_m)
        return out
