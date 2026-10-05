"""GNSS-free target geolocation: image pixel -> ray -> ground plane -> lat/lon, for a given camera pose.

Two pose sources are compared downstream:
  * visual PnP pose (gdnav/pose.py, from the image<->map matches; independent of GNSS and of the EKF)
  * autopilot pose: EKF position + attitude (drifts without GNSS)
Frames: world = local ENU (east, north, up) around a reference lat/lon; camera = OpenCV (x right, y down, z forward).
"""
from __future__ import annotations

import math

import numpy as np

from .geo import meters_per_degree

# ENU -> NED axis permutation
P_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def r_ned_body(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Body (x fwd, y right, z down) -> NED, ZYX Euler."""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def camera_pose_from_attitude(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """R_cw (world ENU -> camera) for the nadir camera of the simulated Zephyr.

    Mount (measured, see sim/eval_recording.py): image right = body forward, image down = body right, optical
    axis = body down, i.e. camera axes coincide with body axes.
    """
    R_cam_body = np.eye(3)
    return R_cam_body @ r_ned_body(roll_deg, pitch_deg, yaw_deg).T @ P_NED_ENU


def pixel_to_ground(uv: np.ndarray, K: np.ndarray, R_cw: np.ndarray, center_enu: np.ndarray,
                    ground_up_m: float = 0.0) -> np.ndarray:
    """Intersect the rays of pixels uv (N x 2) with the plane up = ground_up_m. Returns N x 2 (east, north)."""
    uv1 = np.c_[np.asarray(uv, float), np.ones(len(uv))]
    d = (R_cw.T @ np.linalg.inv(K) @ uv1.T).T                      # ray directions in world
    t = (ground_up_m - center_enu[2]) / d[:, 2]
    p = center_enu[None, :] + t[:, None] * d
    return p[:, :2]


def enu_to_ll(east_north: np.ndarray, ref_ll: tuple[float, float]) -> np.ndarray:
    m_lat, m_lon = meters_per_degree(ref_ll[0])
    return np.c_[ref_ll[0] + east_north[:, 1] / m_lat, ref_ll[1] + east_north[:, 0] / m_lon]
