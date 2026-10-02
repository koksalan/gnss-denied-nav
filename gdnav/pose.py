"""Camera pose from image <-> map correspondences (visual attitude + position, no IMU attitude needed).

The ground is treated as the plane z = 0 in a local East-North-Up frame, so image points and their map
positions are related by a homography. Solving the planar PnP problem gives the camera's 3D position
(including height above ground) and its full orientation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraPose:
    center_enu: np.ndarray        # camera position (east, north, up) in m, relative to the map points' origin
    R_cw: np.ndarray              # rotation world(ENU) -> camera (OpenCV: x right, y down, z forward)
    inliers: int
    reproj_px: float              # median reprojection error of the inliers

    @property
    def off_nadir_deg(self) -> float:
        """Angle between the optical axis and straight down."""
        axis = self.R_cw.T @ np.array([0.0, 0.0, 1.0])
        return math.degrees(math.acos(np.clip(-axis[2], -1.0, 1.0)))


def intrinsics(focal_px: float, width: int, height: int) -> np.ndarray:
    return np.array([[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]], float)


def solve_pose(image_pts: np.ndarray, ground_en: np.ndarray, K: np.ndarray,
               ransac_px: float = 4.0, min_inliers: int = 12) -> CameraPose | None:
    """image_pts: Nx2 pixels in the raw camera image; ground_en: Nx2 (east, north) meters on the ground plane."""
    if len(image_pts) < min_inliers:
        return None
    obj = np.c_[ground_en, np.zeros(len(ground_en))].astype(np.float64)
    img = image_pts.astype(np.float64)
    # robust inlier set from the plane-to-image homography, then planar PnP (IPPE) + LM refinement
    _, mask = cv2.findHomography(ground_en.astype(np.float64), img, cv2.RANSAC, ransac_px)
    if mask is None or mask.sum() < min_inliers:
        return None
    keep = mask.ravel() == 1
    ok, rvec, tvec = cv2.solvePnP(obj[keep], img[keep], K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj[keep], img[keep], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    center = (-R.T @ tvec).ravel()
    if center[2] <= 0:                                   # camera must be above the ground
        return None
    proj, _ = cv2.projectPoints(obj[keep], rvec, tvec, K, None)
    err = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - img[keep], axis=1)))
    return CameraPose(center, R, int(keep.sum()), err)
