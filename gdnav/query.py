"""Turn a raw nadir drone photo into a north-up, metric-scaled patch comparable to satellite tiles.

Only GNSS-free information is used: heading (compass/IMU) and height (barometer).
Both the query and the satellite tiles get the same circular mask, so the visible ground
is identical in shape regardless of heading (the inscribed circle of the photo's short side).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Camera:
    focal_px: float
    heading_sign: int
    heading_offset_deg: float

    @classmethod
    def load(cls, path: str | Path) -> "Camera":
        d = json.loads(Path(path).read_text())
        return cls(d["focal_px"], d["heading_sign"], d["heading_offset_deg"])

    def gsd(self, height_m: float) -> float:
        return height_m / self.focal_px

    def north_up_ccw(self, heading_deg: float) -> float:
        return self.heading_sign * heading_deg + self.heading_offset_deg


def circle_mask(size: int) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    r = size / 2
    return ((xx + 0.5 - r) ** 2 + (yy + 0.5 - r) ** 2) <= r * r


def apply_circle(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[~circle_mask(img.shape[0])] = 0
    return out


def query_warp(img_shape: tuple[int, ...], height_m: float, heading_deg: float, cam: Camera,
               out_px: int, patch_m: float) -> np.ndarray:
    """2x3 affine: raw photo px -> north-up patch px (rotate about the photo center, scale, recenter)."""
    scale = cam.gsd(height_m) / (patch_m / out_px)
    h, w = img_shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), cam.north_up_ccw(heading_deg), scale)
    m[0, 2] += out_px / 2 - w / 2
    m[1, 2] += out_px / 2 - h / 2
    return m


def make_query(img: np.ndarray, height_m: float, heading_deg: float, cam: Camera,
               out_px: int, patch_m: float) -> np.ndarray:
    """North-up, `patch_m`-wide square at `patch_m / out_px` m/px, centered on the photo center."""
    m = query_warp(img.shape, height_m, heading_deg, cam, out_px, patch_m)
    return apply_circle(cv2.warpAffine(img, m, (out_px, out_px), flags=cv2.INTER_AREA))


def max_patch_m(img_shape: tuple[int, int], height_m: float, cam: Camera) -> float:
    """Diameter of the largest circle fully inside the photo, in meters."""
    return min(img_shape[:2]) * cam.gsd(height_m)
