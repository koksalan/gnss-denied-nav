"""Pre-flight map preparation: the operation area's satellite map in RAM + SuperPoint features computed ONCE.

Profiling showed most of a fix's time went into re-reading the GeoTIFF (16 ms per candidate, far more over WSL's
/mnt/c) and re-running SuperPoint on the same map crops (11 ms each). Real systems prepare the map before the
mission; in flight a candidate window is just a slice of precomputed keypoints.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from lightglue.utils import numpy_image_to_torch

from .geo import SatelliteMap, meters_per_degree


@dataclass
class MapFeatures:
    image: np.ndarray            # north-up area map at `gsd` m/px (H x W x 3)
    gsd: float
    center_ll: tuple[float, float]
    keypoints: torch.Tensor      # N x 2 (x, y) in map px, on device
    descriptors: torch.Tensor    # N x 256
    scores: torch.Tensor         # N

    @classmethod
    def build(cls, sat: SatelliteMap, center_ll: tuple[float, float], extent_m: float, gsd: float, extractor,
              device: str, tile_px: int = 1024, overlap_px: int = 64, kp_per_tile: int = 2600) -> "MapFeatures":
        img = sat.crop(center_ll[0], center_ll[1], extent_m, gsd)
        H, W = img.shape[:2]
        step = tile_px - 2 * overlap_px
        kps, descs, scores = [], [], []
        old_max = extractor.conf.max_num_keypoints             # extract() kwargs only reach the preprocessor
        extractor.conf.max_num_keypoints = kp_per_tile
        with torch.no_grad():
            for y0 in range(-overlap_px, H, step):
                for x0 in range(-overlap_px, W, step):
                    xa, ya = max(x0, 0), max(y0, 0)
                    xb, yb = min(x0 + tile_px, W), min(y0 + tile_px, H)
                    if xb - xa < 64 or yb - ya < 64:
                        continue
                    f = extractor.extract(numpy_image_to_torch(img[ya:yb, xa:xb]).to(device))
                    k = f["keypoints"][0] + torch.tensor([xa, ya], device=device, dtype=torch.float32)
                    # keep only keypoints in this tile's core (each map pixel belongs to exactly one core)
                    cx0, cy0 = x0 + overlap_px, y0 + overlap_px
                    core = (k[:, 0] >= cx0) & (k[:, 0] < cx0 + step) & (k[:, 1] >= cy0) & (k[:, 1] < cy0 + step)
                    kps.append(k[core]); descs.append(f["descriptors"][0][core]); scores.append(f["keypoint_scores"][0][core])
        extractor.conf.max_num_keypoints = old_max
        return cls(img, gsd, center_ll, torch.cat(kps), torch.cat(descs), torch.cat(scores))

    # --- geometry --------------------------------------------------------------------------------------------
    def ll_to_px(self, lat, lon):
        m_lat, m_lon = meters_per_degree(self.center_ll[0])
        H, W = self.image.shape[:2]
        return (W / 2 + (np.asarray(lon) - self.center_ll[1]) * m_lon / self.gsd,
                H / 2 - (np.asarray(lat) - self.center_ll[0]) * m_lat / self.gsd)

    def px_to_ll(self, x, y):
        m_lat, m_lon = meters_per_degree(self.center_ll[0])
        H, W = self.image.shape[:2]
        return (self.center_ll[0] - (np.asarray(y) - H / 2) * self.gsd / m_lat,
                self.center_ll[1] + (np.asarray(x) - W / 2) * self.gsd / m_lon)

    def window(self, cx: float, cy: float, size_px: int) -> dict:
        """LightGlue input for a square window centered at map px (cx, cy); keypoints relative to the window."""
        x0, y0 = cx - size_px / 2, cy - size_px / 2
        k = self.keypoints
        sel = (k[:, 0] >= x0) & (k[:, 0] < x0 + size_px) & (k[:, 1] >= y0) & (k[:, 1] < y0 + size_px)
        dev = k.device
        return {"keypoints": (k[sel] - torch.tensor([x0, y0], device=dev, dtype=k.dtype))[None],
                "descriptors": self.descriptors[sel][None],
                "keypoint_scores": self.scores[sel][None],
                "image_size": torch.tensor([[size_px, size_px]], device=dev, dtype=torch.float32)}
