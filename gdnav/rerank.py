"""Fine localization: rerank retrieved candidates with SuperPoint+LightGlue and refine the position.

Heading and scale are known (compass + barometer), so a correct match is a near-identity similarity
transform. We fit one with RANSAC and reject large rotation/scale changes. The refined position is
where the query center lands on the satellite map; the inlier count is the confidence.
The winning match's inlier correspondences are returned too, for full-perspective pose estimation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd

from .geo import SatelliteMap

MAX_ROT_DEG = 15.0
MAX_LOG_SCALE = math.log(1.25)


@dataclass
class Fix:
    lat: float
    lon: float
    inliers: int          # 0 = no valid match (position falls back to the coarse top-1)
    rank: int             # which retrieval candidate won (1-based), -1 if none
    extra: dict = field(default_factory=dict)


@dataclass
class Match:
    """Best candidate: inlier correspondences between the query patch and a north-up satellite crop."""
    rank: int
    inliers: int
    A: np.ndarray                 # 2x3 similarity, query px -> crop px
    query_pts: np.ndarray         # Nx2 inlier keypoints in the query patch
    crop_pts: np.ndarray          # Nx2 inlier keypoints in the crop
    crop_ll: tuple[float, float]  # crop center (lat, lon)
    crop_px: int                  # crop side in px (crop gsd = Reranker.gsd)


class Reranker:
    def __init__(self, device: str, gsd: float = 0.5, search_m: float = 450.0, max_keypoints: int = 2048):
        self.device, self.gsd, self.search_m = device, gsd, search_m
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    def _feats(self, img: np.ndarray):
        return self.extractor.extract(numpy_image_to_torch(img).to(self.device))

    @torch.no_grad()
    def best_match(self, query: np.ndarray, sat: SatelliteMap, candidates_ll: np.ndarray) -> Match | None:
        """query: north-up patch at `self.gsd` m/px. candidates_ll: Kx2 (lat, lon) ranked coarse guesses."""
        fq = self._feats(query)
        kq = rbd(fq)["keypoints"].cpu().numpy()
        spx = int(round(self.search_m / self.gsd))
        best = None
        for rank, (lat, lon) in enumerate(candidates_ll, 1):
            fs = self._feats(sat.crop(lat, lon, self.search_m, self.gsd))
            m = rbd(self.matcher({"image0": fq, "image1": fs}))["matches"].cpu().numpy()
            if len(m) < 8:
                continue
            pa, pb = kq[m[:, 0]], rbd(fs)["keypoints"].cpu().numpy()[m[:, 1]]
            A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=6.0)
            if A is None:
                continue
            s, rot = math.hypot(A[0, 0], A[1, 0]), math.degrees(math.atan2(A[1, 0], A[0, 0]))
            if abs(rot) > MAX_ROT_DEG or abs(math.log(s)) > MAX_LOG_SCALE:
                continue
            keep = inl.ravel() == 1
            if best is None or keep.sum() > best.inliers:
                best = Match(rank, int(keep.sum()), A, pa[keep], pb[keep], (float(lat), float(lon)), spx)
        return best

    def crop_px_to_latlon(self, sat: SatelliteMap, match: Match, x, y):
        """Pixel(s) of the match's crop -> lat/lon."""
        cx, cy = sat.latlon_to_px(*match.crop_ll)                    # crop center in native px
        nx = cx + (np.asarray(x) - match.crop_px / 2) * self.gsd / sat.gsd_x
        ny = cy + (np.asarray(y) - match.crop_px / 2) * self.gsd / sat.gsd_y
        return sat.px_to_latlon(nx, ny)

    def localize(self, query: np.ndarray, sat: SatelliteMap, candidates_ll: np.ndarray) -> Fix:
        """Position of the query patch center (no attitude handling)."""
        best = self.best_match(query, sat, candidates_ll)
        if best is None:
            lat, lon = candidates_ll[0]
            return Fix(float(lat), float(lon), 0, -1)
        c = best.A @ np.array([query.shape[1] / 2, query.shape[0] / 2, 1.0])
        plat, plon = self.crop_px_to_latlon(sat, best, c[0], c[1])
        return Fix(float(plat), float(plon), best.inliers, best.rank)
