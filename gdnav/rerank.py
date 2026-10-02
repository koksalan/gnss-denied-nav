"""Fine localization: rerank retrieved candidates with SuperPoint+LightGlue and refine the position.

Heading and scale are known (compass + barometer), so a correct match is a near-identity similarity
transform. We fit one with RANSAC and reject large rotation/scale changes. The refined position is
where the query center lands on the satellite map; the inlier count is the confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

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


class Reranker:
    def __init__(self, device: str, gsd: float = 0.5, search_m: float = 450.0, max_keypoints: int = 2048):
        self.device, self.gsd, self.search_m = device, gsd, search_m
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    def _feats(self, img: np.ndarray):
        return self.extractor.extract(numpy_image_to_torch(img).to(self.device))

    @torch.no_grad()
    def localize(self, query: np.ndarray, sat: SatelliteMap, candidates_ll: np.ndarray) -> Fix:
        """query: north-up patch at `self.gsd` m/px. candidates_ll: Kx2 (lat, lon) ranked coarse guesses."""
        fq = self._feats(query)
        kq = rbd(fq)["keypoints"].cpu().numpy()
        qc = np.array([query.shape[1] / 2, query.shape[0] / 2, 1.0])
        spx = int(round(self.search_m / self.gsd))
        best = None
        for rank, (lat, lon) in enumerate(candidates_ll, 1):
            crop = sat.crop(lat, lon, self.search_m, self.gsd)
            fs = self._feats(crop)
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
            n_in = int(inl.sum())
            if best is None or n_in > best.inliers:
                c = A @ qc                                                  # query center in crop px
                cx, cy = sat.latlon_to_px(lat, lon)                         # crop center in native px
                nx = cx + (c[0] - spx / 2) * self.gsd / sat.gsd_x
                ny = cy + (c[1] - spx / 2) * self.gsd / sat.gsd_y
                plat, plon = sat.px_to_latlon(nx, ny)
                best = Fix(float(plat), float(plon), n_in, rank)
        if best is None:
            lat, lon = candidates_ll[0]
            best = Fix(float(lat), float(lon), 0, -1)
        return best
