"""Onboard visual localizer: camera frame + heading + height  ->  lat/lon + confidence.

Two modes:
  * global relocalization: search every tile of the map (start-up, or after losing track)
  * tracking: only tiles within `track_radius_m` of the previous fix (faster, fewer false matches)

The tile database is built once at start-up from the satellite map of the operation area.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np
import torch

from .embed import DinoEmbedder, embed_images
from .geo import SatelliteMap, haversine_m, meters_per_degree
from .query import Camera, apply_circle, make_query
from .rerank import Fix, Reranker


def boresight_offset_ne(height_m: float, roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float]:
    """Where a body-fixed nadir camera's optical axis hits the ground, relative to the aircraft (north, east) in m.

    Body frame: x forward, y right, z down (ArduPilot/NED). Axis = R_zyx(yaw, pitch, roll) @ [0, 0, 1].
    Nose-up pitch moves the ground point ahead of the aircraft; right roll moves it to the left.
    """
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    bn = math.cos(r) * math.sin(p) * math.cos(y) + math.sin(r) * math.sin(y)
    be = math.cos(r) * math.sin(p) * math.sin(y) - math.sin(r) * math.cos(y)
    bd = math.cos(r) * math.cos(p)
    return height_m * bn / bd, height_m * be / bd


@dataclass
class LocalizerConfig:
    patch_m: float                   # ground diameter compared (<= circle inside the camera footprint)
    px: int = 224
    stride_m: float = 50.0
    rerank_k: int = 10
    track_k: int = 5
    track_radius_m: float = 400.0
    min_inliers: int = 25
    max_tilt_deg: float = 25.0       # body-fixed camera: beyond this the view is too oblique to trust
    model: str = "facebook/dinov2-base"
    pool: str = "cls+gem"


class Localizer:
    def __init__(self, sat: SatelliteMap, cfg: LocalizerConfig, weights: str | None, device: str = "cuda",
                 bounds_ll: tuple[float, float, float, float] | None = None):
        """bounds_ll = (lat_min, lon_min, lat_max, lon_max) restricts the database to the operation area."""
        self.sat, self.cfg, self.device = sat, cfg, device
        self.model = DinoEmbedder(cfg.model, pool=cfg.pool).to(device).eval()
        if weights:
            self.model.load_state_dict(torch.load(weights, map_location=device))
        self.reranker = Reranker(device)
        gsd = cfg.patch_m / cfg.px
        self.overview = sat.read_overview(gsd)
        h, w = self.overview.shape[:2]
        self._sx, self._sy = sat.width / w, sat.height / h
        half, step = cfg.px / 2, cfg.stride_m / gsd
        xs, ys = np.arange(half, w - half, step), np.arange(half, h - half, step)
        cx, cy = np.meshgrid(xs, ys)
        centers = np.stack([cx.ravel(), cy.ravel()], 1)
        lat, lon = sat.px_to_latlon(centers[:, 0] * self._sx, centers[:, 1] * self._sy)
        keep = np.ones(len(centers), bool)
        if bounds_ll is not None:
            la0, lo0, la1, lo1 = bounds_ll
            keep = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        self.centers, self.tiles_ll = centers[keep], np.stack([lat[keep], lon[keep]], 1)
        tiles = (apply_circle(self._crop(x, y)) for x, y in self.centers)
        batch, embs = [], []
        for t in tiles:
            batch.append(t)
            if len(batch) == 512:
                embs.append(embed_images(self.model, batch, device)); batch = []
        if batch:
            embs.append(embed_images(self.model, batch, device))
        self.db = torch.from_numpy(np.concatenate(embs)).to(device)
        m_lat, m_lon = meters_per_degree(float(self.tiles_ll[:, 0].mean()))
        self._tiles_m = torch.from_numpy(np.stack([self.tiles_ll[:, 0] * m_lat, self.tiles_ll[:, 1] * m_lon], 1)).to(device)
        self._mpd = (m_lat, m_lon)

    def _crop(self, cx, cy):
        p = self.cfg.px
        x0, y0 = int(round(cx - p / 2)), int(round(cy - p / 2))
        return self.overview[y0:y0 + p, x0:x0 + p]

    @torch.no_grad()
    def localize(self, frame: np.ndarray, height_m: float, heading_deg: float, cam: Camera,
                 prior_ll: tuple[float, float] | None = None,
                 roll_deg: float = 0.0, pitch_deg: float = 0.0) -> tuple[Fix, str]:
        """Returns (fix, mode) for the AIRCRAFT position (image-center fix corrected for attitude).

        fix.inliers >= cfg.min_inliers means a trusted position; mode 'tilted' = frame skipped.
        """
        if math.hypot(roll_deg, pitch_deg) > self.cfg.max_tilt_deg:
            return Fix(float("nan"), float("nan"), 0, -1), "tilted"
        q = make_query(frame, height_m, heading_deg, cam, self.cfg.px, self.cfg.patch_m)
        e = torch.from_numpy(embed_images(self.model, [q], self.device)).to(self.device)[0]
        sim = self.db @ e
        mode, k = "global", self.cfg.rerank_k
        if prior_ll is not None:
            p = torch.tensor([prior_ll[0] * self._mpd[0], prior_ll[1] * self._mpd[1]], device=self.device)
            near = (self._tiles_m - p).norm(dim=1) < self.cfg.track_radius_m
            if near.any():
                sim = torch.where(near, sim, torch.full_like(sim, -2.0))
                mode, k = "track", self.cfg.track_k
        top = sim.topk(k).indices.cpu().numpy()
        fine_gsd = self.reranker.gsd
        qpx = int(round(self.cfg.patch_m / fine_gsd))
        q_fine = make_query(frame, height_m, heading_deg, cam, qpx, self.cfg.patch_m)
        fix = self.reranker.localize(q_fine, self.sat, self.tiles_ll[top])
        dn, de = boresight_offset_ne(height_m, roll_deg, pitch_deg, heading_deg)
        fix.lat, fix.lon = fix.lat - dn / self._mpd[0], fix.lon - de / self._mpd[1]
        return fix, mode


def error_m(fix: Fix, lat: float, lon: float) -> float:
    return float(haversine_m(lat, lon, fix.lat, fix.lon))
