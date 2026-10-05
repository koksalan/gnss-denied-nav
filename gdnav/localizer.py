"""Onboard visual localizer: camera frame + heading + height  ->  lat/lon + confidence.

Two modes:
  * global relocalization: search every tile of the map (start-up, or after losing track)
  * tracking: only tiles within `track_radius_m` of the previous fix (faster, fewer false matches)

The tile database is built once at start-up from the satellite map of the operation area.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .embed import DinoEmbedder, embed_images
from .geo import SatelliteMap, haversine_m, meters_per_degree
from .mapfeatures import MapFeatures
from .pose import intrinsics, solve_pose
from .query import Camera, apply_circle, make_query, query_warp
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
    pose: str = "pnp"                # "pnp": camera pose from the matches; "boresight": correct with IMU attitude
    fast: bool = True                # precomputed map features (needs bounds_ll); tracking = one window, no DINO
    track_window_m: float = 600.0    # tracking search window around the prior (query footprint ~250 m)
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
        self.mapf = None
        if cfg.fast and bounds_ll is not None:
            la0, lo0, la1, lo1 = bounds_ll
            center = ((la0 + la1) / 2, (lo0 + lo1) / 2)
            extent = max((la1 - la0) * m_lat, (lo1 - lo0) * m_lon) + 600.0     # margin for windows at the edge
            self.mapf = MapFeatures.build(sat, center, extent, self.reranker.gsd, self.reranker.extractor, device)

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
        rr = self.reranker
        qpx = int(round(self.cfg.patch_m / rr.gsd))
        M = query_warp(frame.shape, height_m, heading_deg, cam, qpx, self.cfg.patch_m)
        q_fine = apply_circle(cv2.warpAffine(frame, M, (qpx, qpx), flags=cv2.INTER_AREA))
        mode, sims = "global", np.array([np.nan, np.nan])
        if self.mapf is not None and prior_ll is not None:
            # tracking: one window around the prior, precomputed map keypoints, no retrieval
            mode = "track"
            cxy = np.array(self.mapf.ll_to_px(*prior_ll), dtype=float)[None]
            match = rr.best_match_map(q_fine, self.mapf, cxy, int(round(self.cfg.track_window_m / rr.gsd)))
            top = None
        else:
            q = make_query(frame, height_m, heading_deg, cam, self.cfg.px, self.cfg.patch_m)
            e = torch.from_numpy(embed_images(self.model, [q], self.device)).to(self.device)[0]
            sim = self.db @ e
            k = self.cfg.rerank_k
            if prior_ll is not None:
                p = torch.tensor([prior_ll[0] * self._mpd[0], prior_ll[1] * self._mpd[1]], device=self.device)
                near = (self._tiles_m - p).norm(dim=1) < self.cfg.track_radius_m
                if near.any():
                    sim = torch.where(near, sim, torch.full_like(sim, -2.0))
                    mode, k = "track", self.cfg.track_k
            topv, topi = sim.topk(k)
            top = topi.cpu().numpy()
            sims = topv.float().cpu().numpy()
            if self.mapf is not None:
                cxy = np.stack(self.mapf.ll_to_px(self.tiles_ll[top, 0], self.tiles_ll[top, 1]), 1)
                match = rr.best_match_map(q_fine, self.mapf, cxy, int(round(rr.search_m / rr.gsd)))
            else:
                match = rr.best_match(q_fine, self.sat, self.tiles_ll[top])
        feats = dict(mode=mode, sim_top1=float(sims[0]), sim_margin=float(sims[0] - sims[1]) if len(sims) > 1 else 0.0,
                     texture=texture_score(q_fine))
        if match is not None:
            feats.update(rank=match.rank, n_matches=match.n_matches, n_query_kp=match.n_query_kp,
                         inlier_ratio=match.inliers / max(match.n_matches, 1), second_inliers=match.second_inliers)
        if match is None:
            lat, lon = prior_ll if top is None else self.tiles_ll[top[0]]
            return Fix(float(lat), float(lon), 0, -1), mode

        if self.cfg.pose == "pnp":
            # inlier matches back to raw-image pixels and to ground meters (east, north) around the crop center
            raw = cv2.transform(match.query_pts[None].astype(np.float64), cv2.invertAffineTransform(M))[0]
            half = match.crop_px / 2
            ground = np.c_[(match.crop_pts[:, 0] - half) * rr.gsd, -(match.crop_pts[:, 1] - half) * rr.gsd]
            pose = solve_pose(raw, ground, intrinsics(cam.focal_px, frame.shape[1], frame.shape[0]))
            if pose is not None:
                e, n, up = pose.center_enu
                lat = match.crop_ll[0] + n / self._mpd[0]
                lon = match.crop_ll[1] + e / self._mpd[1]
                extra = dict(height_m=float(up), off_nadir_deg=pose.off_nadir_deg, reproj_px=pose.reproj_px,
                             R_cw=pose.R_cw.tolist(), **feats)        # camera orientation, for target geolocation
                return Fix(float(lat), float(lon), min(match.inliers, pose.inliers), match.rank, extra), mode
            # PnP failed (degenerate matches): fall through to the IMU-attitude correction

        c = match.A @ np.array([qpx / 2, qpx / 2, 1.0])
        half = match.crop_px / 2                      # window/crop px -> meters around its center -> lat/lon
        plat = match.crop_ll[0] - (c[1] - half) * rr.gsd / self._mpd[0]
        plon = match.crop_ll[1] + (c[0] - half) * rr.gsd / self._mpd[1]
        dn, de = boresight_offset_ne(height_m, roll_deg, pitch_deg, heading_deg)
        return Fix(float(plat) - dn / self._mpd[0], float(plon) - de / self._mpd[1], match.inliers, match.rank,
                   dict(feats)), mode


def texture_score(img: np.ndarray) -> float:
    """Mean gradient magnitude of the (circle-masked) query: low over water / uniform fields = hard to localize."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    return float(mag[g > 0].mean()) if (g > 0).any() else 0.0


def error_m(fix: Fix, lat: float, lon: float) -> float:
    return float(haversine_m(lat, lon, fix.lat, fix.lon))
