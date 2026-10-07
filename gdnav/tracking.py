"""Ground-target tracking in MAP coordinates: is a detected vehicle moving, how fast, which way?

The camera moves, so pixel motion says nothing by itself. Every detection is first geolocated (gdnav/geolocate.py,
camera pose from the same frame), then associated with existing tracks on the ground: nearest predicted position
within a gate. Velocity is a least-squares line fit over the last few seconds of a track's positions, which averages
out the ~2 m geolocation noise; a track is "moving" above `moving_mps`.

Parking lots are the hard case: cars 3 m apart, and a track that jumps to the neighbouring car looks like a car
that moved 3 m. Two defences: optimal one-to-one association (Hungarian), and a consistency test - real motion is
a straight line in time (high R^2 of the position-vs-time fit), identity swaps are a zig-zag.

Common-mode correction (optional): all vehicles in one frame are geolocated with the SAME camera pose, so a pose error shifts
them all together and makes parked cars look like they move. Most vehicles are parked, so the median offset of the
frame's detections from their (static) tracks' positions is that frame's pose error; it is subtracted first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    cls: int
    hist: list = field(default_factory=list)       # (t, east, north)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    r2: float = 0.0                                  # how well a constant velocity explains the positions

    def predict(self, t: float) -> np.ndarray:
        tl, e, n = self.hist[-1]
        return np.array([e, n]) + self.vel * (t - tl)

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.vel))

    @property
    def heading_deg(self) -> float:
        """Direction of travel, degrees clockwise from north."""
        return float(np.degrees(np.arctan2(self.vel[0], self.vel[1])) % 360)


class TargetTracker:
    def __init__(self, gate_m: float = 10.0, window_s: float = 3.0, min_span_s: float = 1.2,
                 max_age_s: float = 3.0, moving_mps: float = 2.5, common_mode: bool = False, min_common: int = 4,
                 hungarian: bool = False, min_r2: float = 0.9):
        self.gate_m, self.window_s, self.min_span_s = gate_m, window_s, min_span_s
        self.max_age_s, self.moving_mps = max_age_s, moving_mps
        self.common_mode, self.min_common = common_mode, min_common
        self.hungarian, self.min_r2 = hungarian, min_r2
        self.last_bias = np.zeros(2)
        self.tracks: list[Track] = []
        self._next = 0

    def update(self, t: float, pts_en: np.ndarray, cls: np.ndarray) -> list[Track]:
        """pts_en: N x 2 ground positions (east, north) at time t. Returns the track of each detection."""
        self.tracks = [k for k in self.tracks if t - k.hist[-1][0] <= self.max_age_s]
        out: list[Track | None] = [None] * len(pts_en)
        pts_en = np.asarray(pts_en, float)
        if self.tracks and len(pts_en):
            if self.common_mode:
                # this frame's pose error: median offset of detections from nearby STATIC tracks (parked cars)
                ref = [k for k in self.tracks if len(k.hist) >= 3 and self.is_moving(k) is False]
                if len(ref) >= self.min_common:
                    pos = np.array([np.median(np.array(k.hist)[:, 1:], axis=0) for k in ref])
                    d = np.linalg.norm(pts_en[:, None] - pos[None], axis=2)
                    j = d.argmin(1)
                    near = d[np.arange(len(pts_en)), j] < self.gate_m
                    if near.sum() >= self.min_common:
                        self.last_bias = np.median(pts_en[near] - pos[j[near]], axis=0)
                        pts_en = pts_en - self.last_bias
            pred = np.array([k.predict(t) for k in self.tracks])
            d = np.linalg.norm(pts_en[:, None] - pred[None], axis=2)
            if self.hungarian:                              # optimal one-to-one assignment within the gate
                big = 1e6
                ii, jj = linear_sum_assignment(np.where(d <= self.gate_m, d, big))
                for i, j in zip(ii, jj):
                    if d[i, j] <= self.gate_m:
                        out[i] = self.tracks[j]
            else:
                for _ in range(min(d.shape)):               # greedy nearest-neighbour assignment
                    i, j = np.unravel_index(np.argmin(d), d.shape)
                    if d[i, j] > self.gate_m:
                        break
                    out[i] = self.tracks[j]
                    d[i, :], d[:, j] = np.inf, np.inf
        for i, p in enumerate(pts_en):
            k = out[i]
            if k is None:
                k = Track(self._next, int(cls[i]))
                self._next += 1
                self.tracks.append(k)
                out[i] = k
            k.hist.append((t, float(p[0]), float(p[1])))
            k.hist = [h for h in k.hist if t - h[0] <= self.window_s]
            h = np.array(k.hist)
            if h[-1, 0] - h[0, 0] >= self.min_span_s and len(h) >= 3:
                A = np.c_[h[:, 0] - h[0, 0], np.ones(len(h))]
                coef = np.linalg.lstsq(A, h[:, 1:], rcond=None)[0]
                k.vel = coef[0]
                ss_res = float(((h[:, 1:] - A @ coef) ** 2).sum())
                ss_tot = float(((h[:, 1:] - h[:, 1:].mean(0)) ** 2).sum())
                k.r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        return out

    def is_moving(self, k: Track) -> bool | None:
        """None until the track is long enough to tell."""
        if len(k.hist) < 3 or k.hist[-1][0] - k.hist[0][0] < self.min_span_s:
            return None
        return k.speed > self.moving_mps and k.r2 >= self.min_r2
