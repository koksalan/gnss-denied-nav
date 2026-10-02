"""Per-flight cache: north-up query patches + metric satellite overview, so training/eval never touch the TIFF.

Patch size is flight-dependent (flights fly at different heights): it is the diameter of the largest
circle inside the drone photo, so query and tiles always show the same ground area.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .query import Camera, apply_circle, make_query, max_patch_m
from .visloc import VisLocFlight

PREP_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "prepared"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


@dataclass
class PreparedFlight:
    flight: str
    queries: np.ndarray        # N x px x px x 3 uint8, north-up, circle-masked
    gt_ll: np.ndarray          # N x 2 (lat, lon)
    files: list[str]
    overview: np.ndarray       # whole map at `gsd` m/px
    gsd: float                 # m/px of queries and overview
    patch_m: float
    px: int
    bounds: tuple              # (lat_top, lon_left, lat_bottom, lon_right) of the overview

    # --- geometry on the overview -------------------------------------------------------------
    def ll_to_ov(self, lat, lon):
        lat_t, lon_l, lat_b, lon_r = self.bounds
        h, w = self.overview.shape[:2]
        return (np.asarray(lon) - lon_l) / (lon_r - lon_l) * w, (np.asarray(lat) - lat_t) / (lat_b - lat_t) * h

    def ov_to_ll(self, x, y):
        lat_t, lon_l, lat_b, lon_r = self.bounds
        h, w = self.overview.shape[:2]
        return lat_t + np.asarray(y) / h * (lat_b - lat_t), lon_l + np.asarray(x) / w * (lon_r - lon_l)

    def crop(self, cx: float, cy: float, size: int | None = None) -> np.ndarray:
        """Zero-padded square crop of the overview centered at overview pixel (cx, cy)."""
        size = size or self.px
        h, w = self.overview.shape[:2]
        x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
        out = np.zeros((size, size, 3), np.uint8)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, w), min(y0 + size, h)
        if xb > xa and yb > ya:
            out[ya - y0:yb - y0, xa - x0:xb - x0] = self.overview[ya:yb, xa:xb]
        return out

    def tile_centers(self, stride_m: float) -> np.ndarray:
        """Grid of tile centers (overview px) fully inside the map."""
        h, w = self.overview.shape[:2]
        half, step = self.px / 2, stride_m / self.gsd
        xs = np.arange(half, w - half + 1e-6, step)
        ys = np.arange(half, h - half + 1e-6, step)
        cx, cy = np.meshgrid(xs, ys)
        return np.stack([cx.ravel(), cy.ravel()], 1)

    def tiles(self, centers: np.ndarray, batch: int = 512):
        for s in range(0, len(centers), batch):
            yield [apply_circle(self.crop(x, y)) for x, y in centers[s:s + batch]]

    # --- io -----------------------------------------------------------------------------------
    @classmethod
    def load(cls, flight: str, root: Path = PREP_ROOT) -> "PreparedFlight":
        d = root / flight
        meta = json.loads((d / "meta.json").read_text())
        return cls(
            flight=flight, queries=np.load(d / "queries.npy", mmap_mode="r"),
            gt_ll=np.load(d / "gt_ll.npy"), files=meta["files"], overview=np.load(d / "overview.npy"),
            gsd=meta["gsd"], patch_m=meta["patch_m"], px=meta["px"], bounds=tuple(meta["bounds"]),
        )


def prepare_flight(flight: str, px: int = 224, root: Path = PREP_ROOT, margin: float = 0.98) -> Path:
    fl = VisLocFlight(flight)
    cam = Camera.load(CONFIG_ROOT / f"camera_visloc{flight}.json")
    first = cv2.imread(str(fl.image_path(0)))
    patch_m = float(np.floor(margin * max_patch_m(first.shape, fl.meta.height.median(), cam)))
    gsd = patch_m / px

    out = root / flight
    out.mkdir(parents=True, exist_ok=True)
    queries = np.zeros((len(fl), px, px, 3), np.uint8)
    for i in tqdm(range(len(fl)), desc=f"queries {flight}"):
        r = fl.row(i)
        img = cv2.cvtColor(cv2.imread(str(fl.image_path(i))), cv2.COLOR_BGR2RGB)
        queries[i] = make_query(img, r.height, r.Phi1, cam, px, patch_m)
    np.save(out / "queries.npy", queries)
    np.save(out / "gt_ll.npy", fl.meta[["lat", "lon"]].to_numpy())
    np.save(out / "overview.npy", fl.sat.read_overview(gsd))
    (out / "meta.json").write_text(json.dumps(dict(
        flight=flight, px=px, patch_m=patch_m, gsd=gsd, bounds=list(fl.sat.bounds_latlon),
        files=fl.meta.filename.tolist(), median_height=float(fl.meta.height.median()),
    ), indent=1))
    return out
