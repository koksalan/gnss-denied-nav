"""Georeferenced satellite map access and lat/lon <-> metric conversions."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """(meters per degree latitude, meters per degree longitude) at a given latitude (WGS84)."""
    phi = math.radians(lat_deg)
    m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    return m_lat, m_lon


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


@dataclass
class SatelliteMap:
    """A north-up GeoTIFF (EPSG:4326). Reads metric-resampled crops without loading the whole map."""

    path: str

    def __post_init__(self):
        self.ds = rasterio.open(self.path)
        if self.ds.crs is None or self.ds.crs.to_epsg() != 4326:
            raise ValueError(f"expected EPSG:4326 map, got {self.ds.crs}")
        t = self.ds.transform
        self.lon0, self.lat0 = t.c, t.f          # top-left corner
        self.dlon, self.dlat = t.a, t.e          # degrees per pixel (dlat < 0)
        self.width, self.height = self.ds.width, self.ds.height
        center_lat = self.lat0 + self.dlat * self.height / 2
        self.m_lat, self.m_lon = meters_per_degree(center_lat)
        # native ground sample distance (meters per pixel) along x and y
        self.gsd_x = abs(self.dlon) * self.m_lon
        self.gsd_y = abs(self.dlat) * self.m_lat

    @property
    def bounds_latlon(self) -> tuple[float, float, float, float]:
        """(lat_top, lon_left, lat_bottom, lon_right)"""
        return (self.lat0, self.lon0, self.lat0 + self.dlat * self.height, self.lon0 + self.dlon * self.width)

    @property
    def size_m(self) -> tuple[float, float]:
        return self.width * self.gsd_x, self.height * self.gsd_y

    def latlon_to_px(self, lat, lon):
        return (np.asarray(lon) - self.lon0) / self.dlon, (np.asarray(lat) - self.lat0) / self.dlat

    def px_to_latlon(self, x, y):
        return self.lat0 + np.asarray(y) * self.dlat, self.lon0 + np.asarray(x) * self.dlon

    def crop(self, lat: float, lon: float, size_m: float, gsd: float) -> np.ndarray:
        """Square north-up crop centered at (lat, lon), `size_m` wide, resampled to `gsd` m/px.

        Areas outside the map are zero-filled. Returns HxWx3 uint8.
        """
        out = int(round(size_m / gsd))
        cx, cy = self.latlon_to_px(lat, lon)
        w, h = size_m / self.gsd_x, size_m / self.gsd_y
        win = Window(cx - w / 2, cy - h / 2, w, h)
        arr = self.ds.read(
            indexes=[1, 2, 3], window=win, out_shape=(3, out, out),
            resampling=Resampling.average, boundless=True, fill_value=0,
        )
        return np.ascontiguousarray(arr.transpose(1, 2, 0))

    def read_overview(self, gsd: float) -> np.ndarray:
        """Whole map resampled to `gsd` m/px (HxWx3 uint8). Used for tiling."""
        out_w = int(round(self.width * self.gsd_x / gsd))
        out_h = int(round(self.height * self.gsd_y / gsd))
        arr = self.ds.read(indexes=[1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.average)
        return np.ascontiguousarray(arr.transpose(1, 2, 0))
