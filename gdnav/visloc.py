"""UAV-VisLoc dataset loader (https://github.com/IntelliSensing/UAV-VisLoc)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .geo import SatelliteMap

DEFAULT_ROOT = os.environ.get(
    "VISLOC_ROOT", str(Path(__file__).resolve().parents[2] / "data" / "UAV-VisLoc")
)


class VisLocFlight:
    """One flight: drone images + their GT metadata + the satellite map that covers them."""

    def __init__(self, flight: str, root: str = DEFAULT_ROOT):
        self.flight = flight
        self.dir = Path(root) / flight
        self.meta = pd.read_csv(self.dir / f"{flight}.csv")
        self.sat = SatelliteMap(str(self.dir / f"satellite{flight}.tif"))

    def __len__(self):
        return len(self.meta)

    def image_path(self, i: int) -> Path:
        return self.dir / "drone" / self.meta.filename.iloc[i]

    def row(self, i: int) -> pd.Series:
        return self.meta.iloc[i]
