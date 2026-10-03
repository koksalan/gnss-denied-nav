"""Split a run's errors by ground texture under the aircraft: real drone photos vs satellite fallback.

The real-imagery world (sim/make_mosaic.py) is only partly covered by drone photos; elsewhere the ground is
the reference satellite map itself (the easy case). Comparing the two inside ONE flight shows the cost of
the real domain gap.

    python sim/analyze_coverage.py real_pnp --world sim/gz/worlds/visloc03_real.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03_real.json")
    ap.add_argument("--coverage", default="outputs/mosaic/visloc03_coverage.png")
    args = ap.parse_args()

    w = json.loads((ROOT / args.world).read_text())
    cov = cv2.imread(str(ROOT / args.coverage), cv2.IMREAD_GRAYSCALE) > 0
    size = cov.shape[0]
    m_lat = 111132.92 - 559.82 * np.cos(2 * np.radians(w["lat0"]))
    m_lon = 111412.84 * np.cos(np.radians(w["lat0"]))
    d = pd.read_csv(ROOT / "outputs" / "sim_runs" / args.tag / "log.csv").drop_duplicates("t")
    d["fix_err_m"] = pd.to_numeric(d.fix_err_m, errors="coerce")
    x = ((d.true_lon - w["lon0"]) * m_lon / w["extent_m"] + 0.5) * size
    y = (0.5 - (d.true_lat - w["lat0"]) * m_lat / w["extent_m"]) * size
    xi, yi = np.clip(x.astype(int), 0, size - 1), np.clip(y.astype(int), 0, size - 1)
    # fraction of drone-photo texture within the camera's ~260 m footprint (radius 130 m)
    r = int(130 / w["extent_m"] * size)
    frac = [cov[max(b - r, 0):b + r, max(a - r, 0):a + r].mean() for a, b in zip(xi, yi)]
    d["drone_texture"] = np.array(frac) > 0.8
    d["sat_texture"] = np.array(frac) < 0.2
    j = d[(d.jammed == 1) & d["mode"].isin(["track", "global"])]
    out = {}
    for name, g in (("drone photos (real domain gap)", j[j.drone_texture]), ("satellite fallback (easy)", j[j.sat_texture])):
        att = g[g["mode"].isin(["track", "global"])]
        out[name] = dict(
            frames=len(att),
            fix_rate=round(float(att.sent.mean()), 3) if len(att) else None,
            fix_err_median_m=round(float(att.fix_err_m[att.sent == 1].median()), 1) if att.sent.any() else None,
            fix_err_p95_m=round(float(att.fix_err_m[att.sent == 1].quantile(0.95)), 1) if att.sent.any() else None,
            ekf_err_median_m=round(float(g.ekf_err_m.median()), 1) if len(g) else None,
        )
    out["all_jammed"] = dict(ekf_err_median_m=round(float(d[d.jammed == 1].ekf_err_m.median()), 1),
                             ekf_err_max_m=round(float(d[d.jammed == 1].ekf_err_m.max()), 1))
    print(json.dumps(out, indent=2))
    (ROOT / "outputs" / "sim_runs" / args.tag / "coverage_split.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
