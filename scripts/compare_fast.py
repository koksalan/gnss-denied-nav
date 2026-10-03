"""Accuracy + latency: original localizer vs. fast path (precomputed map features), same frames and priors.

    python scripts/compare_fast.py --rec outputs/sim_rec/rec_real --n 150
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.geo import haversine_m, meters_per_degree  # noqa: E402
from gdnav.localizer import Localizer, LocalizerConfig  # noqa: E402
from gdnav.query import Camera  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="outputs/sim_rec/rec_real")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03_real.json")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--prior-sigma", type=float, default=40.0)
    args = ap.parse_args()
    rec, w = ROOT / args.rec, json.loads((ROOT / args.world).read_text())
    meta = json.loads((rec / "meta.json").read_text())
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    bounds = (w["lat0"] - half / m_lat, w["lon0"] - half / m_lon, w["lat0"] + half / m_lat, w["lon0"] + half / m_lon)
    sat = VisLocFlight(w["flight"]).sat
    wts = str(ROOT / "outputs/finetune/s1/best.pt")
    t0 = time.time()
    fast = Localizer(sat, LocalizerConfig(patch_m=250.0, fast=True), wts, bounds_ll=bounds)
    print(f"fast localizer ready (map prep incl.) in {time.time() - t0:.0f}s, "
          f"{fast.mapf.keypoints.shape[0]} map keypoints, map {fast.mapf.image.shape[:2]}")
    slow = Localizer(sat, LocalizerConfig(patch_m=250.0, fast=False), wts, bounds_ll=bounds)
    cam = Camera(w["camera"]["focal_px"], -1, 90.0)
    rng = np.random.default_rng(0)
    idx = [i for i, r in enumerate(meta) if math.hypot(r["roll_deg"], r["pitch_deg"]) < 25][:: max(1, len(meta) // args.n)]
    res = {k: {"err": [], "ms": [], "inl": []} for k in ("slow_track", "fast_track", "slow_global", "fast_global")}
    for j, i in enumerate(idx):
        r = meta[i]
        frame = cv2.cvtColor(cv2.imread(str(rec / r["file"])), cv2.COLOR_BGR2RGB)
        dn, de = rng.normal(0, args.prior_sigma, 2)
        prior = (r["true_lat"] + dn / m_lat, r["true_lon"] + de / m_lon)
        runs = [("track", prior)] + ([("global", None)] if j % 5 == 0 else [])
        for kind, pr in runs:
            for name, loc in (("slow", slow), ("fast", fast)):
                torch.cuda.synchronize(); t = time.perf_counter()
                fix, _ = loc.localize(frame, r["rel_alt_m"], r["yaw_deg"], cam, pr, r["roll_deg"], r["pitch_deg"])
                torch.cuda.synchronize()
                d = res[f"{name}_{kind}"]
                d["ms"].append(1000 * (time.perf_counter() - t))
                d["inl"].append(fix.inliers)
                d["err"].append(float(haversine_m(r["true_lat"], r["true_lon"], fix.lat, fix.lon)) if fix.inliers >= 25 else np.nan)
    out = {}
    for k, d in res.items():
        e = np.array(d["err"], float)
        out[k] = dict(n=len(e), confident=round(float(np.isfinite(e).mean()), 3),
                      err_median=round(float(np.nanmedian(e)), 1), err_p95=round(float(np.nanpercentile(e, 95)), 1),
                      bad_gt30=int(np.nansum(e > 30)), inliers_median=int(np.median(d["inl"])),
                      ms_median=round(float(np.median(d["ms"][3:])), 1), ms_p95=round(float(np.percentile(d["ms"][3:], 95)), 1))
    print(json.dumps(out, indent=2))
    (ROOT / "outputs" / "compare_fast.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
