"""Localizability map: how well can the aircraft localize itself at each point of the operation area?

Synthesizes the nadir camera frame the simulator would render at a grid of positions (the Gazebo ground IS the
drone-photo orthomosaic, so a rotated/scaled crop of it is the camera view, minus lighting/perspective) and runs
the real localizer in tracking mode against the satellite map. Per cell: fix success rate and median error.

Also computes a SATELLITE-ONLY proxy (map keypoint density + texture) to see how much of localizability can be
predicted without any camera imagery (what a planner would have for an unseen area).

    python scripts/localizability_map.py --cell-m 100
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.geo import haversine_m, meters_per_degree  # noqa: E402
from gdnav.localizer import Localizer, LocalizerConfig  # noqa: E402
from gdnav.query import Camera  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402


def synth_frame(texture: np.ndarray, tex_gsd: float, e_m: float, n_m: float, heading_deg: float, alt_m: float,
                cam: Camera, wh=(1280, 854)) -> np.ndarray:
    """Nadir camera frame at (east, north) m from the texture center, for a given heading/altitude (no tilt)."""
    W, H = wh
    cam_gsd = alt_m / cam.focal_px
    s = cam_gsd / tex_gsd                                   # texture px per camera px
    th = math.radians(cam.north_up_ccw(heading_deg))        # camera image rotated by th (ccw) is north-up
    # camera px (u,v) relative to center -> texture px: rotate by +th, scale by s
    c, sn = math.cos(th), math.sin(th)
    tx = texture.shape[1] / 2 + e_m / tex_gsd
    ty = texture.shape[0] / 2 - n_m / tex_gsd
    A = np.array([[s * c, s * sn, 0.0], [-s * sn, s * c, 0.0]])  # maps (u - W/2, v - H/2) to texture offsets
    A[:, 2] = [tx, ty] - A[:, :2] @ np.array([W / 2, H / 2])
    return cv2.warpAffine(texture, A, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="sim/gz/worlds/visloc03_real.json")
    ap.add_argument("--texture", default="outputs/mosaic/visloc03_mosaic.png")
    ap.add_argument("--cell-m", type=float, default=100.0)
    ap.add_argument("--margin-m", type=float, default=250.0)
    ap.add_argument("--alt", type=float, default=400.0)
    ap.add_argument("--headings", type=int, default=2)
    ap.add_argument("--prior-sigma", type=float, default=40.0)
    ap.add_argument("--bad-m", type=float, default=30.0)
    args = ap.parse_args()

    w = json.loads((ROOT / args.world).read_text())
    tex = cv2.cvtColor(cv2.imread(str(ROOT / args.texture)), cv2.COLOR_BGR2RGB)
    tex_gsd = w["extent_m"] / tex.shape[1]
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    bounds = (w["lat0"] - half / m_lat, w["lon0"] - half / m_lon, w["lat0"] + half / m_lat, w["lon0"] + half / m_lon)
    loc = Localizer(VisLocFlight(w["flight"]).sat, LocalizerConfig(patch_m=250.0), str(ROOT / "outputs/finetune/s1/best.pt"),
                    bounds_ll=bounds)
    cam = Camera(w["camera"]["focal_px"], -1, 90.0)
    rng = np.random.default_rng(0)

    lim = half - args.margin_m
    grid = np.arange(-lim, lim + 1e-6, args.cell_m)
    n = len(grid)
    success = np.zeros((n, n)); err_med = np.full((n, n), np.nan); kp_density = np.zeros((n, n)); texture = np.zeros((n, n))
    mapf = loc.mapf
    for iy, nn in enumerate(tqdm(grid[::-1], desc="rows")):        # rows north -> south
        for ix, ee in enumerate(grid):
            lat, lon = w["lat0"] + nn / m_lat, w["lon0"] + ee / m_lon
            oks, errs = [], []
            for h in rng.uniform(0, 360, args.headings):
                frame = synth_frame(tex, tex_gsd, ee, nn, h, args.alt, cam)
                dn, de = rng.normal(0, args.prior_sigma, 2)
                fix, _ = loc.localize(frame, args.alt, h, cam, (lat + dn / m_lat, lon + de / m_lon))
                e = float(haversine_m(lat, lon, fix.lat, fix.lon)) if fix.inliers >= 50 else np.inf
                oks.append(e < args.bad_m); errs.append(e)
            success[iy, ix] = np.mean(oks)
            fin = [e for e in errs if np.isfinite(e)]
            err_med[iy, ix] = np.median(fin) if fin else np.nan
            # satellite-only proxy: map keypoints and gradient energy in the 250 m footprint
            cx, cy = mapf.ll_to_px(lat, lon)
            r = 125.0 / mapf.gsd
            k = mapf.keypoints
            kp_density[iy, ix] = int(((k[:, 0] - cx).abs() < r).logical_and((k[:, 1] - cy).abs() < r).sum())
            patch = mapf.image[int(cy - r):int(cy + r), int(cx - r):int(cx + r)]
            g = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
            texture[iy, ix] = float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())

    out = ROOT / "outputs" / "plan"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "localizability.npz", success=success, err_med=err_med, kp_density=kp_density, texture=texture,
             grid_m=grid, cell_m=args.cell_m, lat0=w["lat0"], lon0=w["lon0"])
    flat = lambda a: a.ravel()                                       # noqa: E731
    res = dict(cells=int(n * n), success_mean=round(float(success.mean()), 3),
               corr_success_vs_kp_density=round(float(np.corrcoef(flat(success), flat(kp_density))[0, 1]), 3),
               corr_success_vs_sat_texture=round(float(np.corrcoef(flat(success), flat(texture))[0, 1]), 3))
    viz = cv2.resize((success * 255).astype(np.uint8), (600, 600), interpolation=cv2.INTER_NEAREST)
    viz = cv2.applyColorMap(viz, cv2.COLORMAP_RdYlGn) if hasattr(cv2, "COLORMAP_RdYlGn") else cv2.applyColorMap(viz, cv2.COLORMAP_JET)
    bg = cv2.resize(cv2.cvtColor(tex, cv2.COLOR_RGB2BGR), (600, 600))
    m = int(round(args.margin_m / w["extent_m"] * 600))
    bg_c = cv2.resize(bg[m:600 - m, m:600 - m], (600, 600))
    cv2.imwrite(str(out / "localizability.jpg"), np.concatenate([bg_c, cv2.addWeighted(bg_c, 0.45, viz, 0.55, 0)], 1))
    (out / "localizability.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
