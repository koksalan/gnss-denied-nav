"""Validate PnP visual attitude + height on REAL UAV images (UAV-VisLoc), against the dataset's IMU labels.

For each sampled photo: match against the satellite crop around its labelled position (this isolates pose
estimation from retrieval), lift the inlier matches to raw-image pixels and ground meters, solve planar PnP.
Compared quantities:
  * camera tilt about the image x / y axes  vs  the labelled Omega (pitch) / Kappa (roll)
    (the dataset does not document axis/sign conventions, so the best of the 8 axis/sign mappings is reported,
     chosen on one flight and then applied unchanged to the others)
  * PnP height  vs  labelled height (flights over near-sea-level plains only: `height` is above sea level)
  * PnP camera position vs image-center position: how much the tilt correction moves the fix

    python scripts/eval_pose_real.py --flights 03,04,08,09,01,02 --n 120
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.geo import haversine_m, meters_per_degree  # noqa: E402
from gdnav.pose import intrinsics, solve_pose  # noqa: E402
from gdnav.prepared import CONFIG_ROOT  # noqa: E402
from gdnav.query import Camera, apply_circle, max_patch_m, query_warp  # noqa: E402
from gdnav.rerank import Reranker  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402


def camera_tilts(R_cw: np.ndarray) -> tuple[float, float]:
    """World 'down' expressed in the camera frame -> tilt angles (deg) about the image x and y axes."""
    d = R_cw @ np.array([0.0, 0.0, -1.0])
    return math.degrees(math.atan2(d[1], d[2])), math.degrees(math.atan2(d[0], d[2]))


@torch.no_grad()
def run_flight(f: str, n: int, rr: Reranker) -> pd.DataFrame:
    fl = VisLocFlight(f)
    cam = Camera.load(CONFIG_ROOT / f"camera_visloc{f}.json")
    m_lat, m_lon = meters_per_degree(float(fl.meta.lat.mean()))
    rows, corr = [], []
    for i in tqdm(np.linspace(0, len(fl) - 1, min(n, len(fl))).astype(int), desc=f"flight {f}"):
        r = fl.row(i)
        img = cv2.cvtColor(cv2.imread(str(fl.image_path(i))), cv2.COLOR_BGR2RGB)
        patch_m = math.floor(0.98 * max_patch_m(img.shape, r.height, cam))
        qpx = int(round(patch_m / rr.gsd))
        M = query_warp(img.shape, r.height, r.Phi1, cam, qpx, patch_m)
        q = apply_circle(cv2.warpAffine(img, M, (qpx, qpx), flags=cv2.INTER_AREA))
        match = rr.best_match(q, fl.sat, np.array([[r.lat, r.lon]]))
        if match is None or match.inliers < 40:
            continue
        raw = cv2.transform(match.query_pts[None].astype(np.float64), cv2.invertAffineTransform(M))[0]
        half = match.crop_px / 2
        ground = np.c_[(match.crop_pts[:, 0] - half) * rr.gsd, -(match.crop_pts[:, 1] - half) * rr.gsd]
        pose = solve_pose(raw, ground, intrinsics(cam.focal_px, img.shape[1], img.shape[0]))
        if pose is None:
            continue
        tx, ty = camera_tilts(pose.R_cw)
        e, n_, up = pose.center_enu
        c = match.A @ np.array([qpx / 2, qpx / 2, 1.0])                       # image-center fix (old method)
        clat, clon = rr.crop_px_to_latlon(fl.sat, match, c[0], c[1])
        plat, plon = match.crop_ll[0] + n_ / m_lat, match.crop_ll[1] + e / m_lon
        corr.append((raw.astype(np.float32), ground.astype(np.float32), match.crop_ll))
        rows.append(dict(
            flight=f, file=r.filename, inliers=pose.inliers, reproj_px=pose.reproj_px,
            tilt_x=tx, tilt_y=ty, omega=float(r.Omega), kappa=float(r.Kappa),
            label_tilt=float(math.hypot(r.Omega, r.Kappa)), vis_tilt=pose.off_nadir_deg,
            height_label=float(r.height), height_pnp=float(up),
            center_vs_pnp_m=float(haversine_m(clat, clon, plat, plon)),
            err_center_m=float(haversine_m(r.lat, r.lon, clat, clon)),
            err_pnp_m=float(haversine_m(r.lat, r.lon, plat, plon)),
        ))
    np.save(Path("outputs/pose_real") / f"corr_{f}.npy", np.array(corr, dtype=object), allow_pickle=True)
    return pd.DataFrame(rows)


def best_mapping(df: pd.DataFrame):
    """Which (tilt_x, tilt_y) axis assignment and signs best explain (omega, kappa)? Returns mapping + rmse."""
    best = None
    for swap, s1, s2 in itertools.product((False, True), (1, -1), (1, -1)):
        a, b = (df.tilt_y, df.tilt_x) if swap else (df.tilt_x, df.tilt_y)
        est_o, est_k = s1 * a, s2 * b
        rmse = math.sqrt(((est_o - df.omega) ** 2 + (est_k - df.kappa) ** 2).mean() / 2)
        if best is None or rmse < best[1]:
            best = ((swap, s1, s2), rmse)
    return best


def apply_mapping(df: pd.DataFrame, mp):
    swap, s1, s2 = mp
    a, b = (df.tilt_y, df.tilt_x) if swap else (df.tilt_x, df.tilt_y)
    return s1 * a, s2 * b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", default="03,04,08,09,01,02")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--fit-flight", default="03", help="flight used to choose the axis/sign convention")
    args = ap.parse_args()

    rr = Reranker("cuda")
    Path("outputs/pose_real").mkdir(parents=True, exist_ok=True)
    df = pd.concat([run_flight(f, args.n, rr) for f in args.flights.split(",")], ignore_index=True)
    out = Path("outputs/pose_real")
    out.mkdir(parents=True, exist_ok=True)

    mp, fit_rmse = best_mapping(df[df.flight == args.fit_flight])
    df["est_omega"], df["est_kappa"] = apply_mapping(df, mp)
    df.to_csv(out / "per_image.csv", index=False)

    summary = {"axis_mapping(swap,sign_omega,sign_kappa)": mp, "fit_flight": args.fit_flight}
    for f, g in df.groupby("flight"):
        eo, ek = g.est_omega - g.omega, g.est_kappa - g.kappa
        summary[f] = dict(
            n=len(g),
            omega_mae_deg=round(float(eo.abs().median()), 2), kappa_mae_deg=round(float(ek.abs().median()), 2),
            omega_corr=round(float(np.corrcoef(g.est_omega, g.omega)[0, 1]), 2),
            kappa_corr=round(float(np.corrcoef(g.est_kappa, g.kappa)[0, 1]), 2),
            omega_bias_deg=round(float(eo.median()), 2), kappa_bias_deg=round(float(ek.median()), 2),
            label_tilt_median_deg=round(float(g.label_tilt.median()), 2),
            height_err_median_m=round(float((g.height_pnp - g.height_label).median()), 1),
            height_err_rel_pct=round(float(((g.height_pnp - g.height_label) / g.height_label).abs().median() * 100), 2),
            pos_err_center_median_m=round(float(g.err_center_m.median()), 1),
            pos_err_pnp_median_m=round(float(g.err_pnp_m.median()), 1),
            center_vs_pnp_shift_m=round(float(g.center_vs_pnp_m.median()), 1),
            reproj_px=round(float(g.reproj_px.median()), 2),
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
