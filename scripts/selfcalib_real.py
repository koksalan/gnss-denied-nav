"""Self-calibrate the camera from image<->map matches (ground plane = calibration target, Zhang's method).

Uses only the correspondences cached by eval_pose_real.py (no attitude labels), estimates focal, principal
point and radial distortion per flight with cv2.calibrateCamera, then re-runs PnP with the calibrated camera
and compares tilt / height / position against the labels again.

    python scripts/selfcalib_real.py --flights 03,04,09
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.geo import haversine_m, meters_per_degree  # noqa: E402
from gdnav.prepared import CONFIG_ROOT  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402
from scripts.eval_pose_real import apply_mapping, camera_tilts  # noqa: E402

W, H = 3976, 2652


def pnp(raw, ground, K, dist):
    obj = np.c_[ground, np.zeros(len(ground))].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, raw.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, raw.astype(np.float64), K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, (-R.T @ tvec).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", default="03,04,09")
    args = ap.parse_args()
    out = Path("outputs/pose_real")
    base = pd.read_csv(out / "per_image.csv", dtype={"flight": str})
    summary = json.loads((out / "summary.json").read_text())
    mp = tuple(summary["axis_mapping(swap,sign_omega,sign_kappa)"])
    res = {}
    for f in args.flights.split(","):
        corr = np.load(out / f"corr_{f}.npy", allow_pickle=True)
        g = base[base.flight == f].reset_index(drop=True)
        fl = VisLocFlight(f)
        f0 = json.loads((CONFIG_ROOT / f"camera_visloc{f}.json").read_text())["focal_px"]
        K0 = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1]], float)
        # robust per-view inliers (homography) before calibration
        objs, imgs = [], []
        for raw, ground, _ in corr:
            _, mask = cv2.findHomography(ground, raw, cv2.RANSAC, 4.0)
            keep = mask.ravel() == 1
            objs.append(np.c_[ground[keep], np.zeros(keep.sum())].astype(np.float32))
            imgs.append(raw[keep].astype(np.float32))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
                 | cv2.CALIB_FIX_K3)
        rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, (W, H), K0.copy(), None, flags=flags)
        m_lat, m_lon = meters_per_degree(float(fl.meta.lat.mean()))
        rows = []
        for (raw, ground, crop_ll), o, im, (_, r) in zip(corr, objs, imgs, g.iterrows()):
            sol = pnp(im, o[:, :2], K, dist)
            if sol is None:
                continue
            R, C = sol
            tx, ty = camera_tilts(R)
            lat, lon = crop_ll[0] + C[1] / m_lat, crop_ll[1] + C[0] / m_lon
            lab = fl.meta[fl.meta.filename == r.file].iloc[0]
            rows.append(dict(tilt_x=tx, tilt_y=ty, omega=r.omega, kappa=r.kappa, height_pnp=C[2],
                             height_label=r.height_label, err_pnp_m=float(haversine_m(lab.lat, lab.lon, lat, lon)),
                             err_center_m=r.err_center_m))
        d = pd.DataFrame(rows)
        d["est_omega"], d["est_kappa"] = apply_mapping(d, mp)
        eo, ek = d.est_omega - d.omega, d.est_kappa - d.kappa
        res[f] = dict(
            calib_rms_px=round(float(rms), 2), focal_before=round(f0, 0), focal_after=round(float(K[0, 0]), 0),
            principal_point_shift_px=[round(float(K[0, 2] - W / 2), 1), round(float(K[1, 2] - H / 2), 1)],
            k1=round(float(dist.ravel()[0]), 4), k2=round(float(dist.ravel()[1]), 4),
            omega_mae_deg=round(float(eo.abs().median()), 2), kappa_mae_deg=round(float(ek.abs().median()), 2),
            omega_corr=round(float(np.corrcoef(d.est_omega, d.omega)[0, 1]), 2),
            kappa_corr=round(float(np.corrcoef(d.est_kappa, d.kappa)[0, 1]), 2),
            kappa_bias_deg=round(float(ek.median()), 2),
            height_err_rel_pct=round(float(((d.height_pnp - d.height_label) / d.height_label).abs().median() * 100), 2),
            pos_err_center_median_m=round(float(d.err_center_m.median()), 1),
            pos_err_pnp_median_m=round(float(d.err_pnp_m.median()), 1),
        )
        print(f, json.dumps(res[f]), flush=True)
    (out / "selfcalib_summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
