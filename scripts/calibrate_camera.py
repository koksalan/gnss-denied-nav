"""Recover the effective focal length (px) and heading offset of a flight's camera from data.

UAV-VisLoc gives no intrinsics, and `height` is altitude above sea level (not above ground), so we fit an
*effective* focal f_eff = height / gsd per flight (valid while terrain height is roughly constant).

Heading: flights with strong matches all follow north_up_ccw = -Phi1 (+ a few degrees), so we pre-rotate
with -Phi1 and only search the scale: for each focal hypothesis we resample the photo to the satellite GSD,
match with SuperPoint+LightGlue, and keep the hypothesis with the most RANSAC inliers. The winning
similarity transform then gives the exact scale, residual rotation and center offset.
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
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.query import Camera, make_query  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402

SAT_GSD = 0.6                                                # m/px used for matching
FOCAL_HYPOTHESES = np.geomspace(2500, 25000, 13)             # effective focal lengths to try (px)
MIN_INLIERS = 40


@torch.no_grad()
def match(extractor, matcher, a, b, device):
    fa = extractor.extract(numpy_image_to_torch(a).to(device))
    fb = extractor.extract(numpy_image_to_torch(b).to(device))
    m = rbd(matcher({"image0": fa, "image1": fb}))["matches"].cpu().numpy()
    return rbd(fa)["keypoints"].cpu().numpy()[m[:, 0]], rbd(fb)["keypoints"].cpu().numpy()[m[:, 1]]


def fit(pa, pb):
    if len(pa) < 8:
        return None, 0
    A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    return (A, int(inl.sum())) if A is not None else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="03")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"outputs/calibration_{args.flight}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    fl = VisLocFlight(args.flight)
    shape = cv2.imread(str(fl.image_path(0))).shape

    rows = []
    for i in np.linspace(0, len(fl) - 1, min(args.n, len(fl))).astype(int):
        r = fl.row(i)
        img = cv2.cvtColor(cv2.imread(str(fl.image_path(i))), cv2.COLOR_BGR2RGB)
        best = (0, None, None, None)
        for f in FOCAL_HYPOTHESES:
            cam = Camera(float(f), -1, 0.0)
            diam_m = min(shape[:2]) * cam.gsd(r.height)          # ground diameter seen under this hypothesis
            qpx = int(round(diam_m / SAT_GSD))
            if not 200 <= qpx <= 1600:
                continue
            q = make_query(img, r.height, r.Phi1, cam, qpx, diam_m)
            sat = fl.sat.crop(r.lat, r.lon, 2 * diam_m, SAT_GSD)
            A, n_in = fit(*match(extractor, matcher, q, sat, device))
            if n_in > best[0]:
                best = (n_in, f, A, (qpx, sat.shape[0]))
        n_in, f, A, (qpx, spx) = best if best[1] is not None else (0, None, None, (0, 0))
        if A is None:
            print(f"{r.filename}: no match")
            continue
        s = math.hypot(A[0, 0], A[1, 0])
        rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))       # residual, image coords
        c = A @ np.array([qpx / 2, qpx / 2, 1.0])
        rows.append(dict(file=r.filename, inliers=n_in, focal_px=f / s, residual_ccw=-rot,
                         center_offset_m=float(np.hypot(*(c - spx / 2)) * SAT_GSD)))
        print(f"{r.filename}: inl={n_in:4d} f={f / s:7.0f}px rot={-rot:6.1f} off={rows[-1]['center_offset_m']:5.1f}m")

    good = [x for x in rows if x["inliers"] >= MIN_INLIERS]
    if not good:
        raise SystemExit(f"flight {args.flight}: no reliable matches")
    res = np.array([x["residual_ccw"] for x in good])
    offset = math.degrees(math.atan2(np.sin(np.radians(res)).mean(), np.cos(np.radians(res)).mean()))
    summary = dict(
        n_total=len(rows), n_good=len(good), focal_px=float(np.median([x["focal_px"] for x in good])),
        heading_sign=-1, heading_offset_deg=offset,
        heading_resid_median_deg=float(np.median(np.abs(((res - offset + 180) % 360) - 180))),
        center_offset_median_m=float(np.median([x["center_offset_m"] for x in good])), rows=rows,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    cfg = dict(
        source=f"scripts/calibrate_camera.py on UAV-VisLoc flight {args.flight} ({len(good)} of {len(rows)} matched)",
        focal_px=round(summary["focal_px"], 1), north_up_ccw_deg="heading_sign * Phi1 + heading_offset_deg",
        heading_sign=-1, heading_offset_deg=round(offset, 2),
        heading_resid_median_deg=round(summary["heading_resid_median_deg"], 2),
        gt_center_offset_median_m=round(summary["center_offset_median_m"], 1),
    )
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / f"camera_visloc{args.flight}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {cfg_path}")


if __name__ == "__main__":
    main()
