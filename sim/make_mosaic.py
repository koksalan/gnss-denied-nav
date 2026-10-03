"""Orthomosaic of REAL drone photos over the simulation area, to texture the Gazebo ground.

Why: if the simulated ground is the same satellite image the localizer matches against, the problem is far too
easy. Texturing it with real 2018 drone photos (haze, season, sensor differences) while the localizer still
matches against the satellite map reproduces the real domain gap without any external imagery.

Each photo is registered to the map with SuperPoint+LightGlue (around its labelled position), a homography
photo -> mosaic is fitted, and photos are blended with feathered weights. Gaps fall back to the satellite map;
the coverage mask is saved so evaluations can tell drone-textured from fallback areas.

    python sim/make_mosaic.py --world sim/gz/worlds/visloc03.json
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
from gdnav.geo import meters_per_degree  # noqa: E402
from gdnav.prepared import CONFIG_ROOT  # noqa: E402
from gdnav.query import Camera, apply_circle, max_patch_m, query_warp  # noqa: E402
from gdnav.rerank import Reranker  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402

PHOTO_SCALE = 0.25          # photos are downscaled before warping (0.115 m/px -> ~0.46 m/px)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="sim/gz/worlds/visloc03.json")
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--margin-m", type=float, default=300.0)
    ap.add_argument("--min-inliers", type=int, default=60)
    ap.add_argument("--out", default="outputs/mosaic")
    args = ap.parse_args()

    w = json.loads((ROOT / args.world).read_text())
    fl = VisLocFlight(w["flight"])
    cam = Camera.load(CONFIG_ROOT / f"camera_visloc{w['flight']}.json")
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    size = int(round(w["extent_m"] / args.gsd))
    acc = np.zeros((size, size, 3), np.float32)
    wsum = np.zeros((size, size), np.float32)
    rr = Reranker("cuda")

    e = (fl.meta.lon - w["lon0"]) * m_lon
    n = (fl.meta.lat - w["lat0"]) * m_lat
    sel = np.where((e.abs() < half + args.margin_m) & (n.abs() < half + args.margin_m))[0]
    used, skipped = 0, 0
    for i in tqdm(sel, desc="register photos"):
        r = fl.row(i)
        img = cv2.cvtColor(cv2.imread(str(fl.image_path(i))), cv2.COLOR_BGR2RGB)
        patch_m = math.floor(0.98 * max_patch_m(img.shape, r.height, cam))
        qpx = int(round(patch_m / rr.gsd))
        M = query_warp(img.shape, r.height, r.Phi1, cam, qpx, patch_m)
        q = apply_circle(cv2.warpAffine(img, M, (qpx, qpx), flags=cv2.INTER_AREA))
        match = rr.best_match(q, fl.sat, np.array([[r.lat, r.lon]]))
        if match is None or match.inliers < args.min_inliers:
            skipped += 1
            continue
        raw = cv2.transform(match.query_pts[None].astype(np.float64), cv2.invertAffineTransform(M))[0]
        # crop px -> ground meters around the crop center -> mosaic px
        ce = (match.crop_ll[1] - w["lon0"]) * m_lon
        cn = (match.crop_ll[0] - w["lat0"]) * m_lat
        hp = match.crop_px / 2
        ge = ce + (match.crop_pts[:, 0] - hp) * rr.gsd
        gn = cn - (match.crop_pts[:, 1] - hp) * rr.gsd
        mos = np.c_[ge / args.gsd + size / 2, -gn / args.gsd + size / 2]
        Hm, mask = cv2.findHomography(raw * PHOTO_SCALE, mos, cv2.RANSAC, 3.0)
        if Hm is None or mask.sum() < args.min_inliers:
            skipped += 1
            continue
        small = cv2.resize(img, None, fx=PHOTO_SCALE, fy=PHOTO_SCALE, interpolation=cv2.INTER_AREA)
        h, wd = small.shape[:2]
        # feather weight: distance to the photo border (seams fade out)
        border = np.zeros((h, wd), np.uint8)
        border[1:-1, 1:-1] = 1
        feather = cv2.distanceTransform(border, cv2.DIST_L2, 5).astype(np.float32)
        feather /= feather.max()
        corners = cv2.perspectiveTransform(np.float32([[0, 0], [wd, 0], [wd, h], [0, h]])[None], Hm)[0]
        x0, y0 = np.floor(corners.min(0)).astype(int)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size), min(y1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ Hm
        warped = cv2.warpPerspective(small, T, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        wwarp = cv2.warpPerspective(feather, T, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        acc[y0:y1, x0:x1] += warped.astype(np.float32) * wwarp[..., None]
        wsum[y0:y1, x0:x1] += wwarp
        used += 1

    covered = wsum > 1e-3
    mosaic = np.zeros((size, size, 3), np.uint8)
    mosaic[covered] = np.clip(acc[covered] / wsum[covered, None], 0, 255).astype(np.uint8)
    sat = fl.sat.crop(w["lat0"], w["lon0"], w["extent_m"], args.gsd)
    mosaic[~covered] = sat[~covered]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{Path(args.world).stem}_mosaic.png"), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / f"{Path(args.world).stem}_coverage.png"), covered.astype(np.uint8) * 255)
    preview = np.concatenate([cv2.resize(sat, (1000, 1000)), cv2.resize(mosaic, (1000, 1000))], 1)
    cv2.imwrite(str(out / "preview.jpg"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    meta = dict(photos_in_area=int(len(sel)), photos_used=used, photos_skipped=skipped,
                coverage=round(float(covered.mean()), 3), gsd=args.gsd, extent_m=w["extent_m"])
    (out / "mosaic_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
