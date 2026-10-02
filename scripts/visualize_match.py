"""Figure for the README: drone photo (north-up) | satellite crop, with LightGlue inlier matches drawn."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.prepared import CONFIG_ROOT, PreparedFlight  # noqa: E402
from gdnav.query import Camera, make_query  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402

GSD = 0.5


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="03")
    ap.add_argument("--idx", type=int, nargs="+", default=[100, 400])
    ap.add_argument("--out", default="docs/match_example.jpg")
    args = ap.parse_args()

    device = "cuda"
    ext = SuperPoint(max_num_keypoints=2048).eval().to(device)
    mat = LightGlue(features="superpoint").eval().to(device)
    fl, pf = VisLocFlight(args.flight), PreparedFlight.load(args.flight)
    cam = Camera.load(CONFIG_ROOT / f"camera_visloc{args.flight}.json")
    qpx, spx = int(pf.patch_m / GSD), int(1.5 * pf.patch_m / GSD)
    rows = []
    for i in args.idx:
        r = fl.row(i)
        img = cv2.cvtColor(cv2.imread(str(fl.image_path(i))), cv2.COLOR_BGR2RGB)
        q = make_query(img, r.height, r.Phi1, cam, qpx, pf.patch_m)
        s = fl.sat.crop(r.lat, r.lon, 1.5 * pf.patch_m, GSD)
        fq, fs = ext.extract(numpy_image_to_torch(q).to(device)), ext.extract(numpy_image_to_torch(s).to(device))
        m = rbd(mat({"image0": fq, "image1": fs}))["matches"].cpu().numpy()
        pa, pb = rbd(fq)["keypoints"].cpu().numpy()[m[:, 0]], rbd(fs)["keypoints"].cpu().numpy()[m[:, 1]]
        _, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=6.0)
        pad = np.zeros((spx, qpx, 3), np.uint8)
        pad[(spx - qpx) // 2:(spx - qpx) // 2 + qpx] = q
        canvas = np.concatenate([pad, np.full((spx, 20, 3), 255, np.uint8), s], 1)
        dy, dx = (spx - qpx) // 2, qpx + 20
        rng = np.random.default_rng(0)
        for (a, b) in rng.permutation(np.stack([pa, pb], 1)[inl.ravel() == 1])[:120]:
            cv2.line(canvas, (int(a[0]), int(a[1] + dy)), (int(b[0] + dx), int(b[1])), (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{int(inl.sum())} inliers", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        rows.append(canvas)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = np.concatenate(rows, 0)
    out = cv2.resize(out, None, fx=1100 / out.shape[1], fy=1100 / out.shape[1], interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
