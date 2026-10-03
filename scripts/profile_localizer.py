"""Where does the time go in one localization? Times each stage on recorded frames (tracking mode).

    python scripts/profile_localizer.py --rec outputs/sim_rec/rec_real --n 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.embed import embed_images  # noqa: E402
from gdnav.geo import meters_per_degree  # noqa: E402
from gdnav.localizer import Localizer, LocalizerConfig  # noqa: E402
from gdnav.query import Camera, apply_circle, make_query, query_warp  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402
from lightglue.utils import numpy_image_to_torch, rbd  # noqa: E402


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="outputs/sim_rec/rec_real")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03_real.json")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    rec, w = ROOT / args.rec, json.loads((ROOT / args.world).read_text())
    meta = json.loads((rec / "meta.json").read_text())
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    bounds = (w["lat0"] - half / m_lat, w["lon0"] - half / m_lon, w["lat0"] + half / m_lat, w["lon0"] + half / m_lon)
    loc = Localizer(VisLocFlight(w["flight"]).sat, LocalizerConfig(patch_m=250.0), str(ROOT / "outputs/finetune/s1/best.pt"),
                    bounds_ll=bounds)
    cam = Camera(w["camera"]["focal_px"], -1, 90.0)
    rr = loc.reranker
    T = defaultdict(list)
    idx = [i for i, r in enumerate(meta) if abs(r["roll_deg"]) < 10][:args.n]
    for i in idx:
        r = meta[i]
        frame = cv2.cvtColor(cv2.imread(str(rec / r["file"])), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            t = time.perf_counter()
            q = make_query(frame, r["rel_alt_m"], r["yaw_deg"], cam, 224, 250.0); T["warp_coarse"].append(time.perf_counter() - t)
            sync(); t = time.perf_counter()
            e = embed_images(loc.model, [q], "cuda"); sync(); T["dino_embed"].append(time.perf_counter() - t)
            qpx = int(round(250.0 / rr.gsd))
            t = time.perf_counter()
            M = query_warp(frame.shape, r["rel_alt_m"], r["yaw_deg"], cam, qpx, 250.0)
            qf = apply_circle(cv2.warpAffine(frame, M, (qpx, qpx), flags=cv2.INTER_AREA)); T["warp_fine"].append(time.perf_counter() - t)
            sync(); t = time.perf_counter()
            fq = rr._feats(qf); sync(); T["superpoint_query"].append(time.perf_counter() - t)
            for _ in range(loc.cfg.track_k):                  # tracking mode: track_k candidates
                t = time.perf_counter()
                crop = rr.__class__.__mro__ and loc.sat.crop(r["true_lat"], r["true_lon"], rr.search_m, rr.gsd)
                T["sat_crop_tiff"].append(time.perf_counter() - t)
                sync(); t = time.perf_counter()
                fs = rr._feats(crop); sync(); T["superpoint_crop"].append(time.perf_counter() - t)
                t = time.perf_counter()
                m = rbd(rr.matcher({"image0": fq, "image1": fs}))["matches"]; sync(); T["lightglue"].append(time.perf_counter() - t)
        t = time.perf_counter()
        loc.localize(frame, r["rel_alt_m"], r["yaw_deg"], cam, (r["true_lat"], r["true_lon"]),
                     roll_deg=r["roll_deg"], pitch_deg=r["pitch_deg"]); sync()
        T["TOTAL_localize_track"].append(time.perf_counter() - t)
    res = {k: dict(ms_median=round(1000 * float(np.median(v[2:] if len(v) > 4 else v)), 1),
                   per_fix_ms=round(1000 * float(np.median(v[2:] if len(v) > 4 else v)) *
                                    (loc.cfg.track_k if k in ("sat_crop_tiff", "superpoint_crop", "lightglue") else 1), 1))
           for k, v in T.items()}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
