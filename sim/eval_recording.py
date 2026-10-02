"""Offline check of the localizer on a simulator recording (frames + SIMSTATE ground truth).

1. Determine the camera-mount heading convention from a few frames (which way is "up" in the image).
2. Run the localizer over the recording in time order: global search first, tracking afterwards.

    python sim/eval_recording.py --rec outputs/sim_rec/run2 --weights outputs/finetune/s1/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.geo import meters_per_degree  # noqa: E402
from gdnav.localizer import Localizer, LocalizerConfig, error_m  # noqa: E402
from gdnav.query import Camera  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402


def load(rec: Path, i: int, meta):
    return cv2.cvtColor(cv2.imread(str(rec / meta[i]["file"])), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="outputs/sim_rec/run2")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03.json")
    ap.add_argument("--weights", default="outputs/finetune/s1/best.pt")
    ap.add_argument("--patch-m", type=float, default=250.0)
    ap.add_argument("--pose", choices=["pnp", "boresight"], default="pnp")
    ap.add_argument("--pitch-bias", type=float, default=0.0, help="deg added to the recorded attitude (IMU error)")
    ap.add_argument("--roll-bias", type=float, default=0.0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    rec, w = Path(args.rec), json.loads(Path(args.world).read_text())
    meta = json.loads((rec / "meta.json").read_text())
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    bounds = (w["lat0"] - half / m_lat, w["lon0"] - half / m_lon, w["lat0"] + half / m_lat, w["lon0"] + half / m_lon)
    loc = Localizer(VisLocFlight(w["flight"]).sat, LocalizerConfig(patch_m=args.patch_m, pose=args.pose), args.weights,
                    bounds_ll=bounds)
    print(f"database: {len(loc.tiles_ll)} tiles")

    # 1) heading convention of the camera mount: try both signs, offset fixed by "image up = north at yaw 90"
    f = w["camera"]["focal_px"]
    probe = np.linspace(0, len(meta) - 1, 12).astype(int)
    best = None
    for sign in (1, -1):
        cam = Camera(f, sign, -sign * 90.0)
        ok = sum(loc.localize(load(rec, i, meta), meta[i]["rel_alt_m"], meta[i]["yaw_deg"], cam)[0].inliers
                 >= loc.cfg.min_inliers for i in probe)
        print(f"heading sign {sign:+d}: {ok}/{len(probe)} confident")
        if best is None or ok > best[0]:
            best = (ok, cam)
    cam = best[1]

    # 2) sequential run with tracking
    rows, prior = [], None
    for i, r in enumerate(meta):
        t0 = time.time()
        fix, mode = loc.localize(load(rec, i, meta), r["rel_alt_m"], r["yaw_deg"], cam, prior,
                                 roll_deg=r["roll_deg"] + args.roll_bias, pitch_deg=r["pitch_deg"] + args.pitch_bias)
        dt = time.time() - t0
        conf = fix.inliers >= loc.cfg.min_inliers
        if mode != "tilted":                       # a skipped (banked) frame keeps the previous prior
            prior = (fix.lat, fix.lon) if conf else None
        rows.append(dict(file=r["file"], mode=mode, inliers=fix.inliers, confident=conf, fix_lat=fix.lat, fix_lon=fix.lon,
                         err_m=error_m(fix, r["true_lat"], r["true_lon"]), ms=1000 * dt, yaw=r["yaw_deg"],
                         height_err_m=fix.extra.get("height_m", np.nan) - r["rel_alt_m"],
                         off_nadir_deg=fix.extra.get("off_nadir_deg", np.nan),
                         true_tilt_deg=float(np.hypot(r["roll_deg"], r["pitch_deg"]))))
    df = pd.DataFrame(rows)
    df.loc[df["mode"] == "tilted", "err_m"] = np.nan
    tag = args.tag or f"{args.pose}_p{args.pitch_bias:g}_r{args.roll_bias:g}"
    df.to_csv(rec / f"localization_{tag}.csv", index=False)
    c = df[df.confident]
    summary = dict(frames=len(df), confident=float(df.confident.mean()), median_err_m=float(c.err_m.median()),
                   p95_err_m=float(c.err_m.quantile(0.95)), max_err_m=float(c.err_m.max()),
                   track_frames=int((df["mode"] == "track").sum()), skipped_tilted=int((df["mode"] == "tilted").sum()),
                   ms_global=float(df.ms[df["mode"] == "global"].median()),
                   ms_track=float(df.ms[df["mode"] == "track"].median()) if (df["mode"] == "track").any() else None,
                   heading_sign=cam.heading_sign, heading_offset=cam.heading_offset_deg,
                   height_err_median_m=float(c.height_err_m.abs().median()),
                   tilt_err_median_deg=float((c.off_nadir_deg - c.true_tilt_deg).abs().median()))
    (rec / f"localization_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
