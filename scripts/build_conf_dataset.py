"""Dataset for the learned fix-confidence model: localizer features + exact error, from a simulator recording.

For every recorded frame (ground truth from SIMSTATE) the localizer runs in
  * tracking mode, with a prior = truth + N(0, prior_sigma) (what the onboard node sees while tracking), and
  * global mode on every `global_every`-th frame (relocalization after a loss).
The autopilot inputs are perturbed the way they degrade without GNSS (the recording was made with GNSS on):
attitude + N(0, att_sigma) per axis, barometric height + N(0, alt_sigma).

    python scripts/build_conf_dataset.py --rec outputs/sim_rec/rec_real
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.confidence import fix_features  # noqa: E402
from gdnav.geo import haversine_m, meters_per_degree  # noqa: E402
from gdnav.localizer import Localizer, LocalizerConfig  # noqa: E402
from gdnav.query import Camera  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="outputs/sim_rec/rec_real")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03_real.json")
    ap.add_argument("--weights", default="outputs/finetune/s1/best.pt")
    ap.add_argument("--patch-m", type=float, default=250.0)
    ap.add_argument("--prior-sigma", type=float, default=40.0)
    ap.add_argument("--att-sigma", type=float, default=1.5)
    ap.add_argument("--alt-sigma", type=float, default=2.0)
    ap.add_argument("--global-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rec, w = ROOT / args.rec, json.loads((ROOT / args.world).read_text())
    meta = json.loads((rec / "meta.json").read_text())
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    bounds = (w["lat0"] - half / m_lat, w["lon0"] - half / m_lon, w["lat0"] + half / m_lat, w["lon0"] + half / m_lon)
    loc = Localizer(VisLocFlight(w["flight"]).sat, LocalizerConfig(patch_m=args.patch_m, rerank_k=25),
                    str(ROOT / args.weights), bounds_ll=bounds)
    cam = Camera(w["camera"]["focal_px"], -1, 90.0)

    rows = []
    for i, r in enumerate(tqdm(meta, desc="frames")):
        if math.hypot(r["roll_deg"], r["pitch_deg"]) > loc.cfg.max_tilt_deg:
            continue
        frame = cv2.cvtColor(cv2.imread(str(rec / r["file"])), cv2.COLOR_BGR2RGB)
        roll = r["roll_deg"] + rng.normal(0, args.att_sigma)
        pitch = r["pitch_deg"] + rng.normal(0, args.att_sigma)
        alt = r["rel_alt_m"] + rng.normal(0, args.alt_sigma)
        n_off, e_off = rng.normal(0, args.prior_sigma, 2)
        priors = [("track", (r["true_lat"] + n_off / m_lat, r["true_lon"] + e_off / m_lon))]
        if i % args.global_every == 0:
            priors.append(("global", None))
        for want, prior in priors:
            fix, mode = loc.localize(frame, alt, r["yaw_deg"], cam, prior, roll_deg=roll, pitch_deg=pitch)
            if fix.inliers == 0:
                continue
            rows.append(dict(frame=i, true_lat=r["true_lat"], true_lon=r["true_lon"],
                             east_m=(r["true_lon"] - w["lon0"]) * m_lon, north_m=(r["true_lat"] - w["lat0"]) * m_lat,
                             err_m=float(haversine_m(r["true_lat"], r["true_lon"], fix.lat, fix.lon)),
                             **fix_features(fix, mode, alt, roll, pitch)))
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "conf"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"dataset_{Path(args.rec).name}.csv", index=False)
    print(f"{len(df)} samples, err>30 m: {(df.err_m > 30).mean():.3f}, median err {df.err_m.median():.1f} m")


if __name__ == "__main__":
    main()
