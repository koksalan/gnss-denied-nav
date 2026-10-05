"""GNSS-free target geolocation: error of detected vehicles' coordinates, per camera-pose source.

A detection counts as a true vehicle when its TRUE-pose ground projection lands within `match_m` of a vehicle
(the rest are false positives). For those, the error of the coordinates computed with the autopilot (EKF) pose
and with the visual PnP pose is reported, before and after GNSS jamming.

    python sim/analyze_geoloc.py geo_visual geo_control
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return {}
    return dict(n=int(len(x)), median_m=round(float(x.median()), 1), p90_m=round(float(x.quantile(0.9)), 1),
                within_20m=round(float((x < 20).mean()), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--match-m", type=float, default=10.0)
    args = ap.parse_args()
    res = {}
    for tag in args.tags:
        d = pd.read_csv(ROOT / "outputs" / "sim_runs" / tag / "detections.csv")
        true_det = d[d.dist_truepose_to_vehicle_m < args.match_m]
        j = true_det[true_det.jammed == 1]
        res[tag] = dict(
            detections=int(len(d)), false_positive_rate=round(float(1 - len(true_det) / max(len(d), 1)), 3),
            unique_vehicles_found=int(true_det.true_vehicle_id.nunique()),
            jammed=dict(ekf_pose=stats(j.err_ekf_pose_m), visual_pnp_pose=stats(j.err_pnp_pose_m)),
            before_jam=dict(ekf_pose=stats(true_det[true_det.jammed == 0].err_ekf_pose_m)),
        )
    print(json.dumps(res, indent=2))
    (ROOT / "outputs" / "sim_runs" / "geoloc_summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
