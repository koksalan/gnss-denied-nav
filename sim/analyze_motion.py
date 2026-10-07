"""Moving-target estimation: is a detected vehicle moving, and how fast / which way? (GNSS jammed)

Uses detections.csv of a run with --detector on a world with moving vehicles (sim/make_targets.py --n-moving).
Only true vehicle detections (true-pose projection within `match_m` of a vehicle) whose track is old enough to
have a velocity are scored.

    python sim/analyze_motion.py traffic_visual
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--match-m", type=float, default=10.0)
    args = ap.parse_args()
    d = pd.read_csv(ROOT / "outputs" / "sim_runs" / args.tag / "detections.csv")
    d = d[(d.dist_truepose_to_vehicle_m < args.match_m) & (d.jammed == 1)]
    d = d[d.est_moving.notna() & (d.est_moving.astype(str) != "")]
    d["est_moving"] = d.est_moving.astype(float).astype(int)
    truth = d.true_speed > 0
    est = d.est_moving == 1
    tp, fp, fn = int((truth & est).sum()), int((~truth & est).sum()), int((truth & ~est).sum())
    mv = d[truth & est]
    herr = ((mv.est_heading - mv.true_heading + 180) % 360 - 180).abs()
    res = dict(
        scored_detections=int(len(d)), moving_detections=int(truth.sum()),
        moving_vehicles_seen=int(d[truth].true_vehicle_id.nunique()),
        moving_precision=round(tp / max(tp + fp, 1), 3), moving_recall=round(tp / max(tp + fn, 1), 3),
        static_false_moving_rate=round(fp / max(int((~truth).sum()), 1), 4),
        speed_err_median_mps=round(float((mv.est_speed - mv.true_speed).abs().median()), 2),
        speed_err_p90_mps=round(float((mv.est_speed - mv.true_speed).abs().quantile(0.9)), 2),
        heading_err_median_deg=round(float(herr.median()), 1), heading_err_p90_deg=round(float(herr.quantile(0.9)), 1),
        static_speed_median_mps=round(float(d[~truth].est_speed.median()), 2),
    )
    print(json.dumps(res, indent=2))
    (ROOT / "outputs" / "sim_runs" / args.tag / "motion_summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
