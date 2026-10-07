"""Score a planned-route flight: from GNSS jamming (route start) until arrival over the goal.

    python sim/analyze_route.py route_west_straight route_west_astar --mission-prefix sim/missions/west
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def analyze(tag: str, goal: tuple[float, float], arrive_m: float) -> dict:
    d = pd.read_csv(ROOT / "outputs" / "sim_runs" / tag / "log.csv").drop_duplicates("t").sort_values("t")
    d["fix_err_m"] = pd.to_numeric(d.fix_err_m, errors="coerce")
    j = d[d.jammed == 1].copy()
    m_lat = 111132.92 - 559.82 * np.cos(2 * np.radians(goal[0]))
    m_lon = 111412.84 * np.cos(np.radians(goal[0]))
    j["to_goal_m"] = np.hypot((j.true_lat - goal[0]) * m_lat, (j.true_lon - goal[1]) * m_lon)
    arr = j[j.to_goal_m < arrive_m]
    t_arr = arr.t.iloc[0] if len(arr) else j.t.iloc[-1]
    seg = j[j.t <= t_arr]
    sent_t = seg.t[seg.sent == 1].to_numpy()                    # map fixes (sent == 2: visual odometry)
    gaps = np.diff(np.r_[seg.t.iloc[0], sent_t, t_arr]) if len(seg) else np.array([np.nan])
    any_t = seg.t[seg.sent > 0].to_numpy()
    gaps_any = np.diff(np.r_[seg.t.iloc[0], any_t, t_arr]) if len(seg) else np.array([np.nan])
    return dict(arrived=bool(len(arr)), route_s=round(float(t_arr - seg.t.iloc[0]), 0),
                ekf_err_median_m=round(float(seg.ekf_err_m.median()), 1),
                ekf_err_max_m=round(float(seg.ekf_err_m.max()), 1),
                ekf_err_at_arrival_m=round(float(seg.ekf_err_m.iloc[-1]), 1),
                longest_fix_gap_s=round(float(np.max(gaps)), 1),
                longest_update_gap_s=round(float(np.max(gaps_any)), 1),   # map fix or odometry
                odometry_updates=int((seg.sent == 2).sum()),
                fixes_per_s=round(len(sent_t) / max(float(t_arr - seg.t.iloc[0]), 1.0), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--mission-prefix", default="sim/missions/west")
    ap.add_argument("--arrive-m", type=float, default=150.0)
    args = ap.parse_args()
    goal = tuple(json.loads((ROOT / f"{args.mission_prefix}_astar.json").read_text())["goal"])
    res = {t: analyze(t, goal, args.arrive_m) for t in args.tags}
    print(json.dumps(res, indent=2))
    (ROOT / "outputs" / "sim_runs" / f"routes_{Path(args.mission_prefix).name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
