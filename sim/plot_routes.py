"""Figure: flown routes over the localizability map + EKF error along each route (GNSS jammed at route start).

    python sim/plot_routes.py route_west_straight route_west_rl route_west_astar --out docs/routes_west.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STYLE = {"route_west_straight": ("straight", "#d62728"), "route_west_rl": ("RL (PPO)", "#ff7f0e"),
         "route_west_astar": ("A* (localizability)", "#1f77b4")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03_real.json")
    ap.add_argument("--map", default="outputs/plan/localizability.npz")
    ap.add_argument("--out", default="docs/routes_west.png")
    ap.add_argument("--mission", default="sim/missions/west_astar.json", help="for the goal position")
    args = ap.parse_args()
    w = json.loads((ROOT / args.world).read_text())
    z = np.load(ROOT / args.map)
    g, cell = z["grid_m"], float(z["cell_m"])
    m_lat = 111132.92 - 559.82 * np.cos(2 * np.radians(w["lat0"]))
    m_lon = 111412.84 * np.cos(np.radians(w["lat0"]))
    goal = json.loads((ROOT / args.mission).read_text())["goal"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ext = [g[0] - cell / 2, g[-1] + cell / 2, g[0] - cell / 2, g[-1] + cell / 2]
    im = a1.imshow(z["success"], cmap="RdYlGn", vmin=0, vmax=1, extent=ext, origin="upper")
    fig.colorbar(im, ax=a1, fraction=0.046, label="visual fix success (localizability)")
    for tag in args.tags:
        label, color = STYLE.get(tag, (tag, None))
        d = pd.read_csv(ROOT / "outputs" / "sim_runs" / tag / "log.csv").drop_duplicates("t").sort_values("t")
        j = d[d.jammed == 1]
        to_goal = np.hypot((j.true_lat - goal[0]) * m_lat, (j.true_lon - goal[1]) * m_lon)
        arrived = np.flatnonzero(to_goal.to_numpy() < 150)
        j = j.iloc[:arrived[0] + 1] if len(arrived) else j                # route only, until arrival at the goal
        a1.plot((j.true_lon - w["lon0"]) * m_lon, (j.true_lat - w["lat0"]) * m_lat, color=color, lw=2.2, label=label)
        a2.plot(j.t - j.t.iloc[0], j.ekf_err_m, color=color, lw=1.4, label=label)
    a1.plot(0, 0, "k^", ms=9, label="home / route start")
    a1.plot((goal[1] - w["lon0"]) * m_lon, (goal[0] - w["lat0"]) * m_lat, "k*", ms=13, label="goal")
    a1.set(xlabel="east (m)", ylabel="north (m)", title="true track, GNSS jammed (SITL)", xlim=(-1800, 600), ylim=(-900, 1200))
    a1.legend(loc="lower right", fontsize=8)
    a2.set(xlabel="time since jamming (s)", ylabel="autopilot position error (m)", title="EKF error along the route (until arrival)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
