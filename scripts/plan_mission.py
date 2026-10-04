"""Plan a mission from the world center (home) to a goal cell: straight route and localizability-aware A*.

Writes sim/missions/<name>_{straight,astar}.json (waypoints lat/lon) and a figure over the localizability map.

    python scripts/plan_mission.py --goal 12 2 --name west
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.geo import meters_per_degree  # noqa: E402
from gdnav.planning import BeliefModel, Grid, astar, evaluate_path, straight  # noqa: E402


def simplify(path, eps_cells: float = 1.0):
    pts = np.array(path, np.float32)[:, None, ::-1]                 # (x=col, y=row)
    s = cv2.approxPolyDP(pts, eps_cells, False)[:, 0, ::-1]
    return [tuple(map(int, p)) for p in s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="outputs/plan/localizability.npz")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03_real.json")
    ap.add_argument("--goal", type=int, nargs=2, required=True, help="row col on the localizability grid")
    ap.add_argument("--penalty", type=float, default=3.0)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    z = np.load(ROOT / args.map)
    grid, g = Grid(z["success"], float(z["cell_m"])), z["grid_m"]
    n = len(g)
    w = json.loads((ROOT / args.world).read_text())
    m_lat, m_lon = meters_per_degree(w["lat0"])
    home, goal = (n // 2, n // 2), tuple(args.goal)
    to_ll = lambda rc: (w["lat0"] + g[n - 1 - rc[0]] / m_lat, w["lon0"] + g[rc[1]] / m_lon)  # noqa: E731
    out_dir = ROOT / "sim" / "missions"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz = cv2.applyColorMap((255 - grid.success * 255).astype(np.uint8), cv2.COLORMAP_JET)  # red = hard, blue = easy
    viz = cv2.resize(viz, (n * 16, n * 16), interpolation=cv2.INTER_NEAREST)
    summary = {}
    for name, path, color in (("straight", straight(home, goal), (255, 255, 255)),
                              ("astar", astar(grid, home, goal, args.penalty), (0, 0, 0))):
        ev = evaluate_path(grid, path, BeliefModel(), n_runs=200)
        wps = simplify(path)
        (out_dir / f"{args.name}_{name}.json").write_text(json.dumps(dict(
            home=to_ll(home), goal=to_ll(goal), waypoints=[to_ll(p) for p in wps[1:]], alt_m=400.0,
            planner=name, penalty=args.penalty if name == "astar" else 0.0, predicted=ev), indent=2))
        pts = np.array([[c * 16 + 8, r * 16 + 8] for r, c in path], np.int32)
        cv2.polylines(viz, [pts], False, color, 3, cv2.LINE_AA)
        summary[name] = ev
    cv2.imwrite(str(ROOT / "docs" / f"plan_{args.name}.png"), viz)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
