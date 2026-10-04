"""Straight route vs. localizability-aware A* (several penalty weights) [vs. an RL policy] on random missions.

    python scripts/eval_planners.py --n 100
    python scripts/eval_planners.py --n 100 --rl outputs/plan/ppo_policy.zip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.planning import BeliefModel, Grid, astar, evaluate_path, straight  # noqa: E402


def random_missions(grid: Grid, n: int, min_dist_m: float, seed: int):
    rng = np.random.default_rng(seed)
    H, W = grid.shape
    out = []
    while len(out) < n:
        a, b = tuple(rng.integers(0, [H, W])), tuple(rng.integers(0, [H, W]))
        if np.hypot(a[0] - b[0], a[1] - b[1]) * grid.cell_m >= min_dist_m:
            out.append((tuple(map(int, a)), tuple(map(int, b))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="outputs/plan/localizability.npz")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-dist-m", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--rl", default=None)
    args = ap.parse_args()

    z = np.load(ROOT / args.map)
    grid = Grid(z["success"], float(z["cell_m"]))
    model = BeliefModel()
    planners = {"straight": lambda s, g: straight(s, g)}
    for pen in (1.0, 3.0, 10.0):
        planners[f"astar_pen{pen:g}"] = (lambda p: (lambda s, g: astar(grid, s, g, p)))(pen)
    if args.rl:
        from gdnav.rl_planner import RLPlanner
        rl = RLPlanner(args.rl, grid, model)
        planners["rl_ppo"] = rl.plan
    missions = random_missions(grid, args.n, args.min_dist_m, args.seed)
    res = {k: [] for k in planners}
    for s, g in missions:
        for k, f in planners.items():
            path = f(s, g)
            r = evaluate_path(grid, path, model, n_runs=100)
            r["reached"] = tuple(path[-1]) == tuple(g)
            res[k].append(r)
    summary = {}
    base_len = np.array([r["length_m"] for r in res["straight"]], float)
    for k, rs in res.items():
        L = np.array([r["length_m"] for r in rs], float)
        summary[k] = dict(
            reached=round(float(np.mean([r["reached"] for r in rs])), 3),
            p_lost_mean=round(float(np.mean([r["p_lost"] for r in rs])), 3),
            missions_with_p_lost_gt_10pct=int(sum(r["p_lost"] > 0.1 for r in rs)),
            max_sigma_median=round(float(np.median([r["max_sigma_median"] for r in rs])), 1),
            max_sigma_p95_median=round(float(np.median([r["max_sigma_p95"] for r in rs])), 1),
            length_vs_straight=round(float(np.median(L / base_len)), 3),
            mean_success_along=round(float(np.mean([r["mean_success"] for r in rs])), 3),
        )
    out = ROOT / "outputs" / "plan"
    (out / ("planners_rl.json" if args.rl else "planners.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
