"""Localizability-aware route planning on a grid (A*) and a belief model to score routes.

Map: per-cell probability that a visual fix succeeds (scripts/localizability_map.py, 100 m cells).
Belief model (shared with the RL environment): failures are tied to PLACES, not to seconds -- an unrecognizable
river or field fails every frame. So in each simulated flight every cell is drawn once as "works" (with its success
probability) or "doesn't"; while over a working cell the aircraft gets fixes, elsewhere its position uncertainty
grows (dead-reckoning drift that gets worse the longer it lasts), and a fix resets it. Routes are judged by how large
the uncertainty gets and how often it exceeds the tracking window ("lost").
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

SQ2 = math.sqrt(2.0)
NEIGHBORS = [(-1, -1, SQ2), (-1, 0, 1.0), (-1, 1, SQ2), (0, -1, 1.0), (0, 1, 1.0), (1, -1, SQ2), (1, 0, 1.0), (1, 1, SQ2)]


@dataclass
class Grid:
    success: np.ndarray        # rows north -> south, cols west -> east
    cell_m: float

    @property
    def shape(self):
        return self.success.shape


def astar(grid: Grid, start: tuple[int, int], goal: tuple[int, int], penalty: float) -> list[tuple[int, int]]:
    """Min sum over steps of length * (1 + penalty * (1 - success)). penalty = 0 -> shortest (straight-ish) path."""
    cost = 1.0 + penalty * (1.0 - grid.success)
    H, W = grid.shape
    h = lambda r, c: math.hypot(r - goal[0], c - goal[1])          # admissible: per-step cost >= length  # noqa: E731
    g = {start: 0.0}
    prev = {}
    pq = [(h(*start), 0.0, start)]
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if cur == goal:
            break
        if gc > g.get(cur, math.inf):
            continue
        for dr, dc, ln in NEIGHBORS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            ng = gc + ln * 0.5 * (cost[cur] + cost[nr, nc])
            if ng < g.get((nr, nc), math.inf):
                g[(nr, nc)] = ng
                prev[(nr, nc)] = cur
                heapq.heappush(pq, (ng + h(nr, nc), ng, (nr, nc)))
    path, node = [goal], goal
    while node != start:
        node = prev[node]
        path.append(node)
    return path[::-1]


def straight(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    n = int(max(abs(goal[0] - start[0]), abs(goal[1] - start[1])))
    return [(int(round(start[0] + (goal[0] - start[0]) * t / n)), int(round(start[1] + (goal[1] - start[1]) * t / n)))
            for t in range(n + 1)]


def path_length_m(path, cell_m: float) -> float:
    return float(sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:])) * cell_m)


def densify(path, cell_m: float, step_m: float) -> np.ndarray:
    """Cells visited every `step_m` meters along the polyline (for the belief simulation)."""
    pts = np.array(path, float)
    seg = np.hypot(*np.diff(pts, axis=0).T) * cell_m
    out = []
    for (a, b), L in zip(zip(pts, pts[1:]), seg):
        k = max(1, int(L // step_m))
        out += [a + (b - a) * t / k for t in range(k)]
    out.append(pts[-1])
    return np.round(np.array(out)).astype(int)


@dataclass
class BeliefModel:
    speed_mps: float = 20.0
    sigma_fix_m: float = 10.0        # uncertainty right after a good fix
    drift_mps: float = 0.6           # dead-reckoning drift rate ...
    drift_growth: float = 0.02       # ... that grows the longer it lasts (airspeed/wind error integrates)
    lost_m: float = 150.0            # beyond this the tracking window no longer contains the true position

    def step(self, sigma: float, gap_s: float, cell_works: bool) -> tuple[float, float]:
        """One second of flight over a cell. Returns (new sigma, new gap)."""
        if sigma < self.lost_m and cell_works:
            return self.sigma_fix_m, 0.0
        gap_s += 1.0
        return sigma + self.drift_mps + self.drift_growth * gap_s, gap_s


def draw_working_cells(grid: Grid, rng, corr_m: float = 300.0) -> np.ndarray:
    """One flight's realization: which cells actually give fixes. Spatially correlated (a whole river or field fails
    together, not cell by cell): a smoothed Gaussian field turned into uniforms, compared with the success map."""
    z = gaussian_filter(rng.standard_normal(grid.shape), sigma=corr_m / grid.cell_m, mode="reflect")
    u = ndtr(z / (z.std() + 1e-9))
    return u < grid.success


def evaluate_path(grid: Grid, path, model: BeliefModel, n_runs: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cells = densify(path, grid.cell_m, model.speed_mps)          # one cell per second of flight
    p = grid.success[cells[:, 0], cells[:, 1]]
    max_sig, lost = [], 0
    for _ in range(n_runs):
        works = draw_working_cells(grid, rng)                      # one realization per flight (place-tied)
        s, gap, mx = model.sigma_fix_m, 0.0, 0.0
        for r, c in cells:
            s, gap = model.step(s, gap, bool(works[r, c]))
            mx = max(mx, s)
        max_sig.append(mx)
        lost += mx >= model.lost_m
    return dict(length_m=round(path_length_m(path, grid.cell_m)), flight_s=len(p),
                max_sigma_median=round(float(np.median(max_sig)), 1), max_sigma_p95=round(float(np.quantile(max_sig, 0.95)), 1),
                p_lost=round(lost / n_runs, 3), mean_success=round(float(p.mean()), 3))
