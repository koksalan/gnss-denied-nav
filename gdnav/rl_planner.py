"""RL route planner: a PPO policy that moves cell by cell and sees its own position uncertainty.

Same world model as gdnav/planning.py (place-tied, spatially correlated fix failures; uncertainty grows without
fixes). Difference to A*: the policy observes the CURRENT uncertainty, so it can cut across a hard area when it has
just had a fix and detour when it is already uncertain -- A* only sees a static cost map.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from .planning import NEIGHBORS, BeliefModel, Grid, draw_working_cells

WIN = 3                                     # local map window radius -> 7 x 7 cells


class RoutePlanningEnv(gym.Env):
    def __init__(self, grid: Grid, model: BeliefModel, min_dist_m: float = 1500.0, sigma_weight: float = 1.0,
                 max_steps: int = 120, seed: int | None = None):
        self.grid, self.model = grid, model
        self.min_dist_m, self.sigma_weight, self.max_steps = min_dist_m, sigma_weight, max_steps
        self.rng = np.random.default_rng(seed)
        H, W = grid.shape
        self.padded = np.pad(grid.success, WIN, constant_values=0.5)
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(-5, 5, (5 + (2 * WIN + 1) ** 2,), np.float32)

    def _obs(self):
        r, c = self.pos
        dr, dc = self.goal[0] - r, self.goal[1] - c
        dist = math.hypot(dr, dc)
        win = self.padded[r:r + 2 * WIN + 1, c:c + 2 * WIN + 1].ravel()
        return np.concatenate([[dr / max(dist, 1e-6), dc / max(dist, 1e-6), dist * self.grid.cell_m / 3000.0,
                                self.sigma / self.model.lost_m, self.gap / 100.0], win]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        H, W = self.grid.shape
        while True:
            a, b = tuple(self.rng.integers(0, [H, W])), tuple(self.rng.integers(0, [H, W]))
            if math.hypot(a[0] - b[0], a[1] - b[1]) * self.grid.cell_m >= self.min_dist_m:
                break
        if options and "start" in options:
            a, b = tuple(options["start"]), tuple(options["goal"])
        self.pos, self.goal = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
        self.works = draw_working_cells(self.grid, self.rng) if not (options and options.get("expected")) \
            else self.grid.success >= 0.5
        self.sigma, self.gap, self.steps = self.model.sigma_fix_m, 0.0, 0
        self.path = [self.pos]
        return self._obs(), {}

    def step(self, action):
        dr, dc, ln = NEIGHBORS[int(action)]
        H, W = self.grid.shape
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1
        if not (0 <= nr < H and 0 <= nc < W):
            return self._obs(), -1.0, False, self.steps >= self.max_steps, {}      # bumped the map edge
        d_before = math.hypot(self.goal[0] - self.pos[0], self.goal[1] - self.pos[1])
        self.pos = (nr, nc)
        self.path.append(self.pos)
        seconds = int(round(ln * self.grid.cell_m / self.model.speed_mps))
        sigma_sum = 0.0
        for _ in range(seconds):
            self.sigma, self.gap = self.model.step(self.sigma, self.gap, bool(self.works[nr, nc]))
            sigma_sum += self.sigma
        d_after = math.hypot(self.goal[0] - nr, self.goal[1] - nc)
        # time cost + uncertainty exposure, shaped by progress toward the goal
        reward = -ln * 0.1 - self.sigma_weight * sigma_sum / 1000.0 + 0.3 * (d_before - d_after)
        if self.sigma >= self.model.lost_m:
            return self._obs(), reward - 10.0, True, False, {"lost": True}
        if self.pos == self.goal:
            return self._obs(), reward + 5.0, True, False, {"reached": True}
        return self._obs(), reward, False, self.steps >= self.max_steps, {}


class RLPlanner:
    """Plan with a trained policy on the EXPECTED map (cells with success >= 0.5 work)."""

    def __init__(self, policy_path: str, grid: Grid, model: BeliefModel):
        from stable_baselines3 import PPO
        self.policy = PPO.load(policy_path, device="cpu")
        self.env = RoutePlanningEnv(grid, model)

    def plan(self, start, goal):
        obs, _ = self.env.reset(options=dict(start=start, goal=goal, expected=True))
        done = trunc = False
        while not (done or trunc):
            a, _ = self.policy.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = self.env.step(a)
        return list(self.env.path)
