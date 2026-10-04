"""Train the PPO route planner on the localizability map.

    python scripts/train_rl_planner.py --steps 1500000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.planning import BeliefModel, Grid  # noqa: E402
from gdnav.rl_planner import RoutePlanningEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="outputs/plan/localizability.npz")
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--sigma-weight", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/plan/ppo_policy")
    args = ap.parse_args()
    z = np.load(ROOT / args.map)
    grid = Grid(z["success"], float(z["cell_m"]))
    env = make_vec_env(lambda: RoutePlanningEnv(grid, BeliefModel(), sigma_weight=args.sigma_weight), n_envs=8, seed=0)
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=1024, n_epochs=5, learning_rate=3e-4, gamma=0.99,
                ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]), verbose=0, seed=0, device="cpu")
    t0 = time.time()
    chunk = args.steps // 10
    log = []
    for i in range(10):
        model.learn(chunk, reset_num_timesteps=False)
        ep = list(model.ep_info_buffer)
        r = float(np.mean([e["r"] for e in ep])) if ep else float("nan")
        L = float(np.mean([e["l"] for e in ep])) if ep else float("nan")
        log.append(dict(steps=(i + 1) * chunk, ep_reward=round(r, 2), ep_len=round(L, 1)))
        print(f"{(i + 1) * chunk:>9} steps  ep_reward {r:7.2f}  ep_len {L:5.1f}  ({time.time() - t0:.0f}s)", flush=True)
    model.save(ROOT / args.out)
    (ROOT / f"{args.out}_train.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
