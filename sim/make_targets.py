"""Add ground vehicles (labelled for the bounding-box camera) to a generated world.

Writes sim/gz/worlds/<world>_targets.sdf and sim/gz/worlds/<world>_targets.json (true vehicle positions, the
ground truth for target geolocation). Vehicles: Gazebo Fuel models, random poses + parking-lot-like clusters.
`--n-moving` adds vehicles driving circles (VelocityControl: constant speed v and yaw rate w); their position at
simulation time t is known in closed form (see `target_position`), the ground truth for motion estimation.

    python sim/make_targets.py --world visloc03_real --n-random 250 --clusters 20 --per-cluster 8
    python sim/make_targets.py --world visloc03_real --n-random 500 --clusters 60 --per-cluster 12 --n-moving 40 --suffix traffic
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FUEL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/"
MODELS = {  # name -> (class id, class name)
    "Hatchback": (0, "car"), "Hatchback blue": (0, "car"), "Hatchback red": (0, "car"), "Prius Hybrid": (0, "car"),
    "SUV": (0, "car"), "Pickup": (0, "car"), "Bus": (1, "large_vehicle"), "Ambulance": (1, "large_vehicle"),
}


VISUALS = json.loads((ROOT / "sim" / "vehicle_visuals.json").read_text())   # visual meshes of the Fuel models


def vehicle_sdf(i: int, model: str, e: float, n: float, yaw: float, label: int) -> str:
    """Visual-only static vehicle. The Fuel models carry full-mesh COLLISIONS; with ~1 200 of them the physics step
    dropped the simulation to 6 % of real time. Scenery needs no collisions, only looks + a label."""
    visuals = "".join(f"""
          <visual name="v{k}">
            <pose>{v['pose']}</pose>
            <geometry><mesh><scale>{v['scale']}</scale><uri>{v['uri']}</uri>{v['submesh']}</mesh></geometry>
          </visual>""" for k, v in enumerate(VISUALS[model.lower()]))
    return f"""
    <model name="veh_{i}">
      <static>true</static>
      <pose>{e:.2f} {n:.2f} 0.02 0 0 {yaw:.4f}</pose>
      <link name="link">{visuals}
      </link>
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label + 1}</label></plugin>
    </model>"""


def moving_vehicle_sdf(i: int, model: str, e: float, n: float, yaw: float, label: int, v: float, w: float) -> str:
    """Kinematic vehicle: no gravity, no collisions, body-frame velocity command (forward v, yaw rate w)."""
    visuals = "".join(f"""
          <visual name="v{k}">
            <pose>{vv['pose']}</pose>
            <geometry><mesh><scale>{vv['scale']}</scale><uri>{vv['uri']}</uri>{vv['submesh']}</mesh></geometry>
          </visual>""" for k, vv in enumerate(VISUALS[model.lower()]))
    return f"""
    <model name="veh_{i}">
      <pose>{e:.2f} {n:.2f} 0.02 0 0 {yaw:.4f}</pose>
      <link name="link">
        <gravity>false</gravity>
        <inertial><mass>1000</mass><inertia><ixx>1000</ixx><iyy>1000</iyy><izz>1000</izz></inertia></inertial>{visuals}
      </link>
      <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
        <initial_linear>{v:.3f} 0 0</initial_linear>
        <initial_angular>0 0 {w:.5f}</initial_angular>
      </plugin>
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label + 1}</label></plugin>
    </model>"""


def target_position(t: dict, sim_t: float) -> tuple[float, float]:
    """(east, north) of a vehicle at simulation time sim_t (static vehicles: their pose)."""
    v, w = t.get("speed", 0.0), t.get("yaw_rate", 0.0)
    if v == 0.0:
        return t["east_m"], t["north_m"]
    y0, y1 = t["yaw"], t["yaw"] + w * sim_t
    if abs(w) < 1e-9:
        return t["east_m"] + v * sim_t * math.cos(y0), t["north_m"] + v * sim_t * math.sin(y0)
    return (t["east_m"] + v / w * (math.sin(y1) - math.sin(y0)),
            t["north_m"] - v / w * (math.cos(y1) - math.cos(y0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="visloc03_real")
    ap.add_argument("--n-random", type=int, default=250)
    ap.add_argument("--clusters", type=int, default=20)
    ap.add_argument("--per-cluster", type=int, default=8)
    ap.add_argument("--half-m", type=float, default=1800.0)
    ap.add_argument("--n-moving", type=int, default=0)
    ap.add_argument("--moving-near-square", type=float, default=0.0,
                    help="put the moving vehicles along the patrol square of this half side (m), so they are seen;"
                         " each moving vehicle costs ~2.5 %% of Gazebo's real-time factor")
    ap.add_argument("--suffix", default="targets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    wdir = ROOT / "sim" / "gz" / "worlds"
    base = (wdir / f"{args.world}.sdf").read_text()
    names = list(MODELS)
    weights = np.array([3, 2, 2, 2, 2, 2, 0.6, 0.4]); weights /= weights.sum()
    targets = []

    def add(e, n, yaw):
        m = names[rng.choice(len(names), p=weights)]
        targets.append(dict(id=len(targets), model=m, cls=MODELS[m][0], cls_name=MODELS[m][1],
                            east_m=float(e), north_m=float(n), yaw=float(yaw)))

    for _ in range(args.n_random):
        add(*rng.uniform(-args.half_m, args.half_m, 2), rng.uniform(-math.pi, math.pi))
    for _ in range(args.clusters):                      # parking-lot rows: aligned, 3.2 m apart
        ce, cn, yaw = *rng.uniform(-args.half_m + 50, args.half_m - 50, 2), rng.uniform(-math.pi, math.pi)
        for k in range(args.per_cluster):
            row, col = divmod(k, 4)
            off = np.array([col * 3.2 - 4.8, row * 7.0])
            c, s = math.cos(yaw), math.sin(yaw)
            add(ce + c * off[0] - s * off[1], cn + s * off[0] + c * off[1], yaw + math.pi / 2)
    for _ in range(args.n_moving):                      # circles at 5-15 m/s
        if args.moving_near_square:
            h, along = args.moving_near_square, rng.uniform(-args.moving_near_square, args.moving_near_square)
            side = rng.integers(4)
            e, n = [(h, along), (-h, along), (along, h), (along, -h)][side]
            add(e + rng.uniform(-60, 60), n + rng.uniform(-60, 60), rng.uniform(-math.pi, math.pi))
            v, r = float(rng.uniform(5, 15)), float(rng.uniform(60, 150))
        else:
            add(*rng.uniform(-1200, 1200, 2), rng.uniform(-math.pi, math.pi))
            v, r = float(rng.uniform(5, 15)), float(rng.uniform(80, 300))
        targets[-1].update(speed=v, yaw_rate=float(v / r * rng.choice([-1, 1])))
    vehicles = "".join(
        moving_vehicle_sdf(t["id"], t["model"], t["east_m"], t["north_m"], t["yaw"], t["cls"], t["speed"], t["yaw_rate"])
        if t.get("speed") else vehicle_sdf(t["id"], t["model"], t["east_m"], t["north_m"], t["yaw"], t["cls"])
        for t in targets)
    i = base.rfind("</world>")
    world_name = f"{args.world}_{args.suffix}"
    sdf = base[:i] + vehicles + "\n  " + base[i:]
    sdf = sdf.replace(f'<world name="{args.world}">', f'<world name="{world_name}">', 1)
    (wdir / f"{world_name}.sdf").write_text(sdf)
    meta = json.loads((wdir / f"{args.world}.json").read_text())
    (wdir / f"{world_name}.json").write_text(json.dumps(dict(meta, targets=targets), indent=1))
    print(f"{len(targets)} vehicles -> {wdir / (world_name + '.sdf')}")


if __name__ == "__main__":
    main()
