"""Replay the map-frame target tracker on a recorded flight (telemetry.jsonl detections) to compare settings.

Truth: a detection belongs to the vehicle given by its true-pose association (true_vid) if the visual estimate is
within `match_m` of that vehicle at the frame's simulation time; moving vehicles from the world file.

    python sim/eval_tracking.py traffic_visual
    python sim/eval_tracking.py traffic_visual --rewrite   # re-annotate the telemetry with the final tracker
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
from gdnav.geo import meters_per_degree  # noqa: E402
from gdnav.tracking import TargetTracker  # noqa: E402
from make_targets import target_position  # noqa: E402


def replay(rows, w, tlist, moving, **kw) -> dict:
    m_lat, m_lon = meters_per_degree(w["lat0"])
    tr = TargetTracker(**kw)
    tp = fp = fn = tn = 0
    spd_err, hdg_err = [], []
    for r in rows:
        dets = [d for d in r.get("dets", []) if d.get("pnp") or d.get("ekf")]
        if not r.get("jammed") or not dets or r.get("sim_t") is None:
            continue
        en = np.array([[((d["pnp"] or d["ekf"])[1] - w["lon0"]) * m_lon, ((d["pnp"] or d["ekf"])[0] - w["lat0"]) * m_lat]
                       for d in dets])       # as onboard: PnP pose of the frame, autopilot pose if PnP failed
        tracks = tr.update(r["sim_t"], en, np.array([d["cls"] for d in dets]))
        for d, p, k in zip(dets, en, tracks):
            t = tlist[d["true_vid"]]
            if np.hypot(*(p - np.array(target_position(t, r["sim_t"])))) > 10:
                continue                                        # false detection or far association: not scored
            mv = tr.is_moving(k)
            if mv is None:
                continue
            truth = d["true_vid"] in moving
            tp += truth and mv; fp += (not truth) and mv; fn += truth and not mv; tn += (not truth) and not mv
            if truth and mv:
                yaw = t["yaw"] + t["yaw_rate"] * r["sim_t"]
                spd_err.append(abs(k.speed - t["speed"]))
                hdg_err.append(abs((k.heading_deg - (90 - np.degrees(yaw)) + 180) % 360 - 180))
    return dict(precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
                static_false_moving=round(fp / max(fp + tn, 1), 4), n_moving=tp + fn, n_static=fp + tn,
                speed_err_median=round(float(np.median(spd_err)), 2) if spd_err else None,
                heading_err_median=round(float(np.median(hdg_err)), 1) if hdg_err else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--rewrite", action="store_true",
                    help="replace mov/spd/hdg/trk of the telemetry detections with the default tracker (backup kept)")
    args = ap.parse_args()
    run = ROOT / "outputs" / "sim_runs" / args.tag
    w = json.loads((ROOT / "sim" / "gz" / "worlds" / (json.loads((run / "run.json").read_text())["world"] + ".json")).read_text())
    tlist = w["targets"]
    moving = {t["id"] for t in tlist if t.get("speed")}
    src = run / "telemetry_as_flown.jsonl" if (run / "telemetry_as_flown.jsonl").exists() else run / "telemetry.jsonl"
    rows = [json.loads(line) for line in open(src)]
    variants = {
        "greedy, speed only (as flown)": dict(hungarian=False, min_r2=0.0),
        "+ common-mode correction": dict(hungarian=False, min_r2=0.0, common_mode=True),
        "Hungarian association": dict(hungarian=True, min_r2=0.0),
        "Hungarian + consistency (R2 >= 0.8)": dict(hungarian=True, min_r2=0.8),
        "Hungarian + R2 >= 0.9": dict(hungarian=True, min_r2=0.9),
        "greedy + R2 >= 0.9": dict(hungarian=False, min_r2=0.9),
    }
    res = {name: replay(rows, w, tlist, moving, **kw) for name, kw in variants.items()}
    for k, v in res.items():
        print(f"{k:38s} {v}")
    (run / "tracking_eval.json").write_text(json.dumps(res, indent=1))
    if args.rewrite:
        m_lat, m_lon = meters_per_degree(w["lat0"])
        tr = TargetTracker()
        for r in rows:
            dets = [d for d in r.get("dets", []) if d.get("pnp") or d.get("ekf")]
            if not dets or r.get("sim_t") is None:
                continue
            en = np.array([[((d["pnp"] or d["ekf"])[1] - w["lon0"]) * m_lon, ((d["pnp"] or d["ekf"])[0] - w["lat0"]) * m_lat]
                       for d in dets])       # as onboard: PnP pose of the frame, autopilot pose if PnP failed
            for d, k in zip(dets, tr.update(r["sim_t"], en, np.array([d["cls"] for d in dets]))):
                d.update(trk=k.id, spd=round(k.speed, 1), hdg=round(k.heading_deg), mov=tr.is_moving(k))
        if not (run / "telemetry_as_flown.jsonl").exists():
            shutil.copy(run / "telemetry.jsonl", run / "telemetry_as_flown.jsonl")
        with open(run / "telemetry.jsonl", "w") as f:
            f.writelines(json.dumps(r) + chr(10) for r in rows)
        print("telemetry re-annotated (original: telemetry_as_flown.jsonl)")


if __name__ == "__main__":
    main()
