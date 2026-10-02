"""Onboard visual-GPS node + GNSS-jamming experiment (WSL, ai env).

Camera frame + attitude/height from the autopilot -> Localizer -> MAVLink GPS_INPUT as GPS2.
After `--jam-after` seconds the simulated GNSS receiver (GPS1) is switched off; ArduPilot's EKF then
has only the visual GPS (or nothing, with --no-send = control run). Everything is logged against
SITL's SIMSTATE ground truth.

    python sim/visual_gps.py --jam-after 60 --duration 420 --tag visual
    python sim/visual_gps.py --jam-after 60 --duration 420 --tag control --no-send
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
from gdnav.geo import haversine_m, meters_per_degree  # noqa: E402
from gdnav.localizer import Localizer, LocalizerConfig  # noqa: E402
from gdnav.query import Camera  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402
from pymavlink import mavutil  # noqa: E402
from record import AutopilotState, LatestFrame  # noqa: E402

M = mavutil.mavlink
GPS_EPOCH = 315964800          # 1980-01-06 in unix time
LEAP_SECONDS = 18
IGNORE_ALT = M.GPS_INPUT_IGNORE_FLAG_ALT | M.GPS_INPUT_IGNORE_FLAG_VDOP | M.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY
IGNORE_VEL = M.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ | M.GPS_INPUT_IGNORE_FLAG_VEL_VERT | M.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY


def set_param(m, name: str, value: float):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), value, M.MAV_PARAM_TYPE_REAL32)


def send_gps_input(m, stamp: float, lat: float, lon: float, vel_ne: tuple[float, float] | None, acc_m: float):
    gps_t = stamp - GPS_EPOCH + LEAP_SECONDS
    week, week_ms = int(gps_t // 604800), int((gps_t % 604800) * 1000)
    flags = IGNORE_ALT | (IGNORE_VEL if vel_ne is None else M.GPS_INPUT_IGNORE_FLAG_VEL_VERT)
    vn, ve = vel_ne or (0.0, 0.0)
    m.mav.gps_input_send(
        int(stamp * 1e6), 1, flags, week_ms, week, 3, int(lat * 1e7), int(lon * 1e7), 0.0,
        1.0, 1.0, vn, ve, 0.0, 1.0, acc_m, 5.0, 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03.json")
    ap.add_argument("--weights", default="outputs/finetune/s1/best.pt")
    ap.add_argument("--patch-m", type=float, default=250.0)
    ap.add_argument("--jam-after", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=420.0)
    ap.add_argument("--min-alt", type=float, default=200.0)
    ap.add_argument("--no-send", action="store_true", help="control run: jam GNSS but send no visual fixes")
    ap.add_argument("--attitude", choices=["ekf", "true"], default="ekf",
                    help="attitude used for boresight correction: autopilot estimate, or simulator truth (ablation)")
    ap.add_argument("--tag", default="visual")
    args = ap.parse_args()

    w = json.loads((ROOT / args.world).read_text())
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    bounds = (w["lat0"] - half / m_lat, w["lon0"] - half / m_lon, w["lat0"] + half / m_lat, w["lon0"] + half / m_lon)
    cam = Camera(w["camera"]["focal_px"], -1, 90.0)          # mount convention measured in eval_recording.py
    out = ROOT / "outputs" / "sim_runs" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    loc = None if args.no_send else Localizer(
        VisLocFlight(w["flight"]).sat, LocalizerConfig(patch_m=args.patch_m), str(ROOT / args.weights), bounds_ll=bounds)
    frames, ap_state = LatestFrame(), AutopilotState(args.conn)
    m = ap_state.m
    set_param(m, "SIM_GPS1_ENABLE", 1)
    if not args.no_send:
        set_param(m, "GPS2_TYPE", 14)              # bring the visual GPS online (MAVLink GPS_INPUT driver)
        time.sleep(1.0)
        set_param(m, "GPS2_DELAY_MS", 300)         # localization latency, used by the EKF to time-align fixes
    print("node ready", "(control run, no visual fixes)" if args.no_send else f"({len(loc.tiles_ll)} tiles)", flush=True)

    log = open(out / "log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["t", "jammed", "true_lat", "true_lon", "ekf_lat", "ekf_lon", "ekf_err_m",
                 "roll", "pitch", "yaw", "true_roll", "true_pitch", "true_yaw",
                 "fix_lat", "fix_lon", "fix_err_m", "inliers", "mode", "sent"])
    t0, jammed, prior, history = time.time(), False, None, []      # history: confident (stamp, lat, lon)
    while time.time() - t0 < args.duration:
        t = time.time() - t0
        if not jammed and t > args.jam_after:
            set_param(m, "SIM_GPS1_ENABLE", 0)
            jammed = True
            print(f"t={t:5.0f}s  >>> GNSS JAMMED (GPS1 off)", flush=True)
        st = ap_state.snapshot()
        frame, stamp = frames.get()
        if st is None or frame is None:
            time.sleep(0.1)
            continue
        ekf_err = float(haversine_m(st["true_lat"], st["true_lon"], st["ekf_lat"], st["ekf_lon"]))
        row = [round(t, 2), int(jammed), st["true_lat"], st["true_lon"], st["ekf_lat"], st["ekf_lon"], round(ekf_err, 1),
               *(round(st[k], 2) for k in ("roll_deg", "pitch_deg", "yaw_deg",
                                           "true_roll_deg", "true_pitch_deg", "true_yaw_deg"))]
        if loc is None or st["rel_alt_m"] < args.min_alt:
            wr.writerow(row + ["", "", "", "", "", 0])
            time.sleep(0.25)
            continue
        pre = "true_" if args.attitude == "true" else ""
        fix, mode = loc.localize(frame, st["rel_alt_m"], st[pre + "yaw_deg"], cam, prior,
                                 roll_deg=st[pre + "roll_deg"], pitch_deg=st[pre + "pitch_deg"])
        sent = 0
        if mode == "tilted":
            time.sleep(0.2)                                # banked turn: no fix, don't spin
        else:
            conf = fix.inliers >= loc.cfg.min_inliers
            prior = (fix.lat, fix.lon) if conf else None
            if conf:
                # velocity over a >= 1 s baseline: ~5 m fix noise would make frame-to-frame differences useless
                history = [h for h in history if stamp - h[0] < 2.5] + [(stamp, fix.lat, fix.lon)]
                old = next((h for h in history if stamp - h[0] >= 1.0), None)
                vel = None if old is None else ((fix.lat - old[1]) * m_lat / (stamp - old[0]),
                                                (fix.lon - old[2]) * m_lon / (stamp - old[0]))
                send_gps_input(m, stamp, fix.lat, fix.lon, vel, acc_m=8.0)
                sent = 1
        fix_err = float(haversine_m(st["true_lat"], st["true_lon"], fix.lat, fix.lon)) if fix.inliers else ""
        wr.writerow(row + [fix.lat, fix.lon, round(fix_err, 1) if fix_err != "" else "", fix.inliers, mode, sent])
        if int(t) % 15 == 0:
            print(f"t={t:5.0f}s jammed={int(jammed)} ekf_err={ekf_err:6.1f} m  fix_err={fix_err} mode={mode}", flush=True)
            log.flush()
    log.close()
    set_param(m, "SIM_GPS1_ENABLE", 1)
    print("done ->", out / "log.csv", flush=True)


if __name__ == "__main__":
    main()
