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

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
from gdnav.confidence import ConfidenceModel, fix_features  # noqa: E402
from gdnav.geo import haversine_m, meters_per_degree  # noqa: E402
from gdnav.geolocate import camera_pose_from_attitude, pixel_to_ground  # noqa: E402
from gdnav.pose import intrinsics  # noqa: E402
from gdnav.localizer import Localizer, LocalizerConfig  # noqa: E402
from gdnav.query import Camera  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402
from pymavlink import mavutil  # noqa: E402
from record import AutopilotState, LatestFrame, SimClock  # noqa: E402

M = mavutil.mavlink
GPS_EPOCH = 315964800          # 1980-01-06 in unix time
LEAP_SECONDS = 18
IGNORE_ALT = M.GPS_INPUT_IGNORE_FLAG_ALT | M.GPS_INPUT_IGNORE_FLAG_VDOP | M.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY
IGNORE_VEL = M.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ | M.GPS_INPUT_IGNORE_FLAG_VEL_VERT | M.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY


def consistency_gate(fix, rel_alt_m: float, roll_deg: float, pitch_deg: float, degraded: bool = False,
                     max_height_dev_m: float = 8.0, max_tilt_dev_deg: float = 4.0) -> tuple[bool, str]:
    """Rule-based sanity checks on a visual fix (PnP pose vs barometer / IMU). Returns (accepted, reason).

    A wrong match can still produce a plausible-looking position; its PnP pose, however, rarely agrees with the
    independent barometric height and IMU tilt. On a real-imagery run these checks rejected all 16 fixes that were
    > 50 m off while keeping 86 % of the fixes (thresholds chosen on that run; validated on a separate run).

    Degraded mode (no accepted fix for a while): the EKF attitude itself drifts without position updates, so the
    tilt comparison would reject correct fixes and lock the system out. Then tilt is not checked and a stricter
    match count is required instead; the barometric height check stays (it does not depend on GNSS).
    """
    min_inliers = 80 if degraded else 50
    if fix.inliers < min_inliers:
        return False, "few_inliers"
    if "height_m" not in fix.extra:
        return True, "no_pose"
    if abs(fix.extra["height_m"] - rel_alt_m) > max_height_dev_m:
        return False, "height"
    if not degraded and abs(fix.extra["off_nadir_deg"] - math.hypot(roll_deg, pitch_deg)) > max_tilt_dev_deg:
        return False, "tilt"
    return True, "ok"


def accuracy_from_inliers(n: int) -> float:
    """Reported horizontal accuracy (m): fewer matches -> less trust in the EKF."""
    return float(min(25.0, max(6.0, 5.0 + 600.0 / max(n, 1))))


def set_param(m, name: str, value: float):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), value, M.MAV_PARAM_TYPE_REAL32)


def send_gps_input(m, gps_unix_s: float, lat: float, lon: float, vel_ne: tuple[float, float] | None,
                   acc_m: float, speed_acc: float):
    """gps_unix_s must advance with SIMULATION time: ArduPilot uses the GPS week time for jitter correction."""
    gps_t = gps_unix_s - GPS_EPOCH + LEAP_SECONDS
    week, week_ms = int(gps_t // 604800), int((gps_t % 604800) * 1000)
    flags = IGNORE_ALT | (IGNORE_VEL if vel_ne is None else M.GPS_INPUT_IGNORE_FLAG_VEL_VERT)
    vn, ve = vel_ne or (0.0, 0.0)
    m.mav.gps_input_send(
        int(gps_unix_s * 1e6), 1, flags, week_ms, week, 3, int(lat * 1e7), int(lon * 1e7), 0.0,
        1.0, 1.0, vn, ve, 0.0, speed_acc, acc_m, 5.0, 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03.json")
    ap.add_argument("--weights", default="outputs/finetune/s1/best.pt")
    ap.add_argument("--patch-m", type=float, default=250.0)
    ap.add_argument("--jam-after", type=float, default=60.0)
    ap.add_argument("--jam-at-seq", type=int, default=None,
                    help="jam when the mission reaches this item (e.g. the first route waypoint) instead of a time")
    ap.add_argument("--duration", type=float, default=420.0)
    ap.add_argument("--min-alt", type=float, default=200.0)
    ap.add_argument("--no-send", action="store_true", help="control run: jam GNSS but send no visual fixes")
    ap.add_argument("--no-gate", action="store_true", help="disable the consistency gate (ablation)")
    ap.add_argument("--gate-model", default=None,
                    help="learned confidence model (joblib) instead of the hand-written rules")
    ap.add_argument("--pose", choices=["pnp", "boresight"], default="pnp",
                    help="pnp: position+attitude from the image matches; boresight: correct with IMU attitude")
    ap.add_argument("--attitude", choices=["ekf", "true"], default="ekf",
                    help="attitude used for boresight correction: autopilot estimate, or simulator truth (ablation)")
    ap.add_argument("--detector", default=None, help="YOLO weights: detect + geolocate ground vehicles")
    ap.add_argument("--frame-every", type=int, default=0,
                    help="ground-station telemetry: save every N-th camera frame (0 = telemetry without frames)")
    ap.add_argument("--tag", default="visual")
    args = ap.parse_args()

    w = json.loads((ROOT / args.world).read_text())
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    bounds = (w["lat0"] - half / m_lat, w["lon0"] - half / m_lon, w["lat0"] + half / m_lat, w["lon0"] + half / m_lon)
    cam = Camera(w["camera"]["focal_px"], -1, 90.0)          # mount convention measured in eval_recording.py
    out = ROOT / "outputs" / "sim_runs" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # control runs (--no-send) still localize when a detector is used (to compare target geolocation), but never send
    loc = None if (args.no_send and not args.detector) else Localizer(
        VisLocFlight(w["flight"]).sat, LocalizerConfig(patch_m=args.patch_m, pose=args.pose, rerank_k=25), str(ROOT / args.weights), bounds_ll=bounds)
    frames, ap_state, clock = LatestFrame(), AutopilotState(args.conn), SimClock(Path(args.world).stem)
    model = ConfidenceModel(ROOT / args.gate_model) if args.gate_model else None
    det, det_log, targets_en = None, None, None
    if args.detector:
        from ultralytics import YOLO
        det = YOLO(str(ROOT / args.detector))
        targets_en = np.array([[t["east_m"], t["north_m"]] for t in w.get("targets", [])])
        det_log = csv.writer(open(out / "detections.csv", "w", newline=""))
        det_log.writerow(["t", "jammed", "cls", "conf", "u", "v", "true_vehicle_id", "dist_truepose_to_vehicle_m",
                          "err_ekf_pose_m", "err_pnp_pose_m", "pnp_confident"])
        K_cam = intrinsics(w["camera"]["focal_px"], w["camera"]["width"], w["camera"]["height"])
    m = ap_state.m
    set_param(m, "SIM_GPS1_ENABLE", 1)
    if not args.no_send:
        set_param(m, "GPS2_TYPE", 14)              # bring the visual GPS online (MAVLink GPS_INPUT driver)
        time.sleep(1.0)
        # fixes are latency-compensated below (projected to send time), so the EKF should treat them as current
        set_param(m, "GPS2_DELAY_MS", 0)
    print("node ready", "(control run: no visual fixes sent)" if args.no_send else "", f"({len(loc.tiles_ll)} tiles)" if loc else "",
          flush=True)

    log = open(out / "log.csv", "w", newline="")
    # ground-station telemetry (gcs/): one JSON line per loop, flushed immediately so a live panel can tail it
    tel = open(out / "telemetry.jsonl", "w", buffering=1)
    if args.frame_every:
        (out / "frames").mkdir(exist_ok=True)
    n_loop = 0
    wr = csv.writer(log)
    wr.writerow(["t", "jammed", "true_lat", "true_lon", "ekf_lat", "ekf_lon", "ekf_err_m",
                 "roll", "pitch", "yaw", "true_roll", "true_pitch", "true_yaw",
                 "fix_lat", "fix_lon", "fix_err_m", "inliers", "mode", "sent", "vis_height_m", "vis_off_nadir",
                 "latency_s", "gate", "p_bad", "acc_pred"])
    t0, jammed, prior = time.time(), False, None
    last_accept_sim = -1e9
    gps_epoch_base = time.time() - clock.now()          # GPS time = wall time at start + simulation seconds
    while time.time() - t0 < args.duration:
        t = time.time() - t0
        snap = ap_state.snapshot()
        jam_now = (snap is not None and snap["mission_seq"] >= args.jam_at_seq) if args.jam_at_seq is not None             else t > args.jam_after
        if not jammed and jam_now:
            set_param(m, "SIM_GPS1_ENABLE", 0)
            jammed = True
            print(f"t={t:5.0f}s  >>> GNSS JAMMED (GPS1 off)", flush=True)
        st = ap_state.snapshot()
        frame, stamp, sim_stamp = frames.get()
        if st is None or frame is None:
            time.sleep(0.1)
            continue
        ekf_err = float(haversine_m(st["true_lat"], st["true_lon"], st["ekf_lat"], st["ekf_lon"]))
        row = [round(t, 2), int(jammed), st["true_lat"], st["true_lon"], st["ekf_lat"], st["ekf_lon"], round(ekf_err, 1),
               *(round(st[k], 2) for k in ("roll_deg", "pitch_deg", "yaw_deg",
                                           "true_roll_deg", "true_pitch_deg", "true_yaw_deg"))]
        if loc is None or st["rel_alt_m"] < args.min_alt:
            wr.writerow(row + ["", "", "", "", "", 0, "", ""])
            tel.write(json.dumps(dict(t=round(t, 2), jammed=int(jammed), mode="climb", alt=round(st["rel_alt_m"], 1),
                                      yaw=round(st["yaw_deg"], 1), true=[st["true_lat"], st["true_lon"]],
                                      ekf=[st["ekf_lat"], st["ekf_lon"]], ekf_err=round(ekf_err, 1), dets=[])) + "\n")
            time.sleep(0.25)
            continue
        pre = "true_" if args.attitude == "true" else ""
        fix, mode = loc.localize(frame, st["rel_alt_m"], st[pre + "yaw_deg"], cam, prior,
                                 roll_deg=st[pre + "roll_deg"], pitch_deg=st[pre + "pitch_deg"])
        sent, gate, p_bad, acc_pred = 0, "", "", ""
        if mode == "tilted":
            time.sleep(0.2)                                # banked turn: no fix, don't spin
        else:
            conf = fix.inliers >= loc.cfg.min_inliers
            gate = "off"
            if conf and not args.no_gate:
                degraded = clock.now() - last_accept_sim > 5.0
                if model is not None:
                    feats = fix_features(fix, mode, st["rel_alt_m"], st["roll_deg"], st["pitch_deg"])
                    if degraded:                 # drifting EKF attitude: let the model treat tilt as unknown
                        feats["tilt_dev"] = float("nan")
                    p_bad, acc_pred = model.predict(feats)
                    conf, gate = p_bad < model.threshold, ("model_ok" if p_bad < model.threshold else "model_reject")
                else:
                    conf, gate = consistency_gate(fix, st["rel_alt_m"], st["roll_deg"], st["pitch_deg"], degraded)
                if degraded and gate in ("ok", "model_ok"):
                    gate += "_degraded"
            if conf:
                prior, last_accept_sim = (fix.lat, fix.lon), clock.now()
            elif gate in ("off", ""):                      # genuinely lost: relocalize globally next frame
                prior = None
            # (a fix rejected by the gate keeps the previous prior: one bad frame should not drop tracking)
            if conf:
                # velocity over a >= 1 s baseline: ~5 m fix noise would make frame-to-frame differences useless
                # all timing in SIMULATION seconds (the sim may run slower than the wall clock)
                if args.no_gate:
                    acc = 8.0
                elif model is not None:
                    acc = float(min(60.0, max(5.0, acc_pred)))      # learned q80 error as reported accuracy
                else:
                    acc = accuracy_from_inliers(fix.inliers)
                # latency compensation: the fix is where the aircraft was when the frame was taken; project it to
                # 'now' with the EKF's own (smooth, IMU + airspeed) velocity. No velocity is SENT: the EKF takes
                # position only from GPS (EK3_SRC1_VELXY 0) because finite-difference visual velocity is too noisy.
                latency = clock.now() - sim_stamp
                lat_s, lon_s = fix.lat, fix.lon
                if latency < 3.0:
                    lat_s += st["ekf_vn"] * latency / m_lat
                    lon_s += st["ekf_ve"] * latency / m_lon
                if not args.no_send:
                    send_gps_input(m, gps_epoch_base + clock.now(), lat_s, lon_s, None, acc_m=acc, speed_acc=99.0)
                    sent = 1
        dets_tel = []
        if det is not None and mode != "tilted":
            res = det.predict(frame, imgsz=1280, conf=0.3, verbose=False)[0]
            if len(res.boxes):
                uv = res.boxes.xywh[:, :2].cpu().numpy()
                cls, cf = res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()
                en_ll = lambda lat, lon: np.array([(lon - w["lon0"]) * m_lon, (lat - w["lat0"]) * m_lat])  # noqa: E731
                true_c = np.r_[en_ll(st["true_lat"], st["true_lon"]), st["rel_alt_m"]]
                p_true = pixel_to_ground(uv, K_cam, camera_pose_from_attitude(
                    st["true_roll_deg"], st["true_pitch_deg"], st["true_yaw_deg"]), true_c)
                p_ekf = pixel_to_ground(uv, K_cam, camera_pose_from_attitude(
                    st["roll_deg"], st["pitch_deg"], st["yaw_deg"]), np.r_[en_ll(st["ekf_lat"], st["ekf_lon"]), st["rel_alt_m"]])
                p_pnp = None
                if "R_cw" in fix.extra:
                    p_pnp = pixel_to_ground(uv, K_cam, np.array(fix.extra["R_cw"]),
                                            np.r_[en_ll(fix.lat, fix.lon), fix.extra["height_m"]])
                d_all = np.linalg.norm(p_true[:, None, :] - targets_en[None], axis=2)
                vid = d_all.argmin(1)
                wh = res.boxes.xywh[:, 2:].cpu().numpy()
                to_ll = lambda p: [w["lat0"] + p[1] / m_lat, w["lon0"] + p[0] / m_lon]  # noqa: E731
                for k in range(len(uv)):
                    dets_tel.append(dict(u=float(uv[k, 0]), v=float(uv[k, 1]), w=float(wh[k, 0]), h=float(wh[k, 1]),
                                         cls=int(cls[k]), conf=round(float(cf[k]), 2), true_vid=int(vid[k]),
                                         ekf=to_ll(p_ekf[k]), pnp=to_ll(p_pnp[k]) if p_pnp is not None else None))
                for k in range(len(uv)):
                    tv = targets_en[vid[k]]
                    det_log.writerow([round(t, 2), int(jammed), cls[k], round(float(cf[k]), 3), round(float(uv[k, 0])),
                                      round(float(uv[k, 1])), int(vid[k]), round(float(d_all[k, vid[k]]), 1),
                                      round(float(np.linalg.norm(p_ekf[k] - tv)), 1),
                                      round(float(np.linalg.norm(p_pnp[k] - tv)), 1) if p_pnp is not None else "",
                                      int(fix.inliers >= loc.cfg.min_inliers)])
        fix_err = float(haversine_m(st["true_lat"], st["true_lon"], fix.lat, fix.lon)) if fix.inliers else ""
        wr.writerow(row + [fix.lat, fix.lon, round(fix_err, 1) if fix_err != "" else "", fix.inliers, mode, sent,
                           fix.extra.get("height_m", ""), fix.extra.get("off_nadir_deg", ""),
                           round(clock.now() - sim_stamp, 3), gate, p_bad, acc_pred])
        n_loop += 1
        frame_name = None
        if args.frame_every and n_loop % args.frame_every == 0:
            frame_name = f"{n_loop:06d}.jpg"
            small = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "frames" / frame_name), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        tel.write(json.dumps(dict(
            t=round(t, 2), sim_t=round(sim_stamp, 2), jammed=int(jammed), mode=mode, gate=gate, sent=sent,
            inliers=int(fix.inliers), latency=round(clock.now() - sim_stamp, 3), alt=round(st["rel_alt_m"], 1),
            yaw=round(st["yaw_deg"], 1), roll=round(st["roll_deg"], 1), pitch=round(st["pitch_deg"], 1),
            true=[st["true_lat"], st["true_lon"]], ekf=[st["ekf_lat"], st["ekf_lon"]],
            fix=[fix.lat, fix.lon] if fix.inliers else None, ekf_err=round(ekf_err, 1),
            fix_err=round(fix_err, 1) if fix_err != "" else None, frame=frame_name, dets=dets_tel)) + "\n")
        if int(t) % 15 == 0:
            print(f"t={t:5.0f}s jammed={int(jammed)} ekf_err={ekf_err:6.1f} m  fix_err={fix_err} mode={mode}", flush=True)
            log.flush()
    log.close()
    tel.close()
    set_param(m, "SIM_GPS1_ENABLE", 1)
    print("done ->", out / "log.csv", flush=True)


if __name__ == "__main__":
    main()
