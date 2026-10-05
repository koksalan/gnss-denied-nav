"""Record nadir-camera frames with synchronized autopilot state (WSL, ai env).

Ground truth comes from SITL's SIMSTATE message (true simulator position, independent of GPS/EKF),
heading from ATTITUDE.yaw and height from GLOBAL_POSITION_INT.relative_alt (barometer/EKF).

    python sim/record.py --out /tmp/gdnav_sim/rec --n 80 --every 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
from pymavlink import mavutil


class LatestFrame:
    """Latest camera image with its wall-clock receive time and its SIMULATION timestamp.

    The simulator can run slower than real time (rendering + GPU load), so anything physical
    (velocity, latency x speed) must use simulation time, not the wall clock.
    """

    def __init__(self, topic: str = "/nadir_cam"):
        self.lock, self.frame, self.stamp, self.sim_stamp = threading.Lock(), None, 0.0, 0.0
        self.node = Node()
        if not self.node.subscribe(Image, topic, self._cb):
            raise RuntimeError(f"cannot subscribe {topic}")

    def _cb(self, msg: Image):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
        with self.lock:
            self.frame, self.stamp = arr, time.time()
            self.sim_stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9

    def get(self):
        """(frame, wall_time, sim_time)"""
        with self.lock:
            return self.frame, self.stamp, self.sim_stamp


class LatestBoxes:
    """2D boxes from the bounding-box camera, kept per simulation timestamp (to pair them with camera frames)."""

    def __init__(self, topic: str = "/nadir_boxes", keep: int = 50):
        self.lock, self.by_stamp, self.keep = threading.Lock(), {}, keep
        self.node = Node()
        if not self.node.subscribe(AnnotatedAxisAligned2DBox_V, topic, self._cb):
            raise RuntimeError(f"cannot subscribe {topic}")

    def _cb(self, msg):
        stamp = round(msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9, 3)
        boxes = [(b.label, b.box.min_corner.x, b.box.min_corner.y, b.box.max_corner.x, b.box.max_corner.y)
                 for b in msg.annotated_box]
        with self.lock:
            self.by_stamp[stamp] = boxes
            for k in sorted(self.by_stamp)[:-self.keep]:
                del self.by_stamp[k]

    def at(self, sim_stamp: float, tol: float = 0.02):
        with self.lock:
            if not self.by_stamp:
                return None
            k = min(self.by_stamp, key=lambda x: abs(x - sim_stamp))
            return self.by_stamp[k] if abs(k - sim_stamp) <= tol else None


class SimClock:
    """Current simulation time from Gazebo's world clock topic."""

    def __init__(self, world: str):
        self.sim = 0.0
        self.node = Node()
        if not self.node.subscribe(Clock, f"/world/{world}/clock", self._cb):
            raise RuntimeError("cannot subscribe clock")

    def _cb(self, msg: Clock):
        self.sim = msg.sim.sec + msg.sim.nsec * 1e-9

    def now(self) -> float:
        return self.sim


class AutopilotState:
    """Keeps the latest SIMSTATE / ATTITUDE / GLOBAL_POSITION_INT from a MAVLink link in a background thread."""

    def __init__(self, conn: str):
        self.m = mavutil.mavlink_connection(conn, source_system=254, retries=120)
        self.m.wait_heartbeat(timeout=30)
        self.m.mav.request_data_stream_send(self.m.target_system, self.m.target_component, 0, 10, 1)
        self.last: dict = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            msg = self.m.recv_match(type=["SIMSTATE", "ATTITUDE", "GLOBAL_POSITION_INT", "MISSION_CURRENT"],
                                    blocking=True, timeout=1)
            if msg is not None:
                self.last[msg.get_type()] = msg

    def snapshot(self) -> dict | None:
        s, a, g = (self.last.get(k) for k in ("SIMSTATE", "ATTITUDE", "GLOBAL_POSITION_INT"))
        if not (s and a and g):
            return None
        return dict(true_lat=s.lat / 1e7, true_lon=s.lng / 1e7,
                    true_roll_deg=math.degrees(s.roll), true_pitch_deg=math.degrees(s.pitch),
                    true_yaw_deg=math.degrees(s.yaw), yaw_deg=math.degrees(a.yaw),
                    roll_deg=math.degrees(a.roll), pitch_deg=math.degrees(a.pitch),
                    rel_alt_m=g.relative_alt / 1000, ekf_lat=g.lat / 1e7, ekf_lon=g.lon / 1e7,
                    ekf_vn=g.vx / 100, ekf_ve=g.vy / 100,
                    mission_seq=self.last["MISSION_CURRENT"].seq if "MISSION_CURRENT" in self.last else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--out", default="/tmp/gdnav_sim/rec")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--min-alt", type=float, default=350.0)
    ap.add_argument("--boxes", action="store_true", help="also save YOLO labels from the bounding-box camera")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam, ap_state = LatestFrame(), AutopilotState(args.conn)
    boxes = LatestBoxes() if args.boxes else None
    if boxes:
        (out / "labels").mkdir(exist_ok=True)
    rows = []
    while len(rows) < args.n:
        time.sleep(args.every)
        frame, stamp, sim_stamp = cam.get()
        st = ap_state.snapshot()
        if frame is None or st is None or st["rel_alt_m"] < args.min_alt or time.time() - stamp > 0.5:
            continue
        bx = boxes.at(sim_stamp) if boxes else None
        if boxes and bx is None:
            continue                                   # no label set for exactly this frame
        name = f"f{len(rows):04d}.jpg"
        if boxes:
            h, w = frame.shape[:2]
            lines = []
            for label, x0, y0, x1, y1 in bx:            # label 1 = car, 2 = large vehicle -> YOLO class label-1
                x0, x1, y0, y1 = max(x0, 0), min(x1, w), max(y0, 0), min(y1, h)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                lines.append(f"{label - 1} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            (out / "labels" / name.replace(".jpg", ".txt")).write_text("\n".join(lines))
        cv2.imwrite(str(out / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(dict(file=name, t=stamp, sim_t=sim_stamp, n_boxes=len(bx) if bx else 0, **st))
        print(f"{name} alt={st['rel_alt_m']:.0f} yaw={st['yaw_deg']:.0f} roll={st['roll_deg']:.0f}", flush=True)
    (out / "meta.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
