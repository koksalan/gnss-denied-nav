"""Upload a patrol mission (takeoff -> rectangle at altitude, looping) and fly it in AUTO.

    python sim/fly_mission.py --world sim/gz/worlds/visloc03.json --alt 400 --half 1200
"""
from __future__ import annotations

import argparse
import json
import math
import time

from pymavlink import mavutil

M = mavutil.mavlink


def enu_to_ll(lat0, lon0, e, n):
    m_lat = 111132.92 - 559.82 * math.cos(2 * math.radians(lat0))
    m_lon = 111412.84 * math.cos(math.radians(lat0))
    return lat0 + n / m_lat, lon0 + e / m_lon


def wait_text(m, needle: str, timeout: float):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type="STATUSTEXT", blocking=True, timeout=1)
        if msg:
            print("  [ap]", msg.text)
            if needle.lower() in msg.text.lower():
                return True
    return False


def upload(m, items):
    m.mav.mission_count_send(m.target_system, m.target_component, len(items), M.MAV_MISSION_TYPE_MISSION)
    for _ in range(len(items)):
        req = m.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST"], blocking=True, timeout=10)
        if req is None:
            raise RuntimeError("mission upload timed out")
        seq = req.seq
        cmd, lat, lon, alt, p = items[seq]
        m.mav.mission_item_int_send(
            m.target_system, m.target_component, seq, M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, cmd,
            0, 1, p[0], p[1], p[2], p[3], int(lat * 1e7), int(lon * 1e7), alt, M.MAV_MISSION_TYPE_MISSION)
    ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=10)
    print("mission ack:", ack.type if ack else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="sim/gz/worlds/visloc03.json")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--alt", type=float, default=400.0)
    ap.add_argument("--half", type=float, default=1200.0, help="half side of the patrol square (m)")
    ap.add_argument("--handover-alt", type=float, default=60.0, help="switch from manual climb to AUTO (m)")
    args = ap.parse_args()
    w = json.load(open(args.world))
    lat0, lon0 = w["lat0"], w["lon0"]

    m = mavutil.mavlink_connection(args.conn, source_system=255)  # must equal SYSID_MYGCS or RC overrides are ignored
    m.wait_heartbeat(timeout=60)
    print("connected to system", m.target_system)
    m.mav.request_data_stream_send(m.target_system, m.target_component, M.MAV_DATA_STREAM_ALL, 4, 1)

    print("waiting for 3D GPS fix ...")
    while True:
        g = m.recv_match(type="GPS_RAW_INT", blocking=True, timeout=5)
        if g and g.fix_type >= 3:
            break

    h = args.half
    corners = [(-h, h), (h, h), (h, -h), (-h, -h)]                 # NW, NE, SE, SW (east, north)
    items = [(M.MAV_CMD_NAV_WAYPOINT, lat0, lon0, 0, (0, 0, 0, 0)),  # seq 0 = home (ignored)
             (M.MAV_CMD_NAV_TAKEOFF, lat0, lon0, args.alt, (15, 0, 0, 0))]
    for e, n in corners:
        lat, lon = enu_to_ll(lat0, lon0, e, n)
        items.append((M.MAV_CMD_NAV_WAYPOINT, lat, lon, args.alt, (0, 0, 0, 0)))
    items.append((M.MAV_CMD_DO_JUMP, 0, 0, 0, (2, -1, 0, 0)))       # loop the square forever
    upload(m, items)

    # The Zephyr sits vertically on its tail, which ArduPlane's AUTO takeoff check rejects ("Bad launch").
    # So: FBWA + throttle override to climb out, then switch to AUTO starting at the first patrol waypoint.
    m.set_mode("FBWA")
    t_arm = time.time()
    while not m.motors_armed():                                   # retry until pre-arm checks pass
        if time.time() - t_arm > 120:
            raise RuntimeError("could not arm")
        m.arducopter_arm()
        msg = m.recv_match(type=["STATUSTEXT", "HEARTBEAT"], blocking=True, timeout=3)
        if msg and msg.get_type() == "STATUSTEXT":
            print("  [ap]", msg.text)
    print("armed in FBWA, climbing out")
    rc = [65535] * 8
    rc[2] = 1800                                                  # throttle channel 3
    t0 = time.time()
    while True:
        m.mav.rc_channels_override_send(m.target_system, m.target_component, *rc)
        g = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if g and g.relative_alt / 1000 > args.handover_alt:
            break
        if time.time() - t0 > 90:
            raise RuntimeError("climb-out failed")
    m.mav.rc_channels_override_send(m.target_system, m.target_component, *([0] * 8))   # release override
    m.mav.mission_set_current_send(m.target_system, m.target_component, 2)
    m.set_mode("AUTO")
    print(f"handover to AUTO at {g.relative_alt / 1000:.0f} m")
    t0 = time.time()
    while time.time() - t0 < 60:
        msg = m.recv_match(type=["GLOBAL_POSITION_INT", "STATUSTEXT"], blocking=True, timeout=2)
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT":
            print("  [ap]", msg.text)
        elif int(time.time() - t0) % 5 == 0:
            print(f"  t={time.time() - t0:4.0f}s alt={msg.relative_alt / 1000:6.1f} m")
    print("mission running (loops). Check with: tail /tmp/ArduPlane.log or MAVLink on", args.conn)


if __name__ == "__main__":
    main()
