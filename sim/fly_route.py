"""Fly a planned route: climb at home (loiter to altitude), then the route waypoints, then loiter at the goal.

    python sim/fly_route.py --mission sim/missions/west_astar.json
Mission items: 0 home, 1 takeoff (skipped, see fly_mission.py), 2 loiter-to-alt at home, 3.. route, last loiter at goal.
The first route waypoint is item 3: visual_gps.py --jam-at-seq 3 jams GNSS exactly when the route starts.
"""
from __future__ import annotations

import argparse
import json
import time

from pymavlink import mavutil

from fly_mission import upload  # noqa: E402  (same directory)

M = mavutil.mavlink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--conn", default="tcp:127.0.0.1:5760")
    ap.add_argument("--handover-alt", type=float, default=60.0)
    args = ap.parse_args()
    ms = json.load(open(args.mission))
    alt = ms["alt_m"]
    hlat, hlon = ms["home"]

    m = mavutil.mavlink_connection(args.conn, source_system=255, retries=120)  # sysid 255 = SYSID_MYGCS (RC override)
    m.wait_heartbeat(timeout=60)
    m.mav.request_data_stream_send(m.target_system, m.target_component, M.MAV_DATA_STREAM_ALL, 4, 1)
    while True:
        g = m.recv_match(type="GPS_RAW_INT", blocking=True, timeout=5)
        if g and g.fix_type >= 3:
            break
    items = [(M.MAV_CMD_NAV_WAYPOINT, hlat, hlon, 0, (0, 0, 0, 0)),
             (M.MAV_CMD_NAV_TAKEOFF, hlat, hlon, alt, (15, 0, 0, 0)),
             (M.MAV_CMD_NAV_LOITER_TO_ALT, hlat, hlon, alt, (0, 150, 0, 1))]       # climb in circles to cruise alt
    items += [(M.MAV_CMD_NAV_WAYPOINT, la, lo, alt, (0, 0, 0, 0)) for la, lo in ms["waypoints"]]
    items.append((M.MAV_CMD_NAV_LOITER_UNLIM, *ms["goal"], alt, (0, 0, 150, 0)))
    upload(m, items)

    m.set_mode("FBWA")
    t0 = time.time()
    while not m.motors_armed():
        if time.time() - t0 > 120:
            raise RuntimeError("could not arm")
        m.arducopter_arm()
        m.recv_match(type=["STATUSTEXT", "HEARTBEAT"], blocking=True, timeout=3)
    rc = [65535] * 8
    rc[2] = 1800
    while True:
        m.mav.rc_channels_override_send(m.target_system, m.target_component, *rc)
        g = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if g and g.relative_alt / 1000 > args.handover_alt:
            break
    m.mav.rc_channels_override_send(m.target_system, m.target_component, *([0] * 8))
    # switch to AUTO and CONFIRM it (exiting right after the command can drop it: the aircraft then stays in FBWA)
    t0 = time.time()
    while True:
        m.mav.mission_set_current_send(m.target_system, m.target_component, 2)
        m.set_mode("AUTO")
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb and m.flightmode == "AUTO":
            break
        if time.time() - t0 > 30:
            raise RuntimeError(f"could not switch to AUTO (mode {m.flightmode})")
    print(f"handover to AUTO at {g.relative_alt / 1000:.0f} m, route has {len(ms['waypoints'])} waypoints", flush=True)
    t0 = time.time()
    while time.time() - t0 < 10:                              # stay connected briefly; log mission progress
        mc = m.recv_match(type="MISSION_CURRENT", blocking=True, timeout=2)
        if mc:
            print("  mission item", mc.seq, "mode", m.flightmode, flush=True)
            break


if __name__ == "__main__":
    main()
