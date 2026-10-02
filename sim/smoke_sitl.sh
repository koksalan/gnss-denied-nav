#!/usr/bin/env bash
# Start headless ArduPlane SITL, connect with pymavlink, print GPS fix + attitude, then stop.
set -euo pipefail
SIM_HOME="${SIM_HOME:-$HOME/gdnav_sim}"
. "$SIM_HOME/venv/bin/activate"
cd "$SIM_HOME/ardupilot"
mkdir -p /tmp/sitl && cd /tmp/sitl
"$SIM_HOME/ardupilot/build/sitl/bin/arduplane" --model plane \
  --defaults "$SIM_HOME/ardupilot/Tools/autotest/models/plane.parm" -I0 > sitl.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
python - <<'PY'
import time
from pymavlink import mavutil
m = mavutil.mavlink_connection("tcp:127.0.0.1:5760", retries=30)
hb = m.wait_heartbeat(timeout=60)
if hb is None: print(open("/tmp/sitl/sitl.log").read()[-2000:]); raise SystemExit("no heartbeat")
print("heartbeat from system", m.target_system, "type", hb.type)
t0 = time.time(); got = {}
while time.time() - t0 < 25 and len(got) < 2:
    msg = m.recv_match(type=["GPS_RAW_INT", "ATTITUDE"], blocking=True, timeout=5)
    if msg: got[msg.get_type()] = msg
for k, v in got.items(): print(k, v.to_dict())
PY
