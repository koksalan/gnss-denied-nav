#!/usr/bin/env bash
# Start Gazebo (server, headless rendering) + ArduPlane SITL for a generated world.
#   bash sim/run_sim.sh visloc03 [--gui]
# MAVLink: SITL serial0 on tcp:127.0.0.1:5760 (also 5762, 5763). Logs in /tmp/gdnav_sim/.
set -euo pipefail
WORLD="${1:-visloc03}"
GUI="${2:-}"
SIM_HOME="${SIM_HOME:-$HOME/gdnav_sim}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/tmp/gdnav_sim; mkdir -p "$LOG"

export GZ_SIM_RESOURCE_PATH="$REPO/sim/gz/models:$REPO/sim/gz/worlds:$SIM_HOME/ardupilot_gazebo/models:$SIM_HOME/ardupilot_gazebo/worlds"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$SIM_HOME/ardupilot_gazebo/build"

read -r LAT LON ELEV < <(python3 -c "import json;d=json.load(open('$REPO/sim/gz/worlds/$WORLD.json'));print(d['lat0'],d['lon0'],d['elevation'])")

bash "$REPO/sim/stop_sim.sh" 2>/dev/null || true

if [ "$GUI" = "--gui" ]; then
  gz sim -v3 -r "$WORLD.sdf" > "$LOG/gz.log" 2>&1 &
else
  gz sim -v3 -s -r --headless-rendering "$WORLD.sdf" > "$LOG/gz.log" 2>&1 &
fi
echo $! > "$LOG/gz.pid"; echo "gazebo pid $!"

. "$SIM_HOME/venv/bin/activate"
cd "$LOG"
"$SIM_HOME/ardupilot/Tools/autotest/sim_vehicle.py" -v ArduPlane -f gazebo-zephyr --model JSON \
  --custom-location="$LAT,$LON,$ELEV,0" --add-param-file="$REPO/sim/params/gdnav.parm" \
  --no-mavproxy --no-rebuild -w > "$LOG/sitl.log" 2>&1 &
echo $! > "$LOG/sitl.pid"; echo "sitl pid $!  (home $LAT,$LON)"
# keep this shell alive while the simulator runs (WSL stops the distro when no session is left)
wait
