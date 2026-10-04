#!/usr/bin/env bash
# One GNSS-jamming experiment end to end:  [WORLD=visloc03_real] [GUI=1] bash sim/run_experiment.sh <tag> [node args]
#   start Gazebo+SITL -> climb-out + patrol -> visual-GPS node (jams GPS1 at t=90 s) -> stop simulator.
set -uo pipefail
TAG="${1:-visual}"; shift || true; EXTRA="$*"
SIM_HOME="${SIM_HOME:-$HOME/gdnav_sim}"
WORLD="${WORLD:-visloc03}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
rm -f /tmp/gdnav_sim/sitl.pid
bash sim/run_sim.sh "$WORLD" ${GUI:+--gui} > /tmp/gdnav_sim_run.log 2>&1 &
until [ -f /tmp/gdnav_sim/sitl.pid ]; do sleep 1; done; sleep 10
if [ -n "${MISSION:-}" ]; then          # planned route (sim/fly_route.py); otherwise the square patrol
  "$SIM_HOME/venv/bin/python" -u sim/fly_route.py --mission "$MISSION" --conn tcp:127.0.0.1:5760 > "/tmp/gdnav_sim/fly_$TAG.log" 2>&1
else
  "$SIM_HOME/venv/bin/python" -u sim/fly_mission.py --world "sim/gz/worlds/$WORLD.json" --conn tcp:127.0.0.1:5760 > "/tmp/gdnav_sim/fly_$TAG.log" 2>&1
fi
grep -E "handover|could not|failed" "/tmp/gdnav_sim/fly_$TAG.log"
"$SIM_HOME/ai/bin/python" -u sim/visual_gps.py --world "sim/gz/worlds/$WORLD.json" --tag "$TAG" --jam-after 90 --duration 480 $EXTRA
bash sim/stop_sim.sh
