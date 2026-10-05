#!/usr/bin/env bash
# Stop the simulator started by run_sim.sh. PID files first, then any leftover gz-sim servers / SITL binaries:
# the gz launcher spawns the server as a grandchild, so killing the launcher's children was not enough and stale
# servers kept the FDM port (the next SITL then saw two IMU streams: "Accels inconsistent").
LOG=/tmp/gdnav_sim
for f in "$LOG"/gz.pid "$LOG"/sitl.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f"); pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; rm -f "$f"
done
pkill -f "gz sim -v3" 2>/dev/null          # this script's own command line never contains that string
pkill -x arduplane 2>/dev/null
sleep 2
pkill -9 -f "gz sim -v3" 2>/dev/null; pkill -9 -x arduplane 2>/dev/null
true
