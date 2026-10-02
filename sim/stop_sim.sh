#!/usr/bin/env bash
# Stop the simulator started by run_sim.sh (process groups from pid files; never pattern-match command lines).
LOG=/tmp/gdnav_sim
for f in "$LOG"/gz.pid "$LOG"/sitl.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f"); pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; rm -f "$f"
done
pkill -x gz-sim-server 2>/dev/null; pkill -x arduplane 2>/dev/null
true
