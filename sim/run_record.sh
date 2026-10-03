#!/usr/bin/env bash
# Record a long patrol (frames + ground truth) for offline datasets:
#   [WORLD=visloc03_real] [EXTRA_PARAMS=...] bash sim/run_record.sh <name> <n_frames> <every_s>
# Output: outputs/sim_rec/<name>/ (jpg frames + meta.json). GNSS stays on (labels come from SIMSTATE anyway).
set -uo pipefail
NAME="${1:-rec_real}"; N="${2:-800}"; EVERY="${3:-1.5}"
SIM_HOME="${SIM_HOME:-$HOME/gdnav_sim}"
WORLD="${WORLD:-visloc03_real}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
rm -f /tmp/gdnav_sim/sitl.pid
bash sim/run_sim.sh "$WORLD" > /tmp/gdnav_sim_run.log 2>&1 &
until [ -f /tmp/gdnav_sim/sitl.pid ]; do sleep 1; done; sleep 10
"$SIM_HOME/venv/bin/python" -u sim/fly_mission.py --world "sim/gz/worlds/$WORLD.json" --conn tcp:127.0.0.1:5760 \
  > "/tmp/gdnav_sim/fly_$NAME.log" 2>&1
grep -E "handover|could not|failed" "/tmp/gdnav_sim/fly_$NAME.log"
rm -rf "/tmp/gdnav_sim/$NAME"
"$SIM_HOME/ai/bin/python" -u sim/record.py --out "/tmp/gdnav_sim/$NAME" --n "$N" --every "$EVERY" --min-alt 300 \
  | awk 'NR % 100 == 1'
mkdir -p outputs/sim_rec && rm -rf "outputs/sim_rec/$NAME" && cp -r "/tmp/gdnav_sim/$NAME" "outputs/sim_rec/$NAME"
echo "recorded $(ls "outputs/sim_rec/$NAME" | wc -l) files"
bash sim/stop_sim.sh
