#!/usr/bin/env bash
# ArduPilot (ArduPlane SITL) + ardupilot_gazebo plugin, built as the normal user.
# Run after setup_root.sh:  wsl -e bash sim/setup_user.sh
set -euo pipefail
SIM_HOME="${SIM_HOME:-$HOME/gdnav_sim}"
mkdir -p "$SIM_HOME" && cd "$SIM_HOME"

# isolated python env for ArduPilot build tools + MAVProxy
python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
pip install -q "empy==3.3.4" pexpect future lxml pymavlink MAVProxy dronecan setuptools

if [ ! -d ardupilot ]; then
  git clone --depth 1 --branch ArduPlane-stable --recurse-submodules --shallow-submodules \
    https://github.com/ArduPilot/ardupilot.git
fi
cd ardupilot
./waf configure --board sitl
./waf plane
cd ..

if [ ! -d ardupilot_gazebo ]; then
  git clone --depth 1 https://github.com/ArduPilot/ardupilot_gazebo.git
fi
cd ardupilot_gazebo
export GZ_VERSION=harmonic
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j"$(nproc)"

echo "user setup done: $SIM_HOME"
ls -la "$SIM_HOME/ardupilot/build/sitl/bin/"
