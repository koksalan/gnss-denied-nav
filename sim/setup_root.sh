#!/usr/bin/env bash
# System packages for ArduPilot SITL + Gazebo Harmonic + ardupilot_gazebo plugin (Ubuntu 24.04).
# Run as root:  wsl -u root -e bash sim/setup_root.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  build-essential ccache g++ gawk git make wget curl lsb-release gnupg ca-certificates \
  python3-dev python3-venv python3-pip python-is-python3 libtool libxml2-dev libxslt1-dev \
  pkg-config cmake rapidjson-dev libopencv-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl

# Gazebo Harmonic from the OSRF repository
if ! command -v gz >/dev/null; then
  curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/gazebo-stable.list
  apt-get update
  apt-get install -y gz-harmonic
fi
echo "root setup done: $(gz sim --versions 2>/dev/null | tail -1)"
