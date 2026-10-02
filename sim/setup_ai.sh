#!/usr/bin/env bash
# Python env inside WSL for the onboard localizer (torch + matcher) that can also import Gazebo's python bindings.
set -euo pipefail
SIM_HOME="${SIM_HOME:-$HOME/gdnav_sim}"
python3 -m venv --system-site-packages "$SIM_HOME/ai"
. "$SIM_HOME/ai/bin/activate"
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers>=4.45" faiss-cpu rasterio opencv-python-headless pandas tqdm pymavlink kornia \
  "git+https://github.com/cvg/LightGlue.git"
python -c "import torch, gz.transport13; print('ai env ok', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
