"""Build the per-flight cache (north-up queries + metric overview). Usage: prepare.py 01,02,03"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.prepared import prepare_flight  # noqa: E402

for f in sys.argv[1].split(","):
    print("prepared", prepare_flight(f))
