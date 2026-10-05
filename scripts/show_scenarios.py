"""Grid of the hard test scenarios on one real UAV-VisLoc frame (what the localizer receives: north-up patch).

    python scripts/show_scenarios.py --flight 03 --idx 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav import corruptions as C  # noqa: E402
from gdnav.prepared import CONFIG_ROOT, PreparedFlight  # noqa: E402
from gdnav.query import Camera, make_query  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="03")
    ap.add_argument("--idx", type=int, default=300)
    ap.add_argument("--out", default="docs/hard_scenarios.jpg")
    args = ap.parse_args()
    fl, pf = VisLocFlight(args.flight), PreparedFlight.load(args.flight)
    cam = Camera.load(CONFIG_ROOT / f"camera_visloc{args.flight}.json")
    r = fl.row(args.idx)
    raw = cv2.cvtColor(cv2.imread(str(fl.image_path(args.idx))), cv2.COLOR_BGR2RGB)
    raw = cv2.resize(raw, (raw.shape[1] // 2, raw.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cam = Camera(cam.focal_px / 2, cam.heading_sign, cam.heading_offset_deg)
    tiles = []
    for s in C.SCENARIOS:
        img, h, hd = C.apply(s, raw, r.height, r.Phi1, seed=args.idx)
        q = make_query(img, h, hd, cam, 320, pf.patch_m)
        q = cv2.copyMakeBorder(q, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(q, s.label if len(s.label) <= 34 else "combined: fog+link+compass+baro", (6, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(q)
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(args.out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(args.out)


if __name__ == "__main__":
    main()
