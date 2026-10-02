"""Demo video from a simulator recording: camera frame | satellite map with true track and visual fixes.

    python sim/make_video.py --rec outputs/sim_rec/run2 --loc localization_pnp_p0_r0.csv --out docs/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.geo import meters_per_degree  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402

H = 600              # output height
MAP_PX = 600         # map panel size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="outputs/sim_rec/run2")
    ap.add_argument("--loc", default="localization_pnp_p0_r0.csv")
    ap.add_argument("--world", default="sim/gz/worlds/visloc03.json")
    ap.add_argument("--out", default="docs/demo.mp4")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    rec, w = ROOT / args.rec, json.loads((ROOT / args.world).read_text())
    meta = pd.DataFrame(json.loads((rec / "meta.json").read_text()))
    loc = pd.read_csv(rec / args.loc)
    m_lat, m_lon = meters_per_degree(w["lat0"])
    extent = 2700.0                                       # map panel shows the patrol area
    sat = VisLocFlight(w["flight"]).sat.crop(w["lat0"], w["lon0"], extent, extent / MAP_PX)
    sat = cv2.convertScaleAbs(sat, alpha=0.8)

    def to_px(lat, lon):
        e, n = (lon - w["lon0"]) * m_lon, (lat - w["lat0"]) * m_lat
        return int(MAP_PX / 2 + e / extent * MAP_PX), int(MAP_PX / 2 - n / extent * MAP_PX)

    fix_ll = []
    writer = imageio.get_writer(ROOT / args.out, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
    for i, r in meta.iterrows():
        frame = cv2.cvtColor(cv2.imread(str(rec / r.file)), cv2.COLOR_BGR2RGB)
        cam = cv2.resize(frame, (int(frame.shape[1] * H / frame.shape[0]), H))
        L = loc.iloc[i]
        panel = sat.copy()
        truth = [to_px(a, b) for a, b in zip(meta.true_lat[:i + 1], meta.true_lon[:i + 1])]
        cv2.polylines(panel, [np.array(truth)], False, (255, 255, 255), 2, cv2.LINE_AA)
        conf = bool(L.confident) and L["mode"] != "tilted"
        if conf:
            fix_ll.append((L.fix_lat, L.fix_lon))
        for p in fix_ll:
            cv2.circle(panel, to_px(*p), 3, (0, 255, 120), -1, cv2.LINE_AA)
        cv2.circle(panel, truth[-1], 6, (255, 60, 60), 2, cv2.LINE_AA)
        txt = f"error {L.err_m:4.1f} m" if conf else ("banked turn: skipped" if L["mode"] == "tilted" else "no fix")
        for img, lines in ((cam, ["onboard camera (nadir)", f"alt {r.rel_alt_m:.0f} m  heading {r.yaw_deg % 360:.0f} deg"]),
                           (panel, ["satellite map  (white: true track, green: visual fixes)", txt])):
            for k, s in enumerate(lines):
                cv2.putText(img, s, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, s, (12, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        writer.append_data(np.concatenate([cam, np.full((H, 8, 3), 255, np.uint8), panel], 1))
    writer.close()
    print("wrote", ROOT / args.out, f"({len(meta)} frames)")


if __name__ == "__main__":
    main()
