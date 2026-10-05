"""Sim-to-real matrix: every trained detector on both test sets (real VisDrone val, synthetic Gazebo test).

    python scripts/eval_det_matrix.py det_real det_sim det_mixed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
TESTS = {"real (VisDrone val)": ("../data/det/test_real.yaml", 960), "sim (Gazebo, held-out east)": ("../data/det/test_sim.yaml", 1280)}


def main(names: list[str]):
    res = {}
    for n in names:
        w = ROOT / "outputs" / "det" / n / "weights" / "best.pt"
        if not w.exists():
            print("missing", w); continue
        model = YOLO(str(w))
        res[n] = {}
        for tname, (yaml, imgsz) in TESTS.items():
            m = model.val(data=str((ROOT / yaml).resolve()), imgsz=imgsz, batch=16, split="val", plots=False,
                          verbose=False, project=str(ROOT / "outputs" / "det" / "_val"), name=f"{n}", exist_ok=True)
            car = 0
            res[n][tname] = dict(mAP50=round(float(m.box.map50), 3), mAP50_95=round(float(m.box.map), 3),
                                 car_AP50=round(float(m.box.ap50[car]), 3) if len(m.box.ap50) else None,
                                 recall=round(float(m.box.mr), 3), precision=round(float(m.box.mp), 3))
            print(n, tname, res[n][tname], flush=True)
    (ROOT / "outputs" / "det" / "matrix.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["det_real", "det_sim", "det_mixed"])
