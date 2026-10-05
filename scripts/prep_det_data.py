"""Detection datasets with a common 2-class scheme (0 car, 1 large_vehicle), in Ultralytics YOLO layout.

  real : VisDrone2019-DET (car, van -> car; truck, bus -> large_vehicle; people/bikes dropped), images hard-linked
  sim  : Gazebo recording with bounding-box-camera labels (sim/record.py --boxes), split SPATIALLY:
         frames with the aircraft west of the world center -> train, east -> test
  mixed: real train + sim train

    python scripts/prep_det_data.py --visdrone ../datasets/VisDrone --sim outputs/sim_rec/det_sim
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISDRONE_MAP = {3: 0, 4: 0, 5: 1, 8: 1}
NAMES = {0: "car", 1: "large_vehicle"}


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def write_yaml(path: Path, train: list[str], val: list[str]):
    path.write_text(f"path: {path.parent.as_posix()}\ntrain: {json.dumps(train)}\nval: {json.dumps(val)}\n"
                    f"names: {json.dumps(NAMES)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visdrone", default="../datasets/VisDrone")
    ap.add_argument("--sim", default="outputs/sim_rec/det_sim")
    ap.add_argument("--out", default="../data/det")
    args = ap.parse_args()
    vd, sim, out = (ROOT / args.visdrone).resolve(), (ROOT / args.sim).resolve(), (ROOT / args.out).resolve()

    for split in ("train", "val"):
        n_img = n_box = 0
        for lab in sorted((vd / "labels" / split).glob("*.txt")):
            lines = []
            for ln in lab.read_text().splitlines():
                c, *rest = ln.split()
                if int(c) in VISDRONE_MAP:
                    lines.append(" ".join([str(VISDRONE_MAP[int(c)])] + rest))
            img = vd / "images" / split / (lab.stem + ".jpg")
            link(img, out / "real" / "images" / split / img.name)
            (out / "real" / "labels" / split).mkdir(parents=True, exist_ok=True)
            (out / "real" / "labels" / split / lab.name).write_text("\n".join(lines))
            n_img += 1; n_box += len(lines)
        print(f"real {split}: {n_img} images, {n_box} vehicle boxes")

    write_yaml(out / "real.yaml", [str(out / "real/images/train")], [str(out / "real/images/val")])
    if not (sim / "meta.json").exists():
        print("no sim recording yet -> real only")
        return
    meta = json.loads((sim / "meta.json").read_text())
    world = json.loads((ROOT / "sim/gz/worlds/visloc03_real_targets.json").read_text())
    counts = {"train": [0, 0], "test": [0, 0]}
    for r in meta:
        split = "train" if r["true_lon"] < world["lon0"] else "test"
        img = sim / r["file"]
        link(img, out / "sim" / "images" / split / img.name)
        lab = sim / "labels" / img.name.replace(".jpg", ".txt")
        (out / "sim" / "labels" / split).mkdir(parents=True, exist_ok=True)
        txt = lab.read_text() if lab.exists() else ""
        (out / "sim" / "labels" / split / lab.name).write_text(txt)
        counts[split][0] += 1; counts[split][1] += len([x for x in txt.splitlines() if x.strip()])
    print("sim:", {k: f"{v[0]} images, {v[1]} boxes" for k, v in counts.items()})

    write_yaml(out / "real.yaml", [str(out / "real/images/train")], [str(out / "real/images/val")])
    write_yaml(out / "sim.yaml", [str(out / "sim/images/train")], [str(out / "sim/images/test")])
    write_yaml(out / "mixed.yaml", [str(out / "real/images/train"), str(out / "sim/images/train")],
               [str(out / "sim/images/test")])
    # evaluation-only configs
    write_yaml(out / "test_real.yaml", [str(out / "real/images/train")], [str(out / "real/images/val")])
    write_yaml(out / "test_sim.yaml", [str(out / "sim/images/train")], [str(out / "sim/images/test")])
    print("yaml files in", out)


if __name__ == "__main__":
    main()
