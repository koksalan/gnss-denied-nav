"""Train a YOLO vehicle detector on one of the prepared datasets (real / sim / mixed).

    python scripts/train_det.py --data ../data/det/real.yaml --name det_real
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    model = YOLO(args.model)
    model.train(data=str((ROOT / args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=str(ROOT / "outputs" / "det"), name=args.name, exist_ok=True, workers=4, seed=0,
                plots=False, verbose=False)


if __name__ == "__main__":
    main()
