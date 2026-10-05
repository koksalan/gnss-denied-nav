"""Chart of the hard-scenario benchmark: share of frames with a usable fix (confident and < 50 m) per scenario.

    python scripts/plot_robustness.py --tag main
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.corruptions import BY_NAME  # noqa: E402

LABELS = {"ft": "DINOv2-base fine-tuned (86 M)", "ft_robust": "+ hard-condition training (86 M)",
          "vits_kd": "distilled ViT-S (22 M)", "mnv3_kd": "distilled MobileNetV3 (5 M)"}
SHORT = {"clean": "clean", "fog": "fog", "low_light": "dusk +\nnoise", "clouds": "35 %\nclouds",
         "motion_blur": "vibration\nblur", "weak_link": "weak link\n1/6 res + JPEG", "overexposure": "over-\nexposure",
         "color_shift": "colour /\nseason", "heading_err": "compass\n+10 deg", "alt_high": "baro alt.\n+20 %",
         "alt_low": "baro alt.\n-20 %", "combined": "combined"}
COLORS = {"ft": "#9ca3af", "ft_robust": "#0e7490", "vits_kd": "#22d3ee", "mnv3_kd": "#f59e0b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--metric", default="usable")
    ap.add_argument("--out", default="docs/robustness.png")
    args = ap.parse_args()
    s = pd.read_csv(ROOT / "outputs" / "robust" / args.tag / "summary.csv")
    models = [m for m in LABELS if m in set(s.model)]
    scen = list(dict.fromkeys(s.scenario))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        v = s[s.model == m].set_index("scenario").loc[scen, args.metric]
        ax.bar([i + (k - (len(models) - 1) / 2) * w for i in range(len(scen))], v * 100, w,
               label=LABELS[m], color=COLORS[m])
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels([SHORT.get(n, BY_NAME[n].label) for n in scen], fontsize=8.5)
    ax.set_ylabel("usable fixes (%)\nconfident and < 50 m")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_title("Real held-out flights, global search, hard conditions applied to the camera frame")
    fig.tight_layout()
    fig.savefig(ROOT / args.out, dpi=140)
    print(args.out)


if __name__ == "__main__":
    main()
