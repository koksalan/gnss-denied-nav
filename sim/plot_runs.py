"""Compare GNSS-jamming runs: autopilot position error over time + true ground tracks.

    python sim/plot_runs.py control visual visual_trueatt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"control": "GNSS jammed, no visual GPS", "visual": "visual GPS, image-center + IMU attitude correction",
          "visual_trueatt": "visual GPS, IMU attitude correction (true attitude)",
          "visual_pnp": "visual GPS, PnP pose (no IMU attitude)"}


def load(tag: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "outputs" / "sim_runs" / tag / "log.csv")
    d = d.drop_duplicates("t").sort_values("t")
    d["fix_err_m"] = pd.to_numeric(d.fix_err_m, errors="coerce")
    t_jam = d.t[d.jammed == 1].min()
    d["t_rel"] = d.t - t_jam
    return d


def main(tags: list[str]):
    runs = {t: load(t) for t in tags}
    w = json.loads((ROOT / "sim/gz/worlds/visloc03.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    summary = {}
    for tag, d in runs.items():
        a1.plot(d.t_rel, d.ekf_err_m, label=LABELS.get(tag, tag), lw=1.2)
        m_lat = 111132.92 - 559.82 * np.cos(2 * np.radians(w["lat0"]))
        m_lon = 111412.84 * np.cos(np.radians(w["lat0"]))
        a2.plot((d.true_lon - w["lon0"]) * m_lon, (d.true_lat - w["lat0"]) * m_lat, lw=1, label=LABELS.get(tag, tag))
        j = d[d.jammed == 1]
        summary[tag] = dict(jammed_s=round(float(j.t_rel.max()), 0),
                            ekf_err_median_m=round(float(j.ekf_err_m.median()), 1),
                            ekf_err_p95_m=round(float(j.ekf_err_m.quantile(0.95)), 1),
                            ekf_err_max_m=round(float(j.ekf_err_m.max()), 1),
                            ekf_err_at_end_m=round(float(j.ekf_err_m.iloc[-1]), 1),
                            fix_err_median_m=None if j.fix_err_m.isna().all() else round(float(j.fix_err_m.median()), 1))
    a1.axvline(0, color="k", ls="--", lw=1)
    a1.text(2, a1.get_ylim()[1] * 0.92, "GNSS jammed", fontsize=9)
    a1.set(xlabel="time since jamming (s)", ylabel="autopilot position error (m)", yscale="log",
           title="EKF position error vs. ground truth")
    a1.legend()
    h = 1200
    a2.plot([-h, h, h, -h, -h], [h, h, -h, -h, h], "k--", lw=0.8, label="planned patrol")
    a2.set(xlabel="east (m)", ylabel="north (m)", title="true ground track", aspect="equal")
    a2.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "gnss_jamming.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    (ROOT / "outputs" / "sim_runs" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["control", "visual"])
