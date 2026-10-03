"""Fix-confidence features and the learned model (shared by dataset building, training and the onboard node).

Keeping the feature code in ONE place guarantees the model sees the same features in flight as in training.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FEATURES = ["mode_global", "inliers", "rank", "n_matches", "n_query_kp", "inlier_ratio", "second_inliers",
            "inlier_margin", "sim_top1", "sim_margin", "texture", "reproj_px", "tilt_imu", "height_dev", "tilt_dev"]


def fix_features(fix, mode: str, rel_alt_m: float, roll_deg: float, pitch_deg: float) -> dict:
    """Features of one visual fix. `rel_alt_m`, `roll_deg`, `pitch_deg` are the AUTOPILOT's estimates."""
    x = fix.extra
    f = dict(
        mode_global=int(mode == "global"), inliers=fix.inliers, rank=x.get("rank", -1),
        n_matches=x.get("n_matches", 0), n_query_kp=x.get("n_query_kp", 0), inlier_ratio=x.get("inlier_ratio", 0.0),
        second_inliers=x.get("second_inliers", 0), sim_top1=x.get("sim_top1", np.nan),
        sim_margin=x.get("sim_margin", np.nan), texture=x.get("texture", np.nan),
        reproj_px=x.get("reproj_px", np.nan), tilt_imu=math.hypot(roll_deg, pitch_deg),
    )
    f["inlier_margin"] = (fix.inliers - f["second_inliers"]) / max(fix.inliers, 1)
    f["height_dev"] = abs(x["height_m"] - rel_alt_m) if "height_m" in x else np.nan
    f["tilt_dev"] = abs(x["off_nadir_deg"] - f["tilt_imu"]) if "off_nadir_deg" in x else np.nan
    return f


class ConfidenceModel:
    """P(fix error > bad_m) and the predicted 80th-percentile error (reported to the EKF as accuracy)."""

    def __init__(self, path: str | Path):
        import joblib
        d = joblib.load(path)
        self.clf, self.reg, self.features = d["clf"], d["reg"], d["features"]
        self.threshold, self.bad_m = d["threshold"], d["bad_m"]

    def predict(self, feats: dict) -> tuple[float, float]:
        x = np.array([[feats.get(k, np.nan) for k in self.features]], dtype=float)
        p_bad = float(self.clf.predict_proba(x)[0, 1])
        acc = float(np.expm1(self.reg.predict(x)[0]))
        return p_bad, acc
