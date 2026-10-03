"""Learned fix-confidence model vs. the hand-written consistency gate.

Two heads on the localizer features (gradient-boosted trees, small and fast on a CPU):
  * classifier  P(error > bad_m)        -> replaces the accept/reject rules
  * quantile regressor q80(error)       -> the accuracy reported to the EKF (replaces 5 + 600 / inliers)
Split is SPATIAL (west half train / east half test) so the model cannot memorize places.

    python scripts/train_conf_model.py --data outputs/conf/dataset_rec_real.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.confidence import FEATURES  # noqa: E402


def rule_gate(df: pd.DataFrame) -> np.ndarray:
    """The hand-written gate from sim/visual_gps.py (normal mode)."""
    return ((df.inliers >= 50) & (df.height_dev.fillna(99) <= 8) & (df.tilt_dev.fillna(99) <= 4)).to_numpy()


def heuristic_acc(df: pd.DataFrame) -> np.ndarray:
    return np.clip(5 + 600 / df.inliers.clip(lower=1), 6, 25).to_numpy()


def operating_point(err: np.ndarray, accept: np.ndarray, bad_m: float) -> dict:
    return dict(accepted=round(float(accept.mean()), 3),
                bad_in_accepted=int((err[accept] > bad_m).sum()),
                bad_rate_accepted=round(float((err[accept] > bad_m).mean()), 4) if accept.any() else None,
                median_err_accepted=round(float(np.median(err[accept])), 1) if accept.any() else None,
                p95_err_accepted=round(float(np.quantile(err[accept], 0.95)), 1) if accept.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/conf/dataset_rec_real.csv")
    ap.add_argument("--bad-m", type=float, default=30.0)
    ap.add_argument("--split-axis", default="east_m")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.data)
    train, test = df[df[args.split_axis] < 0], df[df[args.split_axis] >= 0]
    # degraded mode in flight passes tilt_dev = NaN (EKF attitude untrustworthy): teach the trees that case
    masked = train.copy()
    masked["tilt_dev"] = np.nan
    train = pd.concat([train, masked.sample(frac=0.3, random_state=0)], ignore_index=True)
    Xtr, Xte = train[FEATURES], test[FEATURES]
    ytr, yte = (train.err_m > args.bad_m).astype(int), (test.err_m > args.bad_m).astype(int)

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)
    reg = HistGradientBoostingRegressor(loss="quantile", quantile=0.8, max_iter=300, learning_rate=0.05,
                                        max_leaf_nodes=15, random_state=0)
    reg.fit(Xtr, np.log1p(train.err_m))

    p_bad = clf.predict_proba(Xte)[:, 1]
    acc_pred = np.expm1(reg.predict(Xte))
    err = test.err_m.to_numpy()
    rule = rule_gate(test)

    # learned gate: threshold from OUT-OF-FOLD train predictions (in-sample probabilities are overconfident),
    # chosen so that the accepted bad rate is <= 1 %
    from sklearn.model_selection import cross_val_predict
    p_oof = cross_val_predict(clf, Xtr, ytr, cv=5, method="predict_proba")[:, 1]
    thr = 0.02
    for t in np.linspace(0.02, 0.98, 97):
        acc_tr = p_oof < t
        if acc_tr.any() and (train.err_m.to_numpy()[acc_tr] > args.bad_m).mean() <= 0.01:
            thr = t
    learned = p_bad < thr

    res = {
        "n_train": len(train), "n_test": len(test),
        "bad_rate_test": round(float(yte.mean()), 3),
        "auc": {"learned": round(float(roc_auc_score(yte, p_bad)), 3),
                "inliers_only": round(float(roc_auc_score(yte, -test.inliers)), 3),
                "tilt_dev_only": round(float(roc_auc_score(yte, test.tilt_dev.fillna(99))), 3),
                "height_dev_only": round(float(roc_auc_score(yte, test.height_dev.fillna(99))), 3)},
        "rule_gate": operating_point(err, rule, args.bad_m),
        "learned_gate": dict(threshold=round(float(thr), 3), **operating_point(err, learned, args.bad_m)),
        # same acceptance rate as the rule gate: which one lets fewer bad fixes through?
        "learned_at_rule_acceptance": operating_point(err, p_bad <= np.quantile(p_bad, rule.mean()), args.bad_m),
        "inliers_at_rule_acceptance": operating_point(
            err, test.inliers.to_numpy() >= np.quantile(test.inliers, 1 - rule.mean()), args.bad_m),
        "accuracy_calibration_q80": {
            "learned_coverage": round(float((err <= acc_pred).mean()), 3),
            "heuristic_coverage": round(float((err <= heuristic_acc(test)).mean()), 3),
            "learned_median_m": round(float(np.median(acc_pred)), 1),
        },
    }
    imp = sorted(zip(FEATURES, np.abs(np.corrcoef(np.c_[Xte.fillna(0).to_numpy(), p_bad].T)[-1, :-1])),
                 key=lambda x: -np.nan_to_num(x[1]))
    res["most_correlated_with_p_bad"] = [(k, round(float(v), 2)) for k, v in imp[:6]]
    out = ROOT / "outputs" / "conf"
    joblib.dump(dict(clf=clf, reg=reg, features=FEATURES, threshold=float(thr), bad_m=args.bad_m),
                out / "conf_model.joblib")
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
