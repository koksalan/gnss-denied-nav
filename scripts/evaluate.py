"""Benchmark: coarse retrieval (+ optional LightGlue refinement) on one or more held-out flights.

Each flight is searched globally over its own satellite map (no position prior).
Writes outputs/eval/<tag>/{summary.json, per_query.csv}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.embed import DinoEmbedder  # noqa: E402
from gdnav.evaluation import evaluate_retrieval  # noqa: E402
from gdnav.geo import haversine_m  # noqa: E402
from gdnav.prepared import CONFIG_ROOT, PreparedFlight  # noqa: E402
from gdnav.query import Camera, make_query  # noqa: E402
from gdnav.rerank import Reranker  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402


def refined_metrics(df: pd.DataFrame, min_inliers: int) -> dict:
    conf = df.inliers >= min_inliers
    return {
        "refined_median_m": float(df.err_m.median()),
        **{f"refined<{t}m": float((df.err_m < t).mean()) for t in (10, 25, 50, 100)},
        "confident_fraction": float(conf.mean()),
        "confident_median_m": float(df.err_m[conf].median()) if conf.any() else None,
        "confident_p95_m": float(df.err_m[conf].quantile(0.95)) if conf.any() else None,
        "confident_wrong>100m": int((df.err_m[conf] > 100).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", required=True, help="comma-separated flight ids")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--pool", default="cls+gem")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--stride-m", type=float, default=50.0)
    ap.add_argument("--rerank-k", type=int, default=0, help="0 = retrieval only")
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    device = "cuda"
    out = Path("outputs") / "eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    model = DinoEmbedder(args.model, pool=args.pool).to(device).eval()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    reranker = Reranker(device) if args.rerank_k else None

    summary, rows = {"args": vars(args), "flights": {}}, []
    for f in args.flights.split(","):
        pf = PreparedFlight.load(f)
        t0 = time.time()
        res, nn, centers = evaluate_retrieval(model, pf, device, args.stride_m)
        res["retrieval_s"] = round(time.time() - t0, 1)
        lat, lon = pf.ov_to_ll(centers[nn, 0], centers[nn, 1])
        df = pd.DataFrame({"flight": f, "file": pf.files,
                           "coarse_err_m": haversine_m(pf.gt_ll[:, 0], pf.gt_ll[:, 1], lat[:, 0], lon[:, 0])})
        if reranker:
            fl, cam = VisLocFlight(f), Camera.load(CONFIG_ROOT / f"camera_visloc{f}.json")
            qpx = int(round(pf.patch_m / reranker.gsd))
            fixes, t_m = [], []
            for i in tqdm(range(len(fl)), desc=f"rerank {f}"):
                r = fl.row(i)
                img = cv2.cvtColor(cv2.imread(str(fl.image_path(i))), cv2.COLOR_BGR2RGB)
                q = make_query(img, r.height, r.Phi1, cam, qpx, pf.patch_m)
                t1 = time.time()
                fixes.append(reranker.localize(q, fl.sat, np.stack([lat[i, :args.rerank_k], lon[i, :args.rerank_k]], 1)))
                t_m.append(time.time() - t1)
            df["lat"], df["lon"] = [x.lat for x in fixes], [x.lon for x in fixes]
            df["inliers"], df["rank"] = [x.inliers for x in fixes], [x.rank for x in fixes]
            df["err_m"] = haversine_m(pf.gt_ll[:, 0], pf.gt_ll[:, 1], df.lat, df.lon)
            res.update(refined_metrics(df, args.min_inliers), match_ms=round(1000 * float(np.mean(t_m)), 1))
        summary["flights"][f] = res
        rows.append(df)
        print(f, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}), flush=True)

    all_df = pd.concat(rows)
    all_df.to_csv(out / "per_query.csv", index=False)
    agg = {"n_queries": len(all_df), "coarse_median_m": float(all_df.coarse_err_m.median()),
           **{f"coarse<{t}m": float((all_df.coarse_err_m < t).mean()) for t in (50, 100)}}
    if reranker:
        agg.update(refined_metrics(all_df, args.min_inliers))
    summary["all"] = agg
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("ALL", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in agg.items()}))


if __name__ == "__main__":
    main()
