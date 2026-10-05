"""Hard-scenario benchmark: the full localizer (retrieval -> LightGlue -> position) on real held-out flights
under weather, light, sensor, datalink and navigation-sensor errors (gdnav/corruptions.py).

Global search over each flight's whole satellite map (no position prior), every `--every`-th photo.
Several embedders (teacher / robust teacher / distilled students) are compared on exactly the same frames;
the map's SuperPoint features are built once per flight and shared.

    python scripts/eval_robustness.py --models ft=outputs/finetune/s1/best.pt,ft_robust=outputs/finetune/s1_robust/best.pt \
        --tag robust_main
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav import corruptions as C  # noqa: E402
from gdnav.embed import embed_images  # noqa: E402
from gdnav.geo import haversine_m  # noqa: E402
from gdnav.localizer import Localizer, LocalizerConfig  # noqa: E402
from gdnav.prepared import CONFIG_ROOT, PreparedFlight  # noqa: E402
from gdnav.query import Camera, make_query  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def summarize(df: pd.DataFrame, min_inliers: int) -> pd.DataFrame:
    def agg(g):
        conf = g.inliers >= min_inliers
        return pd.Series({
            "n": len(g),
            "R@1<50m": (g.coarse_err_m < 50).mean(),
            "R@10<50m": g.r10_50.mean(),
            "fix_median_m": g.fix_err_m.median(),
            "fix<25m": (g.fix_err_m < 25).mean(),
            "fix<50m": (g.fix_err_m < 50).mean(),
            "confident": conf.mean(),
            "confident_median_m": g.fix_err_m[conf].median() if conf.any() else np.nan,
            # safety: a confident fix that is far off would be fed to the autopilot
            "confident_wrong>100m": int((g.fix_err_m[conf] > 100).sum()),
            "usable": (conf & (g.fix_err_m < 50)).mean(),
            "embed_ms": g.embed_ms.mean(),
        })
    return df.groupby(["model", "scenario"], sort=False).apply(agg, include_groups=False).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", default="03,04,11")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--models", required=True, help="name=weights,... (weights empty = pretrained DINOv2-base)")
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--rerank-k", type=int, default=10)
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    device = "cuda"
    out = ROOT / "outputs" / "robust" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    models = [m.split("=", 1) for m in args.models.split(",")]
    scen = C.SCENARIOS if args.scenarios == "all" else [C.BY_NAME[s] for s in args.scenarios.split(",")]
    rows = []
    for f in args.flights.split(","):
        fl, pf = VisLocFlight(f), PreparedFlight.load(f)
        c0 = Camera.load(CONFIG_ROOT / f"camera_visloc{f}.json")
        cam = Camera(c0.focal_px / 2, c0.heading_sign, c0.heading_offset_deg)   # frames are read at half size
        lat_t, lon_l, lat_b, lon_r = fl.sat.bounds_latlon
        bounds = (min(lat_t, lat_b), min(lon_l, lon_r), max(lat_t, lat_b), max(lon_l, lon_r))
        cfg = LocalizerConfig(patch_m=pf.patch_m, rerank_k=args.rerank_k, min_inliers=args.min_inliers, pose="boresight")
        t0 = time.time()
        locs, mapf = {}, None
        for name, w in models:
            locs[name] = Localizer(fl.sat, cfg, (str(ROOT / w) if w else None), device, bounds_ll=bounds, mapf=mapf)
            mapf = locs[name].mapf
        print(f"flight {f}: {len(locs[models[0][0]].tiles_ll)} tiles, map features {len(mapf.keypoints)} kp "
              f"({time.time() - t0:.0f}s)", flush=True)
        idxs = range(0, len(fl), args.every)
        for n_done, i in enumerate(idxs):
            r = fl.row(i)
            raw = cv2.cvtColor(cv2.imread(str(fl.image_path(i))), cv2.COLOR_BGR2RGB)
            raw = cv2.resize(raw, (raw.shape[1] // 2, raw.shape[0] // 2), interpolation=cv2.INTER_AREA)
            for s in scen:
                img, h, hd = C.apply(s, raw, r.height, r.Phi1, seed=i)
                q = make_query(img, h, hd, cam, cfg.px, cfg.patch_m)
                for name, loc in locs.items():
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    e = torch.from_numpy(embed_images(loc.model, [q], device)).to(device)[0]
                    torch.cuda.synchronize()
                    emb_ms = 1000 * (time.perf_counter() - t1)
                    top = (loc.db @ e).topk(10).indices.cpu().numpy()
                    d = haversine_m(r.lat, r.lon, loc.tiles_ll[top, 0], loc.tiles_ll[top, 1])
                    fix, _ = loc.localize(img, h, hd, cam)
                    rows.append(dict(flight=f, idx=i, scenario=s.name, model=name, coarse_err_m=float(d[0]),
                                     r10_50=bool((d < 50).any()), inliers=fix.inliers, embed_ms=emb_ms,
                                     fix_err_m=float(haversine_m(r.lat, r.lon, fix.lat, fix.lon)) if fix.inliers else np.inf))
            if n_done % 25 == 0:
                print(f"  {f} {n_done + 1}/{len(idxs)} ({time.time() - t0:.0f}s)", flush=True)
        del locs, mapf
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out / "per_query.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_query.csv", index=False)
    summ = summarize(df, args.min_inliers)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(summ)


if __name__ == "__main__":
    main()
