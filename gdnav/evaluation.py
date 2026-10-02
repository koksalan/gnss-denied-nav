"""Retrieval evaluation shared by training (validation) and the benchmark script."""
from __future__ import annotations

import faiss
import numpy as np

from .embed import DinoEmbedder, embed_images
from .geo import haversine_m
from .prepared import PreparedFlight

KS = (1, 5, 10)
THRESH_M = (50, 100)


def recall_table(dists: np.ndarray) -> dict:
    """dists: QxK distances (m) of ranked candidates to ground truth."""
    res = {"top1_median_m": float(np.median(dists[:, 0]))}
    for t in THRESH_M:
        for k in KS:
            res[f"R@{k}<{t}m"] = float((dists[:, :k] < t).any(1).mean())
    return res


def evaluate_retrieval(model: DinoEmbedder, pf: PreparedFlight, device: str, stride_m: float = 50.0,
                       query_idx: np.ndarray | None = None, k: int = max(KS)):
    """Global search over the whole flight map. Returns (metrics, nn indices QxK, tile centers in overview px)."""
    centers = pf.tile_centers(stride_m)
    db = np.concatenate([embed_images(model, b, device) for b in pf.tiles(centers)])
    idx = np.arange(len(pf.queries)) if query_idx is None else query_idx
    q = embed_images(model, [pf.queries[i] for i in idx], device)
    index = faiss.IndexFlatIP(db.shape[1])
    index.add(db.astype(np.float32))
    _, nn = index.search(q.astype(np.float32), k)
    lat, lon = pf.ov_to_ll(centers[nn, 0], centers[nn, 1])
    gt = pf.gt_ll[idx]
    dists = haversine_m(gt[:, None, 0], gt[:, None, 1], lat, lon)
    res = recall_table(dists)
    res.update(n_queries=len(idx), n_tiles=len(centers))
    return res, nn, centers
