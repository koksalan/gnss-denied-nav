"""Does retrieval survive deployment? Query descriptors from ONNX Runtime (FP32 and sensitivity-aware INT8, CPU)
against the map database computed pre-flight in PyTorch on a GPU - the onboard split of the work.

    python scripts/eval_int8.py --models ft_robust=outputs/finetune/s1_robust/best.pt,mnv3_kd=outputs/distill/mnv3_kd/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.embed import embed_images, load_embedder  # noqa: E402
from gdnav.evaluation import recall_table  # noqa: E402
from gdnav.geo import haversine_m  # noqa: E402
from gdnav.prepared import PreparedFlight  # noqa: E402
from scripts.bench_embedders import quantize_sensitivity_aware  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--flights", default="03,04,11")
    ap.add_argument("--sens-thr", type=float, default=0.998)
    ap.add_argument("--out", default="outputs/robust/int8_retrieval.json")
    args = ap.parse_args()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    calib = PreparedFlight.load("06").queries[::40][:16].astype(np.float32).transpose(0, 3, 1, 2) / 255
    flights = [PreparedFlight.load(f) for f in args.flights.split(",")]
    res = {}
    for spec in args.models.split(","):
        name, w = spec.split("=", 1)
        m = load_embedder("facebook/dinov2-base", "cls+gem", str(ROOT / w), "cuda")
        fp32, int8 = ROOT / "outputs" / "robust" / f"{name}.onnx", ROOT / "outputs" / "robust" / f"{name}_int8.onnx"
        cpu = m.float().cpu().eval()
        torch.onnx.export(cpu, torch.rand(1, 3, 224, 224), str(fp32), input_names=["image"],
                          output_names=["descriptor"], opset_version=17, dynamo=False)
        qinfo = quantize_sensitivity_aware(fp32, int8, calib, args.sens_thr, so)
        m = m.cuda()
        sessions = {p: ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"]) for p, f in
                    (("onnx_fp32", fp32), ("onnx_int8", int8))}
        dists = {k: [] for k in ("torch", "onnx_fp32", "onnx_int8")}
        for pf in flights:
            centers = pf.tile_centers(50.0)
            db = np.concatenate([embed_images(m, b, "cuda") for b in pf.tiles(centers)]).astype(np.float32)
            index = faiss.IndexFlatIP(db.shape[1])
            index.add(db)
            qs = np.asarray(pf.queries)
            qd = {"torch": embed_images(m, list(qs), "cuda")}
            x = qs.astype(np.float32).transpose(0, 3, 1, 2) / 255
            for p, sess in sessions.items():
                qd[p] = np.concatenate([sess.run(None, {"image": x[i:i + 1]})[0] for i in range(len(x))])
            for k, q in qd.items():
                q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
                _, nn = index.search(q, 10)
                lat, lon = pf.ov_to_ll(centers[nn, 0], centers[nn, 1])
                dists[k].append(haversine_m(pf.gt_ll[:, None, 0], pf.gt_ll[:, None, 1], lat, lon))
        res[name] = {k: {m_: round(v, 3) for m_, v in recall_table(np.concatenate(d)).items()
                         if m_ in ("R@1<50m", "R@10<50m", "top1_median_m")} for k, d in dists.items()}
        res[name]["int8"] = qinfo
        print(name, json.dumps(res[name]), flush=True)
        del m, cpu
        torch.cuda.empty_cache()
    (ROOT / args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
