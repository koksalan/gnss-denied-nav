"""Size and speed of the place-recognition embedders (teacher vs distilled students).

  * parameters, GMACs per 224 px image (torch flop counter)
  * GPU latency, batch 1, fp16 (median of 200 runs, after warm-up)
  * CPU latency, batch 1, ONNX Runtime fp32 and dynamic INT8 (a proxy for an embedded CPU; the RTX 5090 is not one)
    INT8 fidelity: cosine between fp32 and INT8 descriptors of real drone patches.
    Naive INT8 of every layer breaks DINOv2 (one MLP with outlier activations collapses the descriptor), so the
    quantization is sensitivity-aware: each layer group is quantized alone on calibration patches and groups that
    move the descriptor (cosine < --sens-thr) stay FP32.
  * map database size: tiles of a 10 x 10 km operation area at 50 m stride x descriptor dim x 4 bytes
  * pre-flight time to embed that database on the GPU

    python scripts/bench_embedders.py --models ft_robust=outputs/finetune/s1_robust/best.pt,vits=outputs/distill/vits_kd/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.embed import load_embedder  # noqa: E402
from gdnav.prepared import PreparedFlight  # noqa: E402


def ms(fn, n: int, sync=lambda: None) -> float:
    for _ in range(10):
        fn()
    sync()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def quantize_sensitivity_aware(src: Path, dst: Path, calib: np.ndarray, thr: float, so) -> dict:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    def run(path):
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        e = np.concatenate([sess.run(None, {"image": r[None]})[0] for r in calib])
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def quant(nodes, path):
        quantize_dynamic(str(src), str(path), weight_type=QuantType.QInt8, per_channel=True, nodes_to_quantize=nodes)

    ref = run(src)
    nodes = [n.name for n in onnx.load(str(src)).graph.node if n.op_type in ("MatMul", "Gemm")]
    groups: dict[str, list[str]] = {}
    for n in nodes:                                   # e.g. /backbone/encoder/layer.8/mlp
        groups.setdefault("/".join(n.split("/")[:5]), []).append(n)
    tmp, sens = dst.with_suffix(".tmp.onnx"), {}
    for g, ns in groups.items():
        quant(ns, tmp)
        sens[g] = float(np.median((run(tmp) * ref).sum(1)))
    keep = [n for g, ns in groups.items() if sens[g] >= thr for n in ns]
    quant(keep, dst)
    tmp.unlink(missing_ok=True)
    naive = dst.with_name(dst.stem + "_naive.onnx")
    quant(nodes, naive)
    return dict(fp32_groups=sorted(g for g in groups if sens[g] < thr), n_quantized=len(keep), n_matmul=len(nodes),
                naive_cos=float(np.median((run(naive) * ref).sum(1))), worst_group=min(sens, key=sens.get),
                worst_cos=round(min(sens.values()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--sens-thr", type=float, default=0.998)
    ap.add_argument("--out", default="outputs/robust/bench.json")
    args = ap.parse_args()
    import onnxruntime as ort

    real = PreparedFlight.load("03").queries[::12][:64].astype(np.float32).transpose(0, 3, 1, 2) / 255
    calib = PreparedFlight.load("06").queries[::40][:16].astype(np.float32).transpose(0, 3, 1, 2) / 255  # val flight

    res = {}
    tiles_10km = (10_000 // 50) ** 2
    for spec in args.models.split(","):
        name, w = spec.split("=", 1)
        m = load_embedder("facebook/dinov2-base", "cls+gem", str(ROOT / w) if w else None, "cuda")
        x = torch.rand(1, 3, args.px, args.px, device="cuda")
        with torch.no_grad():
            dim = m(x).shape[1]
            with FlopCounterMode(display=False) as fc:
                m(x)
        params = sum(p.numel() for p in m.parameters())
        sync = torch.cuda.synchronize
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            gpu = ms(lambda: m(x), 200, sync)
            xb = torch.rand(256, 3, args.px, args.px, device="cuda")
            per_batch = ms(lambda: m(xb), 10, sync)
        db_s = per_batch / 1000 * tiles_10km / 256

        onnx_path = ROOT / "outputs" / "robust" / f"{name}.onnx"
        m_cpu = m.float().cpu().eval()
        torch.onnx.export(m_cpu, torch.rand(1, 3, args.px, args.px), str(onnx_path), input_names=["image"],
                          output_names=["descriptor"], opset_version=17, dynamo=False)
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.cpu_threads
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        xn = np.random.rand(1, 3, args.px, args.px).astype(np.float32)
        ref = m_cpu(torch.from_numpy(xn)).detach().numpy()
        out = sess.run(None, {"image": xn})[0]
        cpu = ms(lambda: sess.run(None, {"image": xn}), 30)
        q_path = onnx_path.with_name(f"{name}_int8.onnx")
        qinfo = quantize_sensitivity_aware(onnx_path, q_path, calib, args.sens_thr, so)
        sq = ort.InferenceSession(str(q_path), so, providers=["CPUExecutionProvider"])
        cpu8 = ms(lambda: sq.run(None, {"image": xn}), 30)
        f32 = np.concatenate([sess.run(None, {"image": r[None]})[0] for r in real])
        i8 = np.concatenate([sq.run(None, {"image": r[None]})[0] for r in real])
        cos8 = float(np.median((f32 * i8).sum(1) / np.linalg.norm(f32, axis=1) / np.linalg.norm(i8, axis=1)))
        res[name] = dict(params_M=round(params / 1e6, 1), gmacs=round(fc.get_total_flops() / 2e9, 2), dim=dim,
                         gpu_fp16_ms=round(gpu, 2), cpu_onnx_ms=round(cpu, 1), cpu_int8_ms=round(cpu8, 1),
                         onnx_MB=round(onnx_path.stat().st_size / 1e6, 1), int8_MB=round(q_path.stat().st_size / 1e6, 1),
                         int8_cos_median=round(cos8, 4), int8=qinfo,
                         onnx_max_abs_diff=float(np.abs(out - ref).max()),
                         db_10km_MB=round(tiles_10km * dim * 4 / 1e6, 1), db_10km_build_s=round(db_s, 1))
        print(name, res[name], flush=True)
        del m, m_cpu
        torch.cuda.empty_cache()
    Path(ROOT / args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
