"""Fine-tune DINOv2 for drone->satellite retrieval with a symmetric InfoNCE loss.

Positive pair: north-up drone patch  <->  satellite crop at its GT position (with random offset).
Negatives: the other satellite crops in the batch. Batches are drawn from ONE flight (harder negatives:
same city, same season) and their GT positions are kept far apart, because consecutive drone photos
overlap heavily and would otherwise be false negatives.

Only the last `--train-blocks` transformer blocks are trained; the rest of DINOv2 stays frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdnav.embed import DinoEmbedder  # noqa: E402
from gdnav.evaluation import evaluate_retrieval  # noqa: E402
from gdnav.prepared import PreparedFlight  # noqa: E402
from gdnav.query import circle_mask  # noqa: E402


def far_apart_batches(pf: PreparedFlight, batch: int, min_sep_m: float, rng: random.Random):
    """Shuffle a flight's queries into batches whose GT positions are pairwise >= min_sep_m apart."""
    x, y = pf.ll_to_ov(pf.gt_ll[:, 0], pf.gt_ll[:, 1])
    xy = np.stack([x, y], 1) * pf.gsd                       # meters
    pool = list(range(len(xy)))
    rng.shuffle(pool)
    while len(pool) >= batch // 2:
        chosen, rest = [], []
        for i in pool:
            if len(chosen) < batch and all(np.hypot(*(xy[i] - xy[j])) >= min_sep_m for j in chosen):
                chosen.append(i)
            else:
                rest.append(i)
        if len(chosen) < batch // 2:
            break
        yield chosen
        pool = rest


class Augment(torch.nn.Module):
    """GPU augmentations. Drone side gets haze/blur/color shifts + heading noise; satellite side mild color.

    mode="robust" adds hard flight conditions on the drone side (each with its own probability): plasma clouds and
    cloud shadows, darkness + sensor noise, motion blur, low resolution + JPEG, overexposure, stronger heading
    error and scale error (barometric altitude). Implemented on the 224 px patch with kornia - unlike the test
    scenarios (gdnav/corruptions.py, raw frame, OpenCV) - so the test conditions are not simply memorised.
    """

    def __init__(self, px: int, heading_noise_deg: float, mode: str = "basic"):
        super().__init__()
        self.mode = mode
        self.drone = torch.nn.Sequential(
            K.ColorJiggle(0.3, 0.3, 0.3, 0.03, p=0.9),
            K.RandomGaussianBlur((5, 5), (0.1, 1.5), p=0.3),
            K.RandomRotation(heading_noise_deg, p=1.0),
        )
        if mode == "robust":
            self.hard = torch.nn.Sequential(
                K.RandomAffine(degrees=8.0, scale=(0.8, 1.25), p=0.5),            # compass + baro error
                K.RandomPlasmaShadow(roughness=(0.2, 0.5), shade_intensity=(-0.4, -0.1), shade_quantity=(0.0, 0.4), p=0.25),
                K.RandomPlasmaBrightness(roughness=(0.1, 0.3), intensity=(0.1, 0.35), p=0.2),   # clouds / fog patches
                K.RandomMotionBlur(kernel_size=(5, 11), angle=180.0, direction=(-1.0, 1.0), p=0.25),
                K.RandomGamma(gamma=(1.0, 2.2), gain=(0.4, 1.0), p=0.25),          # dusk
                K.RandomGaussianNoise(std=0.05, p=0.25),
                K.RandomBrightness(brightness=(1.2, 1.6), p=0.15),                 # overexposure
                K.RandomJPEG(jpeg_quality=(10.0, 50.0), p=0.25),
            )
        self.sat = K.ColorJiggle(0.2, 0.2, 0.2, 0.02, p=0.8)
        self.register_buffer("mask", torch.from_numpy(circle_mask(px)).float()[None, None], persistent=False)

    def _lowres(self, q: torch.Tensor, p: float = 0.2) -> torch.Tensor:
        sel = torch.rand(q.shape[0], device=q.device) < p
        if sel.any():
            f = float(torch.empty(1).uniform_(0.15, 0.4))
            small = F.interpolate(q[sel], scale_factor=f, mode="area")
            q = q.clone()
            q[sel] = F.interpolate(small, size=q.shape[-2:], mode="bilinear", align_corners=False)
        return q

    def forward(self, q: torch.Tensor, s: torch.Tensor):
        q = self.drone(q)
        hmax = 0.5 if self.mode == "robust" else 0.35
        haze = torch.rand(q.shape[0], 1, 1, 1, device=q.device) * hmax
        q = q * (1 - haze) + haze * 0.75                     # simple atmospheric haze
        if self.mode == "robust":
            q = self._lowres(self.hard(q).clamp(0, 1))
        return q * self.mask, self.sat(s) * self.mask


def batch_tensors(pf: PreparedFlight, idx: list[int], jitter_m: float, rng: random.Random, device: str):
    q = np.stack([pf.queries[i] for i in idx])
    x, y = pf.ll_to_ov(pf.gt_ll[idx, 0], pf.gt_ll[idx, 1])
    sats = []
    for cx, cy in zip(x, y):
        r, a = jitter_m * math.sqrt(rng.random()) / pf.gsd, rng.random() * 2 * math.pi
        sats.append(pf.crop(cx + r * math.cos(a), cy + r * math.sin(a)))
    to = lambda a: torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float() / 255  # noqa: E731
    return to(q), to(np.stack(sats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="comma-separated flight ids")
    ap.add_argument("--val", required=True, help="flight id used for model selection")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--pool", default="cls+gem")
    ap.add_argument("--train-blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-sep-m", type=float, default=150.0)
    ap.add_argument("--jitter-m", type=float, default=15.0)
    ap.add_argument("--heading-noise", type=float, default=5.0)
    ap.add_argument("--aug", default="basic", choices=["basic", "robust"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/finetune/run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = [PreparedFlight.load(f) for f in args.train.split(",")]
    val = PreparedFlight.load(args.val)
    model = DinoEmbedder(args.model, pool=args.pool).to(device)
    for p in model.parameters():
        p.requires_grad = False
    blocks = model.backbone.encoder.layer
    trainable = list(blocks[-args.train_blocks:]) + [model.backbone.layernorm]
    for m in trainable:
        for p in m.parameters():
            p.requires_grad = True
    log_t = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": args.lr}, {"params": [log_t], "lr": 1e-3}], weight_decay=0.05)
    aug = Augment(train[0].px, args.heading_noise, args.aug).to(device)

    # rough step count for the cosine schedule
    steps_per_epoch = sum(max(1, len(pf.queries) // (args.batch // 2)) for pf in train)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, 1e-3], total_steps=total + 10, pct_start=0.05)

    model.eval()
    base, _, _ = evaluate_retrieval(model, val, device)
    history = [{"epoch": 0, **base}]
    print(f"epoch 0 (pretrained) val: R@1<50m={base['R@1<50m']:.3f} R@5<50m={base['R@5<50m']:.3f}", flush=True)
    best = base["R@1<50m"]
    torch.save({k: v for k, v in model.state_dict().items()}, out / "best.pt")

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        batches = [(pf, b) for pf in train for b in far_apart_batches(pf, args.batch, args.min_sep_m, rng)]
        rng.shuffle(batches)
        for pf, idx in batches:
            q, s = batch_tensors(pf, idx, args.jitter_m, rng, device)
            q, s = aug(q, s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eq, es = model(q), model(s)
            logits = log_t.exp().clamp(max=100) * eq.float() @ es.float().T
            target = torch.arange(len(idx), device=device)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step < total:
                sched.step()
            step += 1
            losses.append(loss.item())
        model.eval()
        res, _, _ = evaluate_retrieval(model, val, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "temp": 1 / log_t.exp().item(), **res})
        print(f"epoch {epoch} loss={np.mean(losses):.3f} batches={len(batches)} ({time.time() - t0:.0f}s) "
              f"val R@1<50m={res['R@1<50m']:.3f} R@5<50m={res['R@5<50m']:.3f} med={res['top1_median_m']:.0f}m",
              flush=True)
        if res["R@1<50m"] > best:
            best = res["R@1<50m"]
            torch.save(model.state_dict(), out / "best.pt")
        (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    print(f"best val R@1<50m = {best:.3f}  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
