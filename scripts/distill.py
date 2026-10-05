"""Distil the fine-tuned DINOv2-base place-recognition model (86 M params, 1536-d) into a small student.

Loss = task + relational KD + feature KD
  * task:        symmetric InfoNCE on (drone patch, satellite crop) pairs, as for the teacher
  * relational:  the student reproduces the teacher's similarity structure over the whole batch
                 (drone patches + their satellite crops + extra random satellite crops), KL on row-wise softmax.
                 This is what retrieval needs: the ranking, not the absolute vectors.
  * feature:     a linear projection of the student descriptor matches the teacher descriptor (cosine);
                 the projection is thrown away after training.
Extra satellite crops cost no labels - the map is free data - and give the student more of the teacher's view
of the map. `--no-teacher` trains the same student with the task loss only (the baseline distillation must beat).

    python scripts/distill.py --student dinov2_small --teacher outputs/finetune/s1_robust/best.pt --out outputs/distill/vits
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.embed import load_embedder  # noqa: E402
from gdnav.evaluation import evaluate_retrieval  # noqa: E402
from gdnav.prepared import PreparedFlight  # noqa: E402
from gdnav.student import StudentEmbedder  # noqa: E402
from scripts.train_finetune import Augment, batch_tensors, far_apart_batches  # noqa: E402


def random_sat(pf: PreparedFlight, n: int, rng: random.Random, device: str) -> torch.Tensor:
    h, w = pf.overview.shape[:2]
    half = pf.px / 2
    crops = [pf.crop(rng.uniform(half, w - half), rng.uniform(half, h - half)) for _ in range(n)]
    return torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="dinov2_small", choices=["dinov2_small", "mobilenetv3"])
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--teacher", default="outputs/finetune/s1_robust/best.pt")
    ap.add_argument("--no-teacher", action="store_true", help="baseline: task loss only")
    ap.add_argument("--train", default="01,02,05,08,09,10")
    ap.add_argument("--val", default="06")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--extra-sat", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="backbone lr (default: 5e-5 ViT, 3e-4 CNN)")
    ap.add_argument("--kd-tau", type=float, default=0.05)
    ap.add_argument("--w-rel", type=float, default=1.0)
    ap.add_argument("--w-feat", type=float, default=1.0)
    ap.add_argument("--aug", default="robust", choices=["basic", "robust"])
    ap.add_argument("--min-sep-m", type=float, default=150.0)
    ap.add_argument("--jitter-m", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    train = [PreparedFlight.load(f) for f in args.train.split(",")]
    val = PreparedFlight.load(args.val)

    student = StudentEmbedder(args.student, args.dim).to(device)
    teacher = None if args.no_teacher else load_embedder("facebook/dinov2-base", "cls+gem", str(ROOT / args.teacher), device)
    if teacher is not None:
        for p in teacher.parameters():
            p.requires_grad = False
        t_dim = teacher(torch.zeros(1, 3, train[0].px, train[0].px, device=device)).shape[1]
        proj = torch.nn.Linear(args.dim, t_dim).to(device)
    else:
        proj = None
    log_t = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    lr = args.lr or (5e-5 if args.student == "dinov2_small" else 3e-4)
    head_params = list(student.head.parameters()) + (list(proj.parameters()) if proj else [])
    groups = [{"params": list(student.backbone.parameters()), "lr": lr},
              {"params": head_params, "lr": 1e-3}, {"params": [log_t], "lr": 1e-3}]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    aug = Augment(train[0].px, 5.0, args.aug).to(device)
    steps_per_epoch = sum(max(1, len(pf.queries) // (args.batch // 2)) for pf in train)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total + 10, pct_start=0.05)

    best, history, step = -1.0, [], 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        t0, logs = time.time(), []
        batches = [(pf, b) for pf in train for b in far_apart_batches(pf, args.batch, args.min_sep_m, rng)]
        rng.shuffle(batches)
        for pf, idx in batches:
            q, s = batch_tensors(pf, idx, args.jitter_m, rng, device)
            q, s = aug(q, s)
            x = random_sat(pf, args.extra_sat, rng, device) if teacher is not None else None
            if x is not None:
                x = aug.sat(x) * aug.mask
            imgs = torch.cat([q, s] + ([x] if x is not None else []))
            n = len(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                es = student(imgs).float()
                if teacher is not None:
                    with torch.no_grad():
                        et = teacher(imgs).float()
            logits = log_t.exp().clamp(max=100) * es[:n] @ es[n:2 * n].T
            target = torch.arange(n, device=device)
            l_task = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            loss, l_rel, l_feat = l_task, torch.zeros(()), torch.zeros(())
            if teacher is not None:
                eye = torch.eye(len(imgs), device=device, dtype=torch.bool)
                st = (et @ et.T / args.kd_tau).masked_fill(eye, -1e4)
                ss = (es @ es.T / args.kd_tau).masked_fill(eye, -1e4)
                l_rel = F.kl_div(F.log_softmax(ss, 1), F.log_softmax(st, 1), log_target=True, reduction="batchmean")
                l_feat = (1 - F.cosine_similarity(proj(es), et, dim=1)).mean()
                loss = l_task + args.w_rel * l_rel + args.w_feat * l_feat
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step < total:
                sched.step()
            step += 1
            logs.append([l_task.item(), float(l_rel.detach()), float(l_feat.detach())])
        student.eval()
        res, _, _ = evaluate_retrieval(student, val, device)
        lt, lr_, lf = np.mean(logs, 0)
        history.append({"epoch": epoch, "task": lt, "rel": lr_, "feat": lf, **res})
        print(f"epoch {epoch} task={lt:.3f} rel={lr_:.3f} feat={lf:.3f} ({time.time() - t0:.0f}s) "
              f"val R@1<50m={res['R@1<50m']:.3f} R@5<50m={res['R@5<50m']:.3f} med={res['top1_median_m']:.0f}m", flush=True)
        if res["R@1<50m"] > best:
            best = res["R@1<50m"]
            torch.save(student.checkpoint(), out / "best.pt")
        (out / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    print(f"best val R@1<50m = {best:.3f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
