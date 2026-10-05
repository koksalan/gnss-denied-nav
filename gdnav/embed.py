"""DINOv2 global descriptors for place recognition."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoEmbedder(torch.nn.Module):
    """pool='cls' | 'mean' | 'gem' (GeM over patch tokens) | 'cls+gem' (concatenated)."""

    def __init__(self, name: str = "facebook/dinov2-base", pool: str = "cls+gem", gem_p: float = 3.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.pool = pool
        self.gem_p = gem_p
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Bx3xHxW float in [0,1]. Returns L2-normalized BxD."""
        x = (x - self.mean) / self.std
        tok = self.backbone(pixel_values=x).last_hidden_state
        cls, patches = tok[:, 0], tok[:, 1:]
        gem = patches.clamp(min=1e-6).pow(self.gem_p).mean(1).pow(1 / self.gem_p) if "gem" in self.pool else None
        if self.pool == "cls":
            d = cls
        elif self.pool == "mean":
            d = patches.mean(1)
        elif self.pool == "gem":
            d = gem
        else:
            d = torch.cat([F.normalize(cls, dim=-1), F.normalize(gem, dim=-1)], dim=-1)
        return F.normalize(d, dim=-1)


def load_embedder(model: str, pool: str, weights: str | None, device: str) -> torch.nn.Module:
    """DINOv2 teacher (plain state dict / no weights) or a distilled student (checkpoint with its config)."""
    ckpt = torch.load(weights, map_location=device) if weights else None
    if isinstance(ckpt, dict) and "student_config" in ckpt:
        from .student import StudentEmbedder
        m = StudentEmbedder(**ckpt["student_config"], pretrained=False)
        m.load_state_dict(ckpt["state_dict"])
    else:
        m = DinoEmbedder(model, pool=pool)
        if ckpt is not None:
            m.load_state_dict(ckpt)
    return m.to(device).eval()


def to_tensor(batch: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def embed_images(model: DinoEmbedder, images, device: str, batch_size: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch_size):
        x = to_tensor(images[i:i + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy()
