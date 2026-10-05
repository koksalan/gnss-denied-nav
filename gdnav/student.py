"""Small "student" place-recognition embedders, distilled from the fine-tuned DINOv2-base teacher.

  * dinov2_small  - ViT-S/14 (same family as the teacher, 22 M params), cls+GeM pooling like the teacher
  * mobilenetv3   - MobileNetV3-Large (5 M params, ImageNet init), GeM over the last feature map

Both end in a linear head to a compact `dim`-d descriptor (smaller map database: 512 vs 1536 floats per tile).
A checkpoint stores its own config, so `gdnav.embed.load_embedder` can rebuild it from the file alone.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .embed import MEAN, STD

ARCHS = {
    "dinov2_small": "facebook/dinov2-small",
    "mobilenetv3": "mobilenetv3_large_100",
}


def gem(x: torch.Tensor, p: float = 3.0, dim=1) -> torch.Tensor:
    return x.clamp(min=1e-6).pow(p).mean(dim).pow(1 / p)


class StudentEmbedder(torch.nn.Module):
    def __init__(self, arch: str = "dinov2_small", dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.config = dict(arch=arch, dim=dim)
        self.arch = arch
        if arch == "dinov2_small":
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(ARCHS[arch])
            feat = 2 * self.backbone.config.hidden_size
        elif arch == "mobilenetv3":
            import timm
            self.backbone = timm.create_model(ARCHS[arch], pretrained=pretrained, num_classes=0, global_pool="")
            feat = self.backbone.num_features
        else:
            raise ValueError(arch)
        self.head = torch.nn.Linear(feat, dim)
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if self.arch == "dinov2_small":
            tok = self.backbone(pixel_values=x).last_hidden_state
            return torch.cat([F.normalize(tok[:, 0], dim=-1), F.normalize(gem(tok[:, 1:]), dim=-1)], -1)
        fm = self.backbone.forward_features(x)                          # B x C x h x w
        return gem(fm.flatten(2), dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Bx3xHxW float in [0,1]. Returns L2-normalized B x dim."""
        return F.normalize(self.head(self.features(x)), dim=-1)

    def checkpoint(self) -> dict:
        return {"student_config": self.config, "state_dict": self.state_dict()}
