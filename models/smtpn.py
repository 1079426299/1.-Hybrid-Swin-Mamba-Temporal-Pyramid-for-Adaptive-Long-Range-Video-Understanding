from typing import Dict, Any
import torch
from torch import nn
import torch.nn.functional as F

from .backbone import HierarchicalVideoBackbone
from .me_cam import MECAM
from .deformable_temporal_pyramid import DeformableTemporalPyramid


class SMTPN(nn.Module):
    """Swin-Mamba Temporal Pyramid Network reference implementation."""

    def __init__(self, num_classes: int, embed_dims=(96, 192, 384, 768),
                 mecam_state_dim: int = 64, pyramid_levels=(1, 2, 3), in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = list(embed_dims)
        self.pyramid_levels = list(pyramid_levels)

        self.backbone = HierarchicalVideoBackbone(in_channels=in_channels, embed_dims=embed_dims)
        self.mecam = nn.ModuleList([MECAM(c, state_dim=mecam_state_dim) for c in embed_dims])

        channels_for_pyramid = [embed_dims[i] for i in self.pyramid_levels]
        fusion_channels = embed_dims[-1]
        self.dtpn = DeformableTemporalPyramid(channels_for_pyramid, out_channels=fusion_channels)

        self.head = nn.Linear(fusion_channels, num_classes)
        self.aux_heads = nn.ModuleList([nn.Linear(fusion_channels, num_classes) for _ in self.pyramid_levels])

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self.backbone(x)
        enhanced = [module(feat) for module, feat in zip(self.mecam, features)]

        selected = [enhanced[i] for i in self.pyramid_levels]
        fused, aligned, offsets = self.dtpn(selected)

        logits = self.head(fused.mean(dim=(2, 3, 4)))
        out = {"logits": logits, "offsets": offsets}
        if return_aux:
            aux_logits = [head(feat.mean(dim=(2, 3, 4))) for feat, head in zip(aligned, self.aux_heads)]
            out["aux_logits"] = aux_logits
        return out


def build_model(cfg: Dict[str, Any]) -> SMTPN:
    model_cfg = cfg.get("model", cfg)
    return SMTPN(
        num_classes=int(model_cfg["num_classes"]),
        embed_dims=tuple(model_cfg.get("embed_dims", [96, 192, 384, 768])),
        mecam_state_dim=int(model_cfg.get("mecam_state_dim", 64)),
        pyramid_levels=tuple(model_cfg.get("pyramid_levels", [1, 2, 3])),
        in_channels=int(cfg.get("dataset", {}).get("channels", 3)),
    )


def semantic_consistency_loss(logits: torch.Tensor, aux_logits, temperature: float = 1.0) -> torch.Tensor:
    if not aux_logits:
        return logits.new_tensor(0.0)
    target = F.softmax(logits.detach() / temperature, dim=-1)
    loss = logits.new_tensor(0.0)
    for aux in aux_logits:
        log_prob = F.log_softmax(aux / temperature, dim=-1)
        loss = loss + F.kl_div(log_prob, target, reduction="batchmean") * (temperature ** 2)
    return loss / len(aux_logits)
