import torch
from torch import nn


class ConvNormAct3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HierarchicalVideoBackbone(nn.Module):
    """Dependency-light hierarchical video backbone.

    It mimics a Video-Swin-style multi-stage hierarchy. For full-scale
    experiments, replace this class with an official Video Swin implementation
    while preserving the output API: list of [B, C_l, T_l, H_l, W_l].
    """

    def __init__(self, in_channels: int = 3, embed_dims=(96, 192, 384, 768)):
        super().__init__()
        dims = list(embed_dims)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=(3, 4, 4), stride=(1, 4, 4), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(dims[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        in_dim = dims[0]
        for idx, out_dim in enumerate(dims):
            stride = (1, 1, 1) if idx == 0 else (1, 2, 2)
            self.stages.append(
                nn.Sequential(
                    ConvNormAct3D(in_dim, out_dim, stride=stride),
                    ConvNormAct3D(out_dim, out_dim, stride=(1, 1, 1)),
                )
            )
            in_dim = out_dim

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
