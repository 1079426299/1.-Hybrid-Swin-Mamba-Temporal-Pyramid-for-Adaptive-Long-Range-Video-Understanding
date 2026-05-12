import torch
from torch import nn
import torch.nn.functional as F


class TemporalOffsetSampler(nn.Module):
    """Temporal deformable sampler implemented with 1D grid_sample."""

    def __init__(self, channels: int, max_offset: float = 2.0):
        super().__init__()
        self.max_offset = max_offset
        hidden = max(channels // 4, 8)
        self.offset_net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        b, c, t, h, w = x.shape
        temporal_desc = x.mean(dim=(-1, -2))
        offsets = self.offset_net(temporal_desc) * self.max_offset

        base = torch.linspace(-1, 1, t, device=x.device, dtype=x.dtype).view(1, 1, t)
        norm_offsets = offsets / max(t - 1, 1) * 2.0
        grid_t = (base + norm_offsets).clamp(-1, 1)

        x_perm = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t, 1)
        grid_t_expand = grid_t.repeat_interleave(h * w, dim=0).transpose(1, 2)
        grid_x = torch.zeros_like(grid_t_expand)
        grid = torch.stack([grid_x, grid_t_expand], dim=-1)

        sampled = F.grid_sample(x_perm, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.view(b, h, w, c, t).permute(0, 3, 4, 1, 2).contiguous()
        return sampled, offsets.squeeze(1)


class DeformableTemporalPyramid(nn.Module):
    """Adaptive multi-scale temporal alignment and fusion."""

    def __init__(self, channels_list, out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv3d(c, out_channels, kernel_size=1) for c in channels_list])
        self.samplers = nn.ModuleList([TemporalOffsetSampler(out_channels) for _ in channels_list])
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * len(channels_list), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, features):
        target_t, target_h, target_w = features[0].shape[2:]
        aligned = []
        offsets = []
        for feat, proj, sampler in zip(features, self.proj, self.samplers):
            x = proj(feat)
            if x.shape[2:] != (target_t, target_h, target_w):
                x = F.interpolate(x, size=(target_t, target_h, target_w), mode="trilinear", align_corners=False)
            x, off = sampler(x)
            aligned.append(x)
            offsets.append(off)
        fused = self.fuse(torch.cat(aligned, dim=1))
        return fused, aligned, offsets
