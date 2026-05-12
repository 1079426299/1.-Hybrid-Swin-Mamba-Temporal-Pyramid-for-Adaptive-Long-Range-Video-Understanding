import torch
from torch import nn
import torch.nn.functional as F


class MECAM(nn.Module):
    """Mamba-Enhanced Contextual Attention Memory.

    This compact selective state-space style temporal memory is written in pure
    PyTorch so the repository can run without custom CUDA kernels.
    """

    def __init__(self, channels: int, state_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        self.in_proj = nn.Linear(channels, state_dim)
        self.delta_proj = nn.Linear(channels, state_dim)
        self.b_proj = nn.Linear(channels, state_dim)
        self.c_proj = nn.Linear(channels, state_dim)
        self.out_proj = nn.Linear(state_dim, channels)
        self.gate = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        seq = x.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        seq_norm = self.norm(seq)

        u = torch.tanh(self.in_proj(seq_norm))
        delta = F.softplus(self.delta_proj(seq_norm))
        b_t = torch.sigmoid(self.b_proj(seq_norm))
        c_t = torch.sigmoid(self.c_proj(seq_norm))

        state = torch.zeros(b, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for idx in range(t):
            decay = torch.exp(-delta[:, idx, :].clamp(max=10.0))
            state = decay * state + b_t[:, idx, :] * u[:, idx, :]
            outputs.append(c_t[:, idx, :] * state)

        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y)
        y = y.transpose(1, 2).view(b, c, t, 1, 1)
        g = torch.sigmoid(self.gate(seq_norm)).transpose(1, 2).view(b, c, t, 1, 1)
        return x + g * y
