"""
Physics-Informed Neural Network architectures for the three SPDE levels.

Architecture per field:
  (x, t) ──▶ Fourier-feature embedding ──▶ MLP with residual connections ──▶ scalar output

The three networks are independent function approximators; coupling happens
only through the PDE-residual loss (see pde_residuals.py).
"""

import math
import torch
import torch.nn as nn

from config import Config


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier features  gamma(v) = [cos(2pi B v), sin(2pi B v)]
    to combat spectral bias in PINNs (Tancik et al., 2020)."""

    def __init__(self, in_dim: int = 2, num_features: int = 48,
                 scale: float = 2.0):
        super().__init__()
        B = torch.randn(in_dim, num_features) * scale
        self.register_buffer("B", B)
        self.out_dim = 2 * num_features

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * v @ self.B          # (N, num_features)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class ResidualBlock(nn.Module):
    """Pre-activation residual block: activation ──▶ linear ──▶ activation ──▶ linear + skip."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + h


class PINN(nn.Module):
    """Single PINN mapping (x, t) -> scalar field value."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = FourierFeatureEmbedding(
            in_dim=2,
            num_features=cfg.num_fourier_features,
            scale=cfg.fourier_scale,
        )
        layers = []
        # Input projection
        layers.append(nn.Linear(self.embed.out_dim, cfg.hidden_dim))
        # Residual blocks
        for _ in range(cfg.hidden_layers):
            layers.append(ResidualBlock(cfg.hidden_dim))
        # Final activation + output
        layers.append(nn.Tanh())
        layers.append(nn.Linear(cfg.hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x, t : (N, 1) tensors with requires_grad=True for autograd.
        Returns: (N, 1) field values.
        """
        inp = torch.cat([x, t], dim=-1)              # (N, 2)
        h = self.embed(inp)                           # (N, 2*n_features)
        return self.net(h)                            # (N, 1)


class HierarchicalPINNSystem(nn.Module):
    """Holds the three PINNs (z, y, p) as a single nn.Module so that
    a single optimiser can be used for joint training."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net_z = PINN(cfg)
        self.net_y = PINN(cfg)
        self.net_p = PINN(cfg)

    def forward_all(self, x: torch.Tensor, t: torch.Tensor):
        """Evaluate all three fields at the same (x, t) points."""
        z = self.net_z(x, t)
        y = self.net_y(x, t)
        p = self.net_p(x, t)
        return z, y, p
