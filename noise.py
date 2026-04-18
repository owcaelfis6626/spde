"""
Noise generation: space-time white noise and fractional Brownian motion (fBm).

The noise is pre-computed on the finite-difference grid and then bilinearly
interpolated at arbitrary (x, t) query points during PINN training.
"""

import numpy as np
import torch
from scipy.linalg import cholesky

from config import Config


# ═══════════════════════════════════════════════════════════════════════
#  Generation on a regular grid
# ═══════════════════════════════════════════════════════════════════════

def _white_noise_grid(Nx: int, Nt: int, sigma: float,
                      dx: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """
    Space-time white noise  xi(x, t),  discretised so that
    < xi_i^n  xi_j^m > = delta_{ij} delta_{nm} / (dx dt)

    Returns array of shape (Nx, Nt) already scaled by sigma.
    """
    # Standard Gaussian divided by sqrt(dx*dt) gives unit white noise
    raw = rng.standard_normal((Nx, Nt))
    return sigma * raw / np.sqrt(dx * dt)


def _fbm_covariance(n: int, H: float) -> np.ndarray:
    """Covariance matrix of fBm increments  DeltaB_k = B(k dt) - B((k-1) dt)
    (with dt factored out – multiply by dt^{2H} afterwards)."""
    idx = np.arange(n)
    # |i-j+1|^{2H} - 2|i-j|^{2H} + |i-j-1|^{2H}
    d = np.abs(idx[:, None] - idx[None, :]).astype(float)
    C = 0.5 * (np.abs(d + 1) ** (2 * H) - 2 * np.abs(d) ** (2 * H)
               + np.abs(d - 1) ** (2 * H))
    return C


def _fbm_noise_grid(Nx: int, Nt: int, sigma: float,
                     dx: float, dt: float, H: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Fractional Gaussian noise field: at each spatial grid point the temporal
    sequence is an fBm increment series with Hurst parameter H, while
    different spatial locations are independent.

    Returns (Nx, Nt).
    """
    C = _fbm_covariance(Nt, H)
    # Regularise for numerical stability
    C += 1e-10 * np.eye(Nt)
    L = cholesky(C, lower=True)                       # (Nt, Nt)

    Z = rng.standard_normal((Nx, Nt))                 # iid normals
    increments = (Z @ L.T) * (dt ** H)                # (Nx, Nt)

    # Scale by sigma / sqrt(dx) so that the spatial density is correct
    return sigma * increments / np.sqrt(dx)


def generate_noise_field(cfg: Config, level: str,
                         rng: np.random.Generator) -> np.ndarray:
    """Return noise array (Nx, Nt) for a given level ('z', 'y', or 'p')."""
    noise_type = getattr(cfg, f"noise_type_{level}")
    sigma      = getattr(cfg, f"sigma_{level}")
    hurst      = getattr(cfg, f"hurst_{level}")

    if noise_type == "white":
        return _white_noise_grid(cfg.Nx, cfg.Nt, sigma, cfg.dx, cfg.dt, rng)
    elif noise_type == "fbm":
        return _fbm_noise_grid(cfg.Nx, cfg.Nt, sigma, cfg.dx, cfg.dt, hurst, rng)
    else:
        raise ValueError(f"Unknown noise type '{noise_type}'")


# ═══════════════════════════════════════════════════════════════════════
#  Bilinear interpolation of noise at arbitrary (x, t) query points
# ═══════════════════════════════════════════════════════════════════════

class NoiseInterpolator:
    """Holds a pre-computed noise grid and interpolates onto collocation
    points during PINN training (no gradient tracking needed)."""

    def __init__(self, noise_grid: np.ndarray, x_grid: np.ndarray,
                 t_grid: np.ndarray, device: torch.device):
        self.device = device
        # Store as torch tensors (float32) on the target device
        self.noise = torch.tensor(noise_grid, dtype=torch.float32, device=device)
        self.x0, self.x1 = float(x_grid[0]), float(x_grid[-1])
        self.t0, self.t1 = float(t_grid[0]), float(t_grid[-1])
        self.Nx = noise_grid.shape[0]
        self.Nt = noise_grid.shape[1]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x, t : tensors of shape (N,) or (N,1).
        Returns: (N, 1) interpolated noise values.
        """
        x_ = x.detach().view(-1)
        t_ = t.detach().view(-1)

        # Map to fractional grid indices
        ix = (x_ - self.x0) / (self.x1 - self.x0) * (self.Nx - 1)
        it = (t_ - self.t0) / (self.t1 - self.t0) * (self.Nt - 1)

        ix = ix.clamp(0, self.Nx - 1 - 1e-5)
        it = it.clamp(0, self.Nt - 1 - 1e-5)

        ix0 = ix.long()
        it0 = it.long()
        ix1 = (ix0 + 1).clamp(max=self.Nx - 1)
        it1 = (it0 + 1).clamp(max=self.Nt - 1)

        fx = ix - ix0.float()
        ft = it - it0.float()

        v00 = self.noise[ix0, it0]
        v01 = self.noise[ix0, it1]
        v10 = self.noise[ix1, it0]
        v11 = self.noise[ix1, it1]

        val = (v00 * (1 - fx) * (1 - ft)
               + v01 * (1 - fx) * ft
               + v10 * fx * (1 - ft)
               + v11 * fx * ft)
        return val.unsqueeze(-1)
