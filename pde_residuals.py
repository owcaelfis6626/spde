"""
PDE residual computation for the three-level hierarchical SPDE system.

All spatial / temporal derivatives are computed via torch.autograd.
The Fokker-Planck diffusion term d^2(y^2 p)/dx^2 is expanded analytically
to avoid building a costly autograd graph through two coupled networks:

  d^2(y^2 p)/dx^2 = y^2 p_xx + 4 y y_x p_x + (2 y_x^2 + 2 y y_xx) p
"""

import torch
from config import Config
from noise import NoiseInterpolator


# ── Utility ──────────────────────────────────────────────────────────

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx with graph retained for higher-order derivatives."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


# ═══════════════════════════════════════════════════════════════════════
#  Batched residual computation (all three levels in one function)
# ═══════════════════════════════════════════════════════════════════════

def compute_all_residuals(
    system,
    x: torch.Tensor, t: torch.Tensor,
    noise_z: NoiseInterpolator,
    noise_y: NoiseInterpolator,
    noise_p: NoiseInterpolator,
    cfg: Config,
):
    """
    Evaluate all three PDE residuals at collocation points (x, t).
    Returns (R_z, R_y, R_p) each of shape (N, 1).

    Forward passes are shared and the Fokker-Planck diffusion term
    is expanded analytically for performance.
    """
    # ── Forward passes (shared x, t) ─────────────────────────────────
    z = system.net_z(x, t)
    y = system.net_y(x, t)
    p = system.net_p(x, t)

    # ── Derivatives for z ────────────────────────────────────────────
    z_t  = _grad(z, t)
    z_x  = _grad(z, x)
    z_xx = _grad(z_x, x)

    # ── Derivatives for y ────────────────────────────────────────────
    y_t  = _grad(y, t)
    y_x  = _grad(y, x)
    y_xx = _grad(y_x, x)

    # ── Derivatives for p ────────────────────────────────────────────
    p_t  = _grad(p, t)
    p_x  = _grad(p, x)
    p_xx = _grad(p_x, x)

    # ── Noise values (no grad) ───────────────────────────────────────
    xi_z = noise_z(x, t)
    xi_y = noise_y(x, t)
    xi_p = noise_p(x, t)

    # ── Level 3: z residual ──────────────────────────────────────────
    R_z = z_t - cfg.D_z * z_xx - cfg.alpha_z * (cfg.mu_z - z) - xi_z

    # ── Level 2: y residual ──────────────────────────────────────────
    theta = cfg.mu_y + cfg.beta_yz * z
    sqrt_y = torch.sqrt(torch.clamp(y, min=1e-6))
    R_y = (y_t - cfg.D_y * y_xx
           - cfg.alpha_y * (theta - y)
           - cfg.sigma_y * sqrt_y * xi_y)

    # ── Level 1: p residual (Fokker-Planck) ──────────────────────────
    #   dp/dt = -mu dp/dx + 0.5 d^2(y^2 p)/dx^2 + xi_p
    #
    # Analytical expansion of d^2(y^2 p)/dx^2:
    #   = y^2 p_xx + 4 y y_x p_x + (2 y_x^2 + 2 y y_xx) p
    d2_yy_p = (y ** 2 * p_xx
               + 4.0 * y * y_x * p_x
               + (2.0 * y_x ** 2 + 2.0 * y * y_xx) * p)

    R_p = p_t + cfg.mu_drift * p_x - 0.5 * d2_yy_p - xi_p

    return R_z, R_y, R_p


# ═══════════════════════════════════════════════════════════════════════
#  Boundary / initial-condition losses
# ═══════════════════════════════════════════════════════════════════════

def ic_loss(net, x: torch.Tensor, t0: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
    pred = net(x, t0)
    return torch.mean((pred - target) ** 2)


def bc_loss_neumann(net, x_bnd: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
    u = net(x_bnd, t)
    u_x = _grad(u, x_bnd)
    return torch.mean(u_x ** 2)


def bc_loss_dirichlet(net, x_bnd: torch.Tensor,
                      t: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    u = net(x_bnd, t)
    return torch.mean((u - value) ** 2)


def positivity_penalty(net_y, x: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
    y = net_y(x, t)
    return torch.mean(torch.relu(-y) ** 2)


def normalisation_penalty(
    net_p, t_samples: torch.Tensor, cfg: Config, n_quad: int = 64,
) -> torch.Tensor:
    """(integral p dx - 1)^2 averaged over t_samples, batched."""
    n_t = t_samples.numel()
    x_quad = torch.linspace(cfg.x_min, cfg.x_max, n_quad,
                            device=t_samples.device)
    dx = (cfg.x_max - cfg.x_min) / (n_quad - 1)

    # Build batched (n_t * n_quad, 1) inputs
    x_rep = x_quad.repeat(n_t).unsqueeze(-1)           # (n_t*n_quad, 1)
    t_rep = t_samples.view(-1, 1).repeat_interleave(n_quad, dim=0)  # same
    p_vals = net_p(x_rep, t_rep).view(n_t, n_quad)     # (n_t, n_quad)
    integrals = torch.trapezoid(p_vals, dx=dx, dim=-1)  # (n_t,)
    return torch.mean((integrals - 1.0) ** 2)
