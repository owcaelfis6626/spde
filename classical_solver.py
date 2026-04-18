"""
Classical finite-difference solver for the coupled SPDE hierarchy.

Scheme
------
  - Diffusion: Crank-Nicolson (implicit, unconditionally stable)
  - Reaction / coupling / noise: explicit Euler
  - Sequential solve per time step:  z -> y -> p

This provides a reference solution to compare against the PINN output.
"""

import numpy as np
from scipy.linalg import solve_banded
from config import Config


# ═══════════════════════════════════════════════════════════════════════
#  Crank-Nicolson helpers for  du/dt = D u_xx  (tridiagonal solver)
# ═══════════════════════════════════════════════════════════════════════

def _cn_matrices(N: int, D: float, dx: float, dt: float):
    """
    Build banded representations of the LHS and RHS matrices for
    Crank-Nicolson applied to  du/dt = D u_xx  with Neumann BCs:
        (I - 0.5*r*A) u^{n+1} = (I + 0.5*r*A) u^n
    where  r = D dt / dx^2  and A is the second-difference matrix.

    Returns (lhs_bands, rhs_builder) where
      lhs_bands : (3, N) array for scipy.linalg.solve_banded
      rhs_builder: function(u_n) -> rhs vector
    """
    r = D * dt / dx ** 2
    half_r = 0.5 * r

    # LHS  =  I - half_r A   (tridiagonal)
    diag = np.ones(N) * (1 + 2 * half_r)
    off  = np.ones(N - 1) * (-half_r)
    # Neumann BCs: ghost-point trick -> modify first / last rows
    diag[0]  = 1 + half_r
    diag[-1] = 1 + half_r

    # Pack in banded format (upper, diag, lower)
    lhs = np.zeros((3, N))
    lhs[0, 1:] = off            # super-diagonal
    lhs[1, :]  = diag           # main diagonal
    lhs[2, :-1] = off           # sub-diagonal

    # RHS multiplier coefficients
    d_rhs = np.ones(N) * (1 - 2 * half_r)
    d_rhs[0]  = 1 - half_r
    d_rhs[-1] = 1 - half_r
    o_rhs = half_r

    def rhs_mult(u):
        v = d_rhs * u
        v[:-1] += o_rhs * u[1:]
        v[1:]  += o_rhs * u[:-1]
        return v

    return lhs, rhs_mult


# ═══════════════════════════════════════════════════════════════════════
#  Initial conditions
# ═══════════════════════════════════════════════════════════════════════

def _ic_z(x: np.ndarray, cfg: Config) -> np.ndarray:
    L = cfg.x_max - cfg.x_min
    return 0.3 * np.cos(np.pi * (x - cfg.x_min) / L)


def _ic_y(x: np.ndarray, cfg: Config) -> np.ndarray:
    L = cfg.x_max - cfg.x_min
    return cfg.mu_y + 0.05 * np.cos(2 * np.pi * (x - cfg.x_min) / L)


def _ic_p(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Normalised Gaussian centred at zero."""
    sig = 0.5
    p = np.exp(-x ** 2 / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi))
    return p


# ═══════════════════════════════════════════════════════════════════════
#  Full solver
# ═══════════════════════════════════════════════════════════════════════

def solve_classical(
    cfg: Config,
    noise_z: np.ndarray,
    noise_y: np.ndarray,
    noise_p: np.ndarray,
):
    """
    Solve the three-level SPDE system on a uniform (x, t) grid.

    Parameters
    ----------
    cfg : Config
    noise_z, noise_y, noise_p : (Nx, Nt) pre-computed noise fields

    Returns
    -------
    x_grid : (Nx,) spatial grid
    t_grid : (Nt,) temporal grid
    Z, Y, P : (Nx, Nt) solution arrays
    """
    Nx, Nt = cfg.Nx, cfg.Nt
    dx, dt = cfg.dx, cfg.dt
    x_grid = np.linspace(cfg.x_min, cfg.x_max, Nx)
    t_grid = np.linspace(cfg.t_min, cfg.t_max, Nt)

    # Allocate solution arrays
    Z = np.zeros((Nx, Nt))
    Y = np.zeros((Nx, Nt))
    P = np.zeros((Nx, Nt))

    # Initial conditions
    Z[:, 0] = _ic_z(x_grid, cfg)
    Y[:, 0] = _ic_y(x_grid, cfg)
    P[:, 0] = _ic_p(x_grid, cfg)

    # Crank-Nicolson for z and y diffusion
    lhs_z, rhs_z = _cn_matrices(Nx, cfg.D_z, dx, dt)
    lhs_y, rhs_y = _cn_matrices(Nx, cfg.D_y, dx, dt)

    # ── Time stepping ────────────────────────────────────────────────
    for n in range(Nt - 1):
        # ·· Level 3: z ··
        z_n = Z[:, n]
        reaction_z = cfg.alpha_z * (cfg.mu_z - z_n) + noise_z[:, n]
        rhs_vec_z = rhs_z(z_n) + dt * reaction_z
        Z[:, n + 1] = solve_banded((1, 1), lhs_z, rhs_vec_z)

        # ·· Level 2: y  (uses z^{n+1}) ··
        y_n = Y[:, n]
        z_new = Z[:, n + 1]
        theta = cfg.mu_y + cfg.beta_yz * z_new
        sqrt_y = np.sqrt(np.maximum(y_n, 1e-10))
        reaction_y = (cfg.alpha_y * (theta - y_n)
                      + cfg.sigma_y * sqrt_y * noise_y[:, n])
        rhs_vec_y = rhs_y(y_n) + dt * reaction_y
        Y[:, n + 1] = solve_banded((1, 1), lhs_y, rhs_vec_y)
        # Enforce non-negativity
        Y[:, n + 1] = np.maximum(Y[:, n + 1], 0.0)

        # ·· Level 1: p  (Fokker-Planck, uses y^{n+1}) ··
        p_n = P[:, n]
        y_new = Y[:, n + 1]

        # Advection:  -mu_drift dp/dx   (central difference, upwind not needed for small mu)
        dp_dx = np.zeros(Nx)
        dp_dx[1:-1] = (p_n[2:] - p_n[:-2]) / (2 * dx)

        # Diffusion of (y^2 p):  0.5 d^2(y^2 p)/dx^2
        yy_p = y_new ** 2 * p_n
        d2_yy_p = np.zeros(Nx)
        d2_yy_p[1:-1] = (yy_p[2:] - 2 * yy_p[1:-1] + yy_p[:-2]) / dx ** 2

        rhs_p = (-cfg.mu_drift * dp_dx
                 + 0.5 * d2_yy_p
                 + noise_p[:, n])

        P[:, n + 1] = p_n + dt * rhs_p

        # Dirichlet BCs for p
        P[0, n + 1] = 0.0
        P[-1, n + 1] = 0.0

        # Re-normalise p (soft: avoid large drifts)
        integral = np.trapezoid(P[:, n + 1], x_grid)
        if integral > 1e-12:
            P[:, n + 1] /= integral

    return x_grid, t_grid, Z, Y, P


# ── Expose initial-condition functions for the PINN ──────────────────

def ic_z_func(x: np.ndarray, cfg: Config) -> np.ndarray:
    return _ic_z(x, cfg)

def ic_y_func(x: np.ndarray, cfg: Config) -> np.ndarray:
    return _ic_y(x, cfg)

def ic_p_func(x: np.ndarray, cfg: Config) -> np.ndarray:
    return _ic_p(x, cfg)
