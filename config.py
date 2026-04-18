"""
Configuration for the hierarchical SPDE-PINN system.

Three-level hierarchy:
  Level 3: Correlation field  z(x,t)  -- drives Level 2
  Level 2: Volatility field   y(x,t)  -- drives Level 1
  Level 1: Probability density p(x,t) -- Fokker-Planck equation

Each field obeys a stochastic PDE with white-noise or fBm forcing.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ── Domain ────────────────────────────────────────────────────────
    x_min: float = -3.0
    x_max: float = 3.0
    t_min: float = 0.0
    t_max: float = 1.0

    # ── Finite-difference grid (classical solver & noise field) ──────
    Nx: int = 128
    Nt: int = 512

    # ── Level 3 – correlation field z ────────────────────────────────
    #   dz/dt = D_z z_xx + alpha_z (mu_z - z) + sigma_z xi_z
    D_z: float = 0.05
    alpha_z: float = 2.0
    mu_z: float = 0.0
    sigma_z: float = 0.08
    noise_type_z: str = "white"        # "white" | "fbm"
    hurst_z: float = 0.7

    # ── Level 2 – volatility field y ─────────────────────────────────
    #   dy/dt = D_y y_xx + alpha_y (theta(z) - y) + sigma_y sqrt(y) xi_y
    #   theta(z) = mu_y + beta_yz * z
    D_y: float = 0.02
    alpha_y: float = 3.0
    mu_y: float = 0.20
    beta_yz: float = 0.10
    sigma_y: float = 0.12
    noise_type_y: str = "white"
    hurst_y: float = 0.7

    # ── Level 1 – probability density p (Fokker-Planck) ──────────────
    #   dp/dt = -mu_drift dp/dx + 0.5 d^2(y^2 p)/dx^2 + sigma_p xi_p
    mu_drift: float = 0.0
    sigma_p: float = 0.01
    noise_type_p: str = "white"
    hurst_p: float = 0.7

    # ── Neural-network architecture ──────────────────────────────────
    hidden_layers: int = 4
    hidden_dim: int = 64
    num_fourier_features: int = 32
    fourier_scale: float = 2.0

    # ── Training hyper-parameters ────────────────────────────────────
    num_epochs: int = 15000
    warmup_epochs: int = 2000         # IC+BC+data only (no PDE residual)
    learning_rate: float = 1e-3
    lr_min: float = 1e-6
    batch_coll: int = 2048            # collocation (interior) points
    batch_ic: int = 200               # initial-condition points
    batch_bc: int = 200               # boundary-condition points
    batch_data: int = 512             # data-fitting points
    grad_clip: float = 5.0            # max gradient norm

    # ── Loss weights ─────────────────────────────────────────────────
    w_pde_z: float = 1.0
    w_pde_y: float = 1.0
    w_pde_p: float = 1.0
    w_ic: float = 50.0
    w_bc: float = 10.0
    w_data: float = 5.0
    w_pos_y: float = 0.5              # soft positivity on y
    w_norm_p: float = 0.5             # soft normalisation on p

    # ── Misc ─────────────────────────────────────────────────────────
    seed: int = 42
    output_dir: str = "results"
    log_every: int = 200

    # ── Derived helpers ──────────────────────────────────────────────
    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.Nx - 1)

    @property
    def dt(self) -> float:
        return (self.t_max - self.t_min) / (self.Nt - 1)
