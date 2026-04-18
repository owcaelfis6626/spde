#!/usr/bin/env python3
"""
Hierarchical SPDE-PINN System
==============================

Three-level stochastic PDE hierarchy solved jointly by Physics-Informed
Neural Networks, with comparison against a classical finite-difference solver.

  Level 3 (z):  dz/dt = D_z z_xx + alpha_z(mu_z - z)         + sigma_z xi_z
  Level 2 (y):  dy/dt = D_y y_xx + alpha_y(theta(z) - y)     + sigma_y sqrt(y) xi_y
  Level 1 (p):  dp/dt = -mu dp/dx + 0.5 d^2(y^2 p)/dx^2     + sigma_p xi_p

Usage:
    python main.py                         # full run with defaults
    python main.py --epochs 5000           # quick test
    python main.py --noise fbm --hurst 0.7 # fractional Brownian motion on all levels
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from config import Config
from noise import generate_noise_field, NoiseInterpolator
from models import HierarchicalPINNSystem
from classical_solver import solve_classical
from trainer import Trainer
from visualization import generate_all_plots


def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical SPDE-PINN solver")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of training epochs")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--noise", choices=["white", "fbm"], default=None,
                   help="Set noise type for ALL levels")
    p.add_argument("--hurst", type=float, default=None,
                   help="Hurst parameter for fBm (all levels)")
    p.add_argument("--Nx", type=int, default=None,
                   help="Spatial grid points")
    p.add_argument("--Nt", type=int, default=None,
                   help="Temporal grid points")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--hidden-layers", type=int, default=None)
    return p.parse_args()


def apply_overrides(cfg: Config, args):
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.noise is not None:
        cfg.noise_type_z = args.noise
        cfg.noise_type_y = args.noise
        cfg.noise_type_p = args.noise
    if args.hurst is not None:
        cfg.hurst_z = args.hurst
        cfg.hurst_y = args.hurst
        cfg.hurst_p = args.hurst
    if args.Nx is not None:
        cfg.Nx = args.Nx
    if args.Nt is not None:
        cfg.Nt = args.Nt
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output is not None:
        cfg.output_dir = args.output
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.hidden_layers is not None:
        cfg.hidden_layers = args.hidden_layers


def main():
    args = parse_args()
    cfg = Config()
    apply_overrides(cfg, args)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Reproducibility ──────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: Nx={cfg.Nx}, Nt={cfg.Nt}, epochs={cfg.num_epochs}, "
          f"lr={cfg.learning_rate}")
    print(f"Noise:  z={cfg.noise_type_z}, y={cfg.noise_type_y}, "
          f"p={cfg.noise_type_p}")
    if "fbm" in (cfg.noise_type_z, cfg.noise_type_y, cfg.noise_type_p):
        print(f"Hurst:  z={cfg.hurst_z}, y={cfg.hurst_y}, p={cfg.hurst_p}")

    # ── Step 1: Generate noise fields ────────────────────────────────
    print("\n[1/4] Generating noise fields ...")
    t0 = time.time()
    noise_z_grid = generate_noise_field(cfg, "z", rng)
    noise_y_grid = generate_noise_field(cfg, "y", rng)
    noise_p_grid = generate_noise_field(cfg, "p", rng)
    print(f"  Done in {time.time()-t0:.2f}s  "
          f"(grid shape: {noise_z_grid.shape})")

    # ── Step 2: Classical finite-difference solver ───────────────────
    print("\n[2/4] Running classical FD solver ...")
    t0 = time.time()
    x_grid, t_grid, Z_ref, Y_ref, P_ref = solve_classical(
        cfg, noise_z_grid, noise_y_grid, noise_p_grid,
    )
    print(f"  Done in {time.time()-t0:.2f}s")
    print(f"  z range: [{Z_ref.min():.4f}, {Z_ref.max():.4f}]")
    print(f"  y range: [{Y_ref.min():.4f}, {Y_ref.max():.4f}]")
    print(f"  p range: [{P_ref.min():.4f}, {P_ref.max():.4f}]")

    # ── Step 3: Build PINN system and train ──────────────────────────
    print("\n[3/4] Building and training PINN system ...")
    system = HierarchicalPINNSystem(cfg).to(device)
    n_params = sum(p.numel() for p in system.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Build noise interpolators (for evaluating noise at collocation points)
    noise_interp_z = NoiseInterpolator(noise_z_grid, x_grid, t_grid, device)
    noise_interp_y = NoiseInterpolator(noise_y_grid, x_grid, t_grid, device)
    noise_interp_p = NoiseInterpolator(noise_p_grid, x_grid, t_grid, device)

    trainer = Trainer(
        cfg, system,
        noise_interp_z, noise_interp_y, noise_interp_p,
        x_grid, t_grid, Z_ref, Y_ref, P_ref,
        device,
    )
    history = trainer.train()

    # ── Step 4: Visualise ────────────────────────────────────────────
    print("\n[4/4] Generating visualisations ...")
    generate_all_plots(
        cfg, system,
        x_grid, t_grid, Z_ref, Y_ref, P_ref,
        history, device,
    )

    # Save the trained model
    model_path = os.path.join(cfg.output_dir, "model.pt")
    torch.save(system.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
