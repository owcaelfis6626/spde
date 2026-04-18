"""
Training loop for the coupled hierarchical PINN system.

Two-phase training:
  Phase 1 (warmup): IC + BC + data losses only — get the networks to
      approximate the fields before introducing PDE residuals.
  Phase 2 (full):   All losses including PDE residuals, with linear ramp-up.
"""

import time
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from config import Config
from models import HierarchicalPINNSystem
from noise import NoiseInterpolator
from pde_residuals import (
    compute_all_residuals,
    ic_loss, bc_loss_neumann, bc_loss_dirichlet,
    positivity_penalty, normalisation_penalty,
)
from classical_solver import ic_z_func, ic_y_func, ic_p_func


# ═══════════════════════════════════════════════════════════════════════
#  Sampling helpers
# ═══════════════════════════════════════════════════════════════════════

def _sample_interior(cfg: Config, N: int, device: torch.device):
    x = (torch.rand(N, 1, device=device) * (cfg.x_max - cfg.x_min) + cfg.x_min)
    t = (torch.rand(N, 1, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def _sample_ic(cfg: Config, N: int, device: torch.device):
    x = (torch.rand(N, 1, device=device) * (cfg.x_max - cfg.x_min) + cfg.x_min)
    t = torch.zeros(N, 1, device=device)
    x.requires_grad_(True)
    t.requires_grad_(True)
    x_np = x.detach().cpu().numpy().ravel()
    z0 = torch.tensor(ic_z_func(x_np, cfg), dtype=torch.float32,
                       device=device).unsqueeze(-1)
    y0 = torch.tensor(ic_y_func(x_np, cfg), dtype=torch.float32,
                       device=device).unsqueeze(-1)
    p0 = torch.tensor(ic_p_func(x_np, cfg), dtype=torch.float32,
                       device=device).unsqueeze(-1)
    return x, t, z0, y0, p0


def _sample_bc(cfg: Config, N: int, device: torch.device):
    t = (torch.rand(N, 1, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min)
    t.requires_grad_(True)
    x_lo = torch.full((N, 1), cfg.x_min, device=device, requires_grad=True)
    x_hi = torch.full((N, 1), cfg.x_max, device=device, requires_grad=True)
    return x_lo, x_hi, t


def _sample_data(
    x_grid: np.ndarray, t_grid: np.ndarray,
    Z_ref: np.ndarray, Y_ref: np.ndarray, P_ref: np.ndarray,
    N: int, device: torch.device,
):
    Nx, Nt = Z_ref.shape
    ix = np.random.randint(0, Nx, size=N)
    it = np.random.randint(0, Nt, size=N)
    x = torch.tensor(x_grid[ix], dtype=torch.float32, device=device).unsqueeze(-1)
    t = torch.tensor(t_grid[it], dtype=torch.float32, device=device).unsqueeze(-1)
    x.requires_grad_(True)
    t.requires_grad_(True)
    z_d = torch.tensor(Z_ref[ix, it], dtype=torch.float32, device=device).unsqueeze(-1)
    y_d = torch.tensor(Y_ref[ix, it], dtype=torch.float32, device=device).unsqueeze(-1)
    p_d = torch.tensor(P_ref[ix, it], dtype=torch.float32, device=device).unsqueeze(-1)
    return x, t, z_d, y_d, p_d


# ═══════════════════════════════════════════════════════════════════════
#  Trainer
# ═══════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(
        self,
        cfg: Config,
        system: HierarchicalPINNSystem,
        noise_interp_z: NoiseInterpolator,
        noise_interp_y: NoiseInterpolator,
        noise_interp_p: NoiseInterpolator,
        x_grid: np.ndarray, t_grid: np.ndarray,
        Z_ref: np.ndarray, Y_ref: np.ndarray, P_ref: np.ndarray,
        device: torch.device,
    ):
        self.cfg = cfg
        self.system = system
        self.noise_z = noise_interp_z
        self.noise_y = noise_interp_y
        self.noise_p = noise_interp_p
        self.x_grid = x_grid
        self.t_grid = t_grid
        self.Z_ref = Z_ref
        self.Y_ref = Y_ref
        self.P_ref = P_ref
        self.device = device

        self.optimiser = torch.optim.Adam(system.parameters(), lr=cfg.learning_rate)
        self.scheduler = CosineAnnealingLR(
            self.optimiser, T_max=cfg.num_epochs, eta_min=cfg.lr_min,
        )

        self.history = {
            "total": [], "pde_z": [], "pde_y": [], "pde_p": [],
            "ic": [], "bc": [], "data": [], "pos_y": [], "norm_p": [],
            "lr": [],
        }

    def _step(self, use_pde: bool, pde_ramp: float = 1.0):
        cfg = self.cfg
        dev = self.device

        # ── Sample points ────────────────────────────────────────────
        x_ic, t_ic, z0, y0, p0 = _sample_ic(cfg, cfg.batch_ic, dev)
        x_lo, x_hi, t_bc = _sample_bc(cfg, cfg.batch_bc, dev)
        x_d, t_d, z_d, y_d, p_d = _sample_data(
            self.x_grid, self.t_grid,
            self.Z_ref, self.Y_ref, self.P_ref,
            cfg.batch_data, dev,
        )

        # ── IC ───────────────────────────────────────────────────────
        L_ic = (ic_loss(self.system.net_z, x_ic, t_ic, z0)
                + ic_loss(self.system.net_y, x_ic, t_ic, y0)
                + ic_loss(self.system.net_p, x_ic, t_ic, p0))

        # ── BC ───────────────────────────────────────────────────────
        L_bc = (bc_loss_neumann(self.system.net_z, x_lo, t_bc)
                + bc_loss_neumann(self.system.net_z, x_hi, t_bc)
                + bc_loss_neumann(self.system.net_y, x_lo, t_bc)
                + bc_loss_neumann(self.system.net_y, x_hi, t_bc)
                + bc_loss_dirichlet(self.system.net_p, x_lo, t_bc)
                + bc_loss_dirichlet(self.system.net_p, x_hi, t_bc))

        # ── Data ─────────────────────────────────────────────────────
        z_pred, y_pred, p_pred = self.system.forward_all(x_d, t_d)
        L_data = (torch.mean((z_pred - z_d) ** 2)
                  + torch.mean((y_pred - y_d) ** 2)
                  + torch.mean((p_pred - p_d) ** 2))

        # ── Constraints (reuse data points — cheap) ──────────────────
        L_pos_y = torch.mean(torch.relu(-y_pred) ** 2)
        t_norm = torch.rand(4, 1, device=dev) * (cfg.t_max - cfg.t_min) + cfg.t_min
        L_norm_p = normalisation_penalty(self.system.net_p, t_norm, cfg, n_quad=48)

        # ── PDE residuals ────────────────────────────────────────────
        if use_pde:
            x_c, t_c = _sample_interior(cfg, cfg.batch_coll, dev)
            R_z, R_y, R_p = compute_all_residuals(
                self.system, x_c, t_c,
                self.noise_z, self.noise_y, self.noise_p, cfg,
            )
            L_pde_z = torch.mean(R_z ** 2)
            L_pde_y = torch.mean(R_y ** 2)
            L_pde_p = torch.mean(R_p ** 2)
        else:
            L_pde_z = L_pde_y = L_pde_p = torch.tensor(0.0, device=dev)
            pde_ramp = 0.0

        # ── Total loss ───────────────────────────────────────────────
        loss = (pde_ramp * cfg.w_pde_z * L_pde_z
                + pde_ramp * cfg.w_pde_y * L_pde_y
                + pde_ramp * cfg.w_pde_p * L_pde_p
                + cfg.w_ic * L_ic
                + cfg.w_bc * L_bc
                + cfg.w_data * L_data
                + cfg.w_pos_y * L_pos_y
                + cfg.w_norm_p * L_norm_p)

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.system.parameters(),
                                       max_norm=cfg.grad_clip)
        self.optimiser.step()

        return {
            "total": loss.item(),
            "pde_z": L_pde_z.item(),
            "pde_y": L_pde_y.item(),
            "pde_p": L_pde_p.item(),
            "ic": L_ic.item(),
            "bc": L_bc.item(),
            "data": L_data.item(),
            "pos_y": L_pos_y.item(),
            "norm_p": L_norm_p.item(),
        }

    def train(self):
        cfg = self.cfg
        total_epochs = cfg.num_epochs
        warmup = cfg.warmup_epochs
        ramp_epochs = 1000

        pbar = trange(total_epochs, desc="Training", ncols=120)
        t0 = time.time()

        for epoch in pbar:
            if epoch < warmup:
                losses = self._step(use_pde=False)
                phase = "warmup"
            else:
                progress = min((epoch - warmup) / max(ramp_epochs, 1), 1.0)
                losses = self._step(use_pde=True, pde_ramp=progress)
                phase = "full" if progress >= 1.0 else "ramp"

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            for k, v in losses.items():
                self.history[k].append(v)
            self.history["lr"].append(lr)

            if epoch % cfg.log_every == 0:
                pbar.set_postfix(
                    ph=phase,
                    L=f"{losses['total']:.2e}",
                    Rz=f"{losses['pde_z']:.1e}",
                    Ry=f"{losses['pde_y']:.1e}",
                    Rp=f"{losses['pde_p']:.1e}",
                    dat=f"{losses['data']:.1e}",
                    lr=f"{lr:.1e}",
                )

        elapsed = time.time() - t0
        print(f"\nTraining completed in {elapsed:.1f}s  "
              f"({total_epochs} epochs, {elapsed / total_epochs * 1000:.1f} ms/epoch)")
        return self.history
