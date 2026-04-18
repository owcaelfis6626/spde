"""
Comprehensive visualisation suite for the hierarchical SPDE-PINN results.

Generates:
  1. Field evolution heatmaps (z, y, p) — PINN vs classical
  2. Error heatmaps (absolute difference)
  3. Temporal snapshots at selected times
  4. Training loss curves (total + per-component)
  5. Cross-section plots at fixed x
  6. Summary error statistics
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import Config


# ── Style defaults ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "image.cmap": "RdBu_r",
})


# ═══════════════════════════════════════════════════════════════════════
#  Evaluate PINN on the FD grid
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_pinn_on_grid(system, x_grid, t_grid, device):
    """
    Evaluate the three PINNs on the Cartesian product of x_grid x t_grid.
    Returns Z_pinn, Y_pinn, P_pinn each of shape (Nx, Nt).
    """
    Nx, Nt = len(x_grid), len(t_grid)
    xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")     # (Nx, Nt)
    x_flat = torch.tensor(xx.ravel(), dtype=torch.float32,
                          device=device).unsqueeze(-1)
    t_flat = torch.tensor(tt.ravel(), dtype=torch.float32,
                          device=device).unsqueeze(-1)

    # Evaluate in chunks to avoid OOM on large grids
    chunk = 8192
    Z_list, Y_list, P_list = [], [], []
    for i in range(0, x_flat.shape[0], chunk):
        xc = x_flat[i:i + chunk]
        tc = t_flat[i:i + chunk]
        z, y, p = system.forward_all(xc, tc)
        Z_list.append(z.cpu().numpy())
        Y_list.append(y.cpu().numpy())
        P_list.append(p.cpu().numpy())

    Z = np.concatenate(Z_list).reshape(Nx, Nt)
    Y = np.concatenate(Y_list).reshape(Nx, Nt)
    P = np.concatenate(P_list).reshape(Nx, Nt)
    return Z, Y, P


# ═══════════════════════════════════════════════════════════════════════
#  1. Heatmaps: PINN vs classical  + error
# ═══════════════════════════════════════════════════════════════════════

def _add_colorbar(ax, im):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax)


def plot_field_comparison(
    x_grid, t_grid,
    ref, pinn, label, out_path,
):
    """Side-by-side heatmaps: reference | PINN | absolute error."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    extent = [t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]]

    vmin = min(ref.min(), pinn.min())
    vmax = max(ref.max(), pinn.max())

    im0 = axes[0].imshow(ref, aspect="auto", origin="lower", extent=extent,
                         vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{label} — Classical FD")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
    _add_colorbar(axes[0], im0)

    im1 = axes[1].imshow(pinn, aspect="auto", origin="lower", extent=extent,
                         vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{label} — PINN")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("x")
    _add_colorbar(axes[1], im1)

    err = np.abs(ref - pinn)
    im2 = axes[2].imshow(err, aspect="auto", origin="lower", extent=extent,
                         cmap="hot")
    axes[2].set_title(f"|Error|  (max={err.max():.3e})")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("x")
    _add_colorbar(axes[2], im2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  2. Temporal snapshots
# ═══════════════════════════════════════════════════════════════════════

def plot_snapshots(
    x_grid, t_grid,
    ref, pinn, label, out_path,
    t_fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """Overlay PINN vs classical at selected time fractions."""
    fig, axes = plt.subplots(1, len(t_fracs), figsize=(4 * len(t_fracs), 3.5),
                             sharey=True)
    Nt = len(t_grid)

    for ax, frac in zip(axes, t_fracs):
        idx = min(int(frac * (Nt - 1)), Nt - 1)
        ax.plot(x_grid, ref[:, idx], "k-", lw=1.5, label="Classical")
        ax.plot(x_grid, pinn[:, idx], "r--", lw=1.5, label="PINN")
        ax.set_title(f"t = {t_grid[idx]:.2f}")
        ax.set_xlabel("x")
        ax.legend(fontsize=8)

    axes[0].set_ylabel(label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  3. Training loss curves
# ═══════════════════════════════════════════════════════════════════════

def plot_training_loss(history: dict, out_path: str):
    """Plot total loss + individual components on a log scale."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Total loss
    ax = axes[0]
    ax.semilogy(history["total"], "k-", lw=0.6, alpha=0.6)
    # Smoothed version
    w = min(100, len(history["total"]) // 10)
    if w > 1:
        kernel = np.ones(w) / w
        smoothed = np.convolve(history["total"], kernel, mode="valid")
        ax.semilogy(np.arange(w - 1, w - 1 + len(smoothed)), smoothed, "b-", lw=1.5)
    ax.set_title("Total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # (b) PDE residuals
    ax = axes[1]
    for key, clr, lbl in [("pde_z", "tab:blue", "R_z"),
                           ("pde_y", "tab:orange", "R_y"),
                           ("pde_p", "tab:green", "R_p")]:
        ax.semilogy(history[key], lw=0.5, alpha=0.4, color=clr)
        if w > 1:
            sm = np.convolve(history[key], kernel, mode="valid")
            ax.semilogy(np.arange(w-1, w-1+len(sm)), sm, lw=1.5,
                        color=clr, label=lbl)
    ax.set_title("PDE residuals")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) IC / BC / data / constraints
    ax = axes[2]
    for key, clr, lbl in [("ic", "tab:red", "IC"),
                           ("bc", "tab:purple", "BC"),
                           ("data", "tab:brown", "Data"),
                           ("pos_y", "tab:cyan", "Pos(y)"),
                           ("norm_p", "tab:pink", "Norm(p)")]:
        if key in history and len(history[key]) > 0:
            ax.semilogy(history[key], lw=0.5, alpha=0.4, color=clr)
            if w > 1:
                sm = np.convolve(history[key], kernel, mode="valid")
                ax.semilogy(np.arange(w-1, w-1+len(sm)), sm, lw=1.5,
                            color=clr, label=lbl)
    ax.set_title("Auxiliary losses")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  4. Cross-sections at fixed x
# ═══════════════════════════════════════════════════════════════════════

def plot_cross_sections(
    x_grid, t_grid,
    refs, pinns, labels, out_path,
    x_vals=(0.0,),
):
    """Time traces at fixed spatial locations."""
    nfields = len(labels)
    nxvals = len(x_vals)
    fig, axes = plt.subplots(nfields, nxvals,
                             figsize=(5 * nxvals, 3.5 * nfields),
                             squeeze=False)

    for row, (ref, pinn, lbl) in enumerate(zip(refs, pinns, labels)):
        for col, xv in enumerate(x_vals):
            ix = np.argmin(np.abs(x_grid - xv))
            ax = axes[row, col]
            ax.plot(t_grid, ref[ix, :], "k-", lw=1.2, label="Classical")
            ax.plot(t_grid, pinn[ix, :], "r--", lw=1.2, label="PINN")
            ax.set_title(f"{lbl} at x={x_grid[ix]:.2f}")
            ax.set_xlabel("t")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  5. Error statistics table
# ═══════════════════════════════════════════════════════════════════════

def print_error_stats(x_grid, t_grid, refs, pinns, labels):
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]
    print("\n" + "=" * 62)
    print(f"  {'Field':<12} {'L2-rel':<14} {'Linf':<14} {'RMSE':<14}")
    print("-" * 62)
    for ref, pinn, lbl in zip(refs, pinns, labels):
        err = np.abs(ref - pinn)
        l2_ref = np.sqrt(np.sum(ref ** 2) * dx * dt)
        l2_err = np.sqrt(np.sum(err ** 2) * dx * dt)
        l2_rel = l2_err / max(l2_ref, 1e-15)
        linf = err.max()
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"  {lbl:<12} {l2_rel:<14.4e} {linf:<14.4e} {rmse:<14.4e}")
    print("=" * 62 + "\n")


# ═══════════════════════════════════════════════════════════════════════
#  6. Combined 3-field evolution (single figure)
# ═══════════════════════════════════════════════════════════════════════

def plot_combined_evolution(
    x_grid, t_grid,
    Z_ref, Y_ref, P_ref,
    Z_pinn, Y_pinn, P_pinn,
    out_path,
):
    """3x2 panel: each row is a field (z, y, p); columns are classical / PINN."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    extent = [t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]]

    data = [
        (Z_ref, Z_pinn, "z(x,t) — Correlation"),
        (Y_ref, Y_pinn, "y(x,t) — Volatility"),
        (P_ref, P_pinn, "p(x,t) — Density"),
    ]

    for row, (ref, pinn, title) in enumerate(data):
        vmin = min(ref.min(), pinn.min())
        vmax = max(ref.max(), pinn.max())

        im0 = axes[row, 0].imshow(ref, aspect="auto", origin="lower",
                                   extent=extent, vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"{title} [Classical]")
        axes[row, 0].set_ylabel("x")
        _add_colorbar(axes[row, 0], im0)

        im1 = axes[row, 1].imshow(pinn, aspect="auto", origin="lower",
                                   extent=extent, vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"{title} [PINN]")
        _add_colorbar(axes[row, 1], im1)

    for ax in axes[-1, :]:
        ax.set_xlabel("t")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Master plotting function
# ═══════════════════════════════════════════════════════════════════════

def generate_all_plots(
    cfg: Config,
    system,
    x_grid, t_grid,
    Z_ref, Y_ref, P_ref,
    history: dict,
    device,
):
    """Generate and save all figures."""
    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)

    print("\nEvaluating PINN on the FD grid ...")
    Z_pinn, Y_pinn, P_pinn = evaluate_pinn_on_grid(
        system, x_grid, t_grid, device,
    )

    refs   = [Z_ref,  Y_ref,  P_ref]
    pinns  = [Z_pinn, Y_pinn, P_pinn]
    labels = ["z (correlation)", "y (volatility)", "p (density)"]

    print("\nGenerating plots ...")

    # Combined evolution
    plot_combined_evolution(x_grid, t_grid,
                           Z_ref, Y_ref, P_ref,
                           Z_pinn, Y_pinn, P_pinn,
                           os.path.join(out, "combined_evolution.png"))

    # Per-field comparison
    for ref, pinn, lbl, tag in zip(refs, pinns, labels, ["z", "y", "p"]):
        plot_field_comparison(x_grid, t_grid, ref, pinn, lbl,
                              os.path.join(out, f"field_{tag}.png"))

    # Snapshots
    for ref, pinn, lbl, tag in zip(refs, pinns, labels, ["z", "y", "p"]):
        plot_snapshots(x_grid, t_grid, ref, pinn, lbl,
                       os.path.join(out, f"snapshots_{tag}.png"))

    # Training loss
    plot_training_loss(history, os.path.join(out, "training_loss.png"))

    # Cross-sections
    plot_cross_sections(
        x_grid, t_grid, refs, pinns, labels,
        os.path.join(out, "cross_sections.png"),
        x_vals=(-1.0, 0.0, 1.0),
    )

    # Error statistics
    print_error_stats(x_grid, t_grid, refs, pinns, labels)

    print(f"All figures saved to  {os.path.abspath(out)}/")
    return Z_pinn, Y_pinn, P_pinn
