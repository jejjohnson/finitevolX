# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Elliptic Solvers on Different Geometries
#
# This notebook demonstrates finitevolX's elliptic solver stack by solving the
# **Helmholtz equation**
#
# $$
# (\nabla^2 - \lambda)\,\psi = f
# $$
#
# on four progressively harder domain geometries.  For each geometry we solve a
# manufactured problem, visualise the RHS / solution / residual error triplet,
# and collect timing data for a final accuracy-vs-speed comparison.
#
# ## Discrete 5-point stencil
#
# On a uniform grid with spacing $\Delta x, \Delta y$ the Laplacian is
# approximated by the standard 5-point stencil:
#
# $$
# \nabla^2 \psi \approx
#   \frac{\psi_{i+1,j} - 2\psi_{i,j} + \psi_{i-1,j}}{\Delta x^2}
# + \frac{\psi_{i,j+1} - 2\psi_{i,j} + \psi_{i,j-1}}{\Delta y^2}
# $$
#
# so the discrete Helmholtz operator reads
# $A\psi = \nabla^2_h \psi - \lambda\,\psi$.
#
# ## Solver taxonomy
#
# | Solver | Mechanism | Handles masks? | Variable coeff? | Complexity |
# |--------|-----------|:-:|:-:|-----------|
# | **Spectral DST** | Eigenvalue division in frequency space | No | No | $O(N \log N)$ |
# | **Capacitance matrix** | Sherman-Morrison-Woodbury correction | Yes (few bdry pts) | No | $O(N \log N + B^2)$ |
# | **CG + spectral PC** | Krylov iteration, spectral preconditioner | Yes | No | $O(k \cdot N \log N)$ |
# | **Multigrid** | Recursive V-cycle coarsening / smoothing | Yes | Yes | $O(N)$ |
# | **MG + CG** | MG V-cycle as CG preconditioner | Yes | Yes | $O(k \cdot N)$ |

# %%
from __future__ import annotations

from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import finitevolx as fvx

# Internal import: _apply_operator is needed to compute the multigrid residual
# because the multigrid discretisation (with mask-aware boundary treatment)
# differs from fvx.masked_laplacian.  There is no public equivalent yet.
from finitevolx._src.solvers.multigrid import _apply_operator

IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "demo_solvers"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Shared Setup
#
# ### Grid parameters
#
# We use a **64 x 64** grid on the unit square $[0,1]^2$.  64 is a power of
# two, which keeps multigrid coarsening clean (no fractional grid sizes) and
# is small enough for interactive exploration while large enough for
# representative timing.
#
# ### Helmholtz parameter
#
# We choose $\lambda = 4$.  A positive $\lambda$ shifts the operator away
# from the Poisson null-space, making the system strictly negative-definite
# and well-conditioned.  $\lambda = 4$ is large enough to show the damping
# effect on the solution amplitude but small enough that the system is still
# dominated by the Laplacian.
#
# ### RHS field
#
# A single-mode sinusoidal $f(x,y) = \sin(\pi x') \sin(\pi y')$ (where
# $x', y'$ are normalised to vanish at the boundaries) is smooth, symmetric,
# and has an analytic eigenfunction expansion — ideal for verifying solvers.

# %%
Ny, Nx = 64, 64
Lx, Ly = 1.0, 1.0
dx, dy = Lx / Nx, Ly / Ny
lambda_ = 4.0

j_idx = jnp.arange(Ny)[:, None]
i_idx = jnp.arange(Nx)[None, :]
rhs = jnp.sin(jnp.pi * (j_idx + 1) / (Ny + 1)) * jnp.sin(
    jnp.pi * (i_idx + 1) / (Nx + 1)
)

Y, X = np.mgrid[:Ny, :Nx]

print(f"Grid: {Ny}x{Nx},  dx={dx:.4f},  dy={dy:.4f},  lambda={lambda_}")

# %% [markdown]
# ### Plotting and timing helpers
#
# `plot_triplet` produces the standard 3-panel figure used throughout:
# **RHS | Solution | Error**, where "Error" is the operator residual
# $f - A\hat\psi$ restricted to wet cells.
#
# `time_fn` JIT-compiles a zero-argument callable, warms it up, then
# averages wall-clock time over several calls.


# %%
def _erode_mask(mask: np.ndarray) -> np.ndarray:
    """Erode a binary mask by 1 cell (min-pool with 3x3 kernel).

    This removes the outermost wet layer — i.e. coast-adjacent cells
    where the discrete Laplacian stencil reads land zeros, producing
    spurious residual values that dominate the error colorbar.
    """
    from scipy.ndimage import minimum_filter

    return minimum_filter(mask, size=3)


def plot_triplet(rhs_field, sol_field, err_field, mask, suptitle, solver_info, fname):
    """Save and show a 3-panel RHS | Solution | Error figure."""
    rhs_np = np.where(mask > 0.5, np.asarray(rhs_field), np.nan)
    sol_np = np.where(mask > 0.5, np.asarray(sol_field), np.nan)

    # For the error panel, erode the mask by 1 cell.  At boundary-adjacent
    # cells the Laplacian stencil reads land zeros, creating O(eps) residual
    # artefacts that visually dominate the colorbar.  Eroding hides this
    # boundary layer so the plot shows the true interior error structure.
    interior_mask = _erode_mask(mask)
    err_np = np.where(interior_mask > 0.5, np.asarray(err_field), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    im0 = axes[0].imshow(rhs_np, origin="lower", cmap="RdBu_r", interpolation="nearest")
    axes[0].set_title("RHS  $f$", fontsize=11)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(sol_np, origin="lower", cmap="RdBu_r", interpolation="nearest")
    axes[1].set_title(f"Solution  ({solver_info})", fontsize=11)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    err_abs_max = np.nanmax(np.abs(err_np)) if np.any(np.isfinite(err_np)) else 1.0
    if err_abs_max == 0.0:
        err_abs_max = 1.0
    im2 = axes[2].imshow(
        err_np,
        origin="lower",
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-err_abs_max,
        vmax=err_abs_max,
    )
    axes[2].set_title("Error  $f - A\\hat{\\psi}$", fontsize=11)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(IMG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {fname}")


def time_fn(fn, warmup=2, repeats=5):
    """JIT-compile, warm up, then time *repeats* calls (seconds)."""
    jitted = jax.jit(fn)
    for _ in range(warmup):
        jitted().block_until_ready()
    t0 = time.perf_counter()
    for _ in range(repeats):
        jitted().block_until_ready()
    return (time.perf_counter() - t0) / repeats


def rel_residual(sol, rhs_loc, matvec, interior_mask=None):
    """||rhs - A(sol)|| / ||rhs||, optionally restricted to interior."""
    residual = rhs_loc - matvec(sol)
    if interior_mask is not None:
        residual = residual * interior_mask
        rhs_loc = rhs_loc * interior_mask
    rhs_norm = float(jnp.linalg.norm(rhs_loc))
    if rhs_norm == 0.0:
        return 0.0
    return float(jnp.linalg.norm(residual)) / rhs_norm


# Results accumulator: geometry → solver → {time_ms, rel_residual, label}
results: dict[str, dict] = {}

# %% [markdown]
# ## 2. Domain Geometries Overview
#
# We test four domains of increasing difficulty.
#
# ```
#  Rectangle        Basin             Circle            Notch
#  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
#  │██████████│    │░░░░░░░░░░│    │    ██    │    │░░░░░░░░░░│
#  │██████████│    │░████████░│    │  ██████  │    │░████████░│
#  │██████████│    │░████████░│    │ ████████ │    │░████████░│
#  │██████████│    │░████████░│    │  ██████  │    │░░░░█████░│
#  │██████████│    │░████████░│    │    ██    │    │░░░░█████░│
#  └──────────┘    │░░░░░░░░░░│    └──────────┘    │░████████░│
#                  └──────────┘                    │░░░░░░░░░░│
#                                                  └──────────┘
#  ██ = wet (mask=1)   ░ = land (mask=0)
# ```
#
# - **Rectangle** — full grid, no masking.  Spectral solver applies directly.
# - **Basin** — rectangular ocean with a 4-cell land border on all sides.
# - **Circle** — disk inscribed in the grid (radius = 0.4 N).
# - **Notch** — rectangle with thick walls and a notch cut from the left,
#   combined with a spatially varying coefficient $c(x,y)$.

# %%
mask_rect = np.ones((Ny, Nx))

mask_basin = np.ones((Ny, Nx))
mask_basin[:4, :] = mask_basin[-4:, :] = 0.0
mask_basin[:, :4] = mask_basin[:, -4:] = 0.0

mask_circle = ((X - Nx / 2) ** 2 + (Y - Ny / 2) ** 2 < (0.4 * Ny) ** 2).astype(float)

mask_notch = np.ones((Ny, Nx))
mask_notch[:8, :] = mask_notch[-8:, :] = 0.0
mask_notch[:, :8] = mask_notch[:, -8:] = 0.0
mask_notch[20:44, :20] = 0.0

geometries = {
    "Rectangle": mask_rect,
    "Basin": mask_basin,
    "Circle": mask_circle,
    "Notch\n(variable coeff)": mask_notch,
}

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, (name, mask) in zip(axes, geometries.items(), strict=True):
    ax.imshow(
        mask, origin="lower", cmap="Blues", interpolation="nearest", vmin=0, vmax=1
    )
    wet = int(mask.sum())
    ax.set_title(f"{name}\n({wet} wet cells)", fontsize=11)
    ax.axis("off")
fig.suptitle("Domain Geometries", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(IMG_DIR / "geometries.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved geometries.png")

# %% [markdown]
# ![Domain geometries](../images/demo_solvers/geometries.png)

# %% [markdown]
# ## 3. Rectangle — Spectral DST, CG, Multigrid
#
# The simplest geometry: no land mask, pure Dirichlet boundary conditions.
# Three solvers compete:
#
# ### Spectral DST (exact)
#
# The Discrete Sine Transform diagonalises the 5-point Laplacian with
# homogeneous Dirichlet BCs.  The Helmholtz solve reduces to pointwise
# division by known eigenvalues in frequency space:
#
# $$
# \hat\psi_{k,l} = \frac{\hat{f}_{k,l}}
#   {\mu_k + \mu_l - \lambda},
# \qquad
# \mu_k = \frac{2}{\Delta x^2}\bigl(\cos(k\pi / (N+1)) - 1\bigr)
# $$
#
# Cost is $O(N \log N)$ (two DSTs) with **zero** iteration — the answer is
# exact up to floating-point rounding.
#
# ### CG with spectral preconditioner
#
# Conjugate Gradient treats the Helmholtz operator as a black-box
# matrix-vector product (Krylov method).  The spectral preconditioner
# approximately inverts each CG step via a DST solve, giving rapid
# convergence.
#
# ### Multigrid
#
# Geometric multigrid recursively smooths on a hierarchy of coarsened grids
# (V-cycles).  Each V-cycle costs $O(N)$; 8 cycles suffice for good accuracy
# on a 64x64 grid.

# %%
print("Rectangle:")
mask_rect_jnp = jnp.array(mask_rect)
interior_rect = jnp.zeros((Ny, Nx)).at[1:-1, 1:-1].set(1.0)

# ── Spectral (exact) ──
sol_sp_rect = fvx.solve_helmholtz_dst(rhs, dx, dy, lambda_)
t_sp_rect = time_fn(lambda: fvx.solve_helmholtz_dst(rhs, dx, dy, lambda_))
A_rect_sp = lambda x: fvx.masked_laplacian(x, mask_rect_jnp, dx, dy, lambda_=lambda_)
err_sp_rect = rhs - A_rect_sp(sol_sp_rect)

plot_triplet(
    rhs,
    sol_sp_rect,
    err_sp_rect,
    mask_rect,
    "Rectangle — Spectral DST (exact)",
    "exact, DST",
    "solver_rect_spectral.png",
)

# %% [markdown]
# ![Rectangle: Spectral solver](../images/demo_solvers/solver_rect_spectral.png)

# %%
# ── CG + spectral preconditioner ──
A_rect = lambda x: fvx.masked_laplacian(x, mask_rect_jnp, dx, dy, lambda_=lambda_)
pc_rect = fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="dst")
sol_cg_rect, info_cg_rect = fvx.solve_cg(
    A_rect, rhs, preconditioner=pc_rect, rtol=1e-10, atol=1e-10
)
t_cg_rect = time_fn(
    lambda: fvx.solve_cg(A_rect, rhs, preconditioner=pc_rect, rtol=1e-10, atol=1e-10)[0]
)
rr_cg_rect = rel_residual(sol_cg_rect, rhs, A_rect, interior_mask=interior_rect)
err_cg_rect = rhs - A_rect(sol_cg_rect)

plot_triplet(
    rhs,
    sol_cg_rect,
    err_cg_rect,
    mask_rect,
    f"Rectangle — CG + spectral PC ({info_cg_rect.iterations} iters)",
    f"CG, {info_cg_rect.iterations} iters",
    "solver_rect_cg.png",
)

# %% [markdown]
# ![Rectangle: CG solver](../images/demo_solvers/solver_rect_cg.png)

# %%
# ── Multigrid (8 V-cycles) ──
mg_rect = fvx.build_multigrid_solver(mask_rect, dx, dy, lambda_=lambda_, n_cycles=8)
sol_mg_rect = mg_rect(rhs)
t_mg_rect = time_fn(lambda: mg_rect(rhs))
A_mg_rect = lambda u: _apply_operator(u, mg_rect.levels[0])
rr_mg_rect = rel_residual(sol_mg_rect, rhs, A_mg_rect, interior_mask=interior_rect)
err_mg_rect = rhs - A_mg_rect(sol_mg_rect)

plot_triplet(
    rhs,
    sol_mg_rect,
    err_mg_rect,
    mask_rect,
    "Rectangle — Multigrid (8 V-cycles)",
    "MG, 8 V-cycles",
    "solver_rect_mg.png",
)

# %% [markdown]
# ![Rectangle: Multigrid solver](../images/demo_solvers/solver_rect_mg.png)

# %%
results["Rectangle"] = {
    "Spectral": {"time_ms": t_sp_rect * 1000, "rel_residual": 0.0, "label": "exact"},
    "CG": {
        "time_ms": t_cg_rect * 1000,
        "rel_residual": rr_cg_rect,
        "label": f"{info_cg_rect.iterations} iters",
    },
    "Multigrid": {
        "time_ms": t_mg_rect * 1000,
        "rel_residual": rr_mg_rect,
        "label": "8 V-cyc",
    },
}

print(f"  Spectral: {t_sp_rect * 1000:.2f} ms")
print(f"  CG:       {t_cg_rect * 1000:.2f} ms, {info_cg_rect.iterations} iters")
print(f"  MG:       {t_mg_rect * 1000:.2f} ms, rel residual = {rr_mg_rect:.2e}")

# %% [markdown]
# ## 4. Basin with Land Border — Capacitance, CG, Multigrid
#
# A rectangular ocean basin with a 4-cell land border on all edges.  The
# interior wet region is slightly smaller than the full grid.
#
# ### Capacitance matrix (Sherman-Morrison-Woodbury)
#
# The capacitance method extends the spectral solver to masked domains.  It
# works by:
#
# 1. Solving the full-grid (unmasked) spectral problem
# 2. Precomputing Green's functions at each boundary point $b \in B$
# 3. Correcting with a dense $B \times B$ capacitance system
#
# The offline cost is $O(B \cdot N \log N)$ for $B$ boundary points plus an
# $O(B^3)$ factorisation.  Each subsequent solve is $O(N \log N + B^2)$.
# This is fast when $B \ll N$ (near-rectangular masks).
#
# ### CG + spectral preconditioner
#
# Same Krylov approach as for the rectangle, but using `fvx.masked_laplacian`
# which zeroes out land cells.
#
# ### Multigrid
#
# Geometric multigrid handles the mask natively by restricting smoothing and
# residual computation to wet cells.

# %%
print("\nBasin:")
mask_basin_jnp = jnp.array(mask_basin)
rhs_basin = rhs * mask_basin_jnp
A_basin = lambda x: fvx.masked_laplacian(x, mask_basin_jnp, dx, dy, lambda_=lambda_)

# ── Capacitance ──
cap = fvx.build_capacitance_solver(
    mask_basin.astype(bool), dx, dy, lambda_=lambda_, base_bc="dst"
)
sol_cap = cap(rhs_basin)
t_cap = time_fn(lambda: cap(rhs_basin))
err_cap = rhs_basin - A_basin(sol_cap)

plot_triplet(
    rhs_basin,
    sol_cap,
    err_cap,
    mask_basin,
    "Basin — Capacitance Matrix (direct)",
    "capacitance, direct",
    "solver_basin_cap.png",
)
print(f"  Capacitance: {t_cap * 1000:.2f} ms, boundary pts = {len(cap._j_b)}")

# %% [markdown]
# ![Basin: Capacitance solver](../images/demo_solvers/solver_basin_cap.png)

# %%
# ── CG + spectral preconditioner ──
pc_basin = fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="dst")
sol_cg_basin, info_cg_basin = fvx.solve_cg(
    A_basin, rhs_basin, preconditioner=pc_basin, rtol=1e-10, atol=1e-10
)
sol_cg_basin = sol_cg_basin * mask_basin_jnp
t_cg_basin = time_fn(
    lambda: (
        fvx.solve_cg(
            A_basin, rhs_basin, preconditioner=pc_basin, rtol=1e-10, atol=1e-10
        )[0]
        * mask_basin_jnp
    )
)
rr_cg_basin = rel_residual(
    sol_cg_basin, rhs_basin, A_basin, interior_mask=mask_basin_jnp
)
err_cg_basin = (rhs_basin - A_basin(sol_cg_basin)) * mask_basin_jnp

plot_triplet(
    rhs_basin,
    sol_cg_basin,
    err_cg_basin,
    mask_basin,
    f"Basin — CG + spectral PC ({info_cg_basin.iterations} iters)",
    f"CG, {info_cg_basin.iterations} iters",
    "solver_basin_cg.png",
)

# %% [markdown]
# ![Basin: CG solver](../images/demo_solvers/solver_basin_cg.png)

# %%
# ── Multigrid (8 V-cycles) ──
mg_basin = fvx.build_multigrid_solver(mask_basin, dx, dy, lambda_=lambda_, n_cycles=8)
sol_mg_basin = mg_basin(rhs_basin)
t_mg_basin = time_fn(lambda: mg_basin(rhs_basin))
A_mg_basin = lambda u: _apply_operator(u, mg_basin.levels[0])
rr_mg_basin = rel_residual(
    sol_mg_basin, rhs_basin, A_mg_basin, interior_mask=mask_basin_jnp
)
err_mg_basin = (rhs_basin - A_mg_basin(sol_mg_basin)) * mask_basin_jnp

plot_triplet(
    rhs_basin,
    sol_mg_basin,
    err_mg_basin,
    mask_basin,
    "Basin — Multigrid (8 V-cycles)",
    "MG, 8 V-cycles",
    "solver_basin_mg.png",
)

# %% [markdown]
# ![Basin: Multigrid solver](../images/demo_solvers/solver_basin_mg.png)

# %%
results["Basin"] = {
    "Capacitance": {"time_ms": t_cap * 1000, "rel_residual": 0.0, "label": "direct"},
    "CG": {
        "time_ms": t_cg_basin * 1000,
        "rel_residual": rr_cg_basin,
        "label": f"{info_cg_basin.iterations} iters",
    },
    "Multigrid": {
        "time_ms": t_mg_basin * 1000,
        "rel_residual": rr_mg_basin,
        "label": "8 V-cyc",
    },
}

print(f"  CG:       {t_cg_basin * 1000:.2f} ms, {info_cg_basin.iterations} iters")
print(f"  MG:       {t_mg_basin * 1000:.2f} ms, rel residual = {rr_mg_basin:.2e}")

# %% [markdown]
# ## 5. Circular Basin — CG, Multigrid
#
# A circular ocean basin (radius = 0.4 N cells) inscribed in the grid.  The
# capacitance method becomes expensive here because the number of boundary
# points $B$ scales with the circumference, making the $O(B^3)$ factorisation
# dominant.  Instead we compare:
#
# ### CG with spectral preconditioner (Krylov method)
#
# Conjugate Gradient iterates $x_{k+1} = x_k + \alpha_k p_k$ in a Krylov
# subspace $\mathcal{K}_k(A, r_0)$.  The spectral preconditioner applies an
# approximate inverse via a full-grid DST solve at each step, reducing the
# effective condition number.
#
# ### Multigrid standalone
#
# The multigrid hierarchy coarsens the circular mask along with the grid.  At
# each level, smoothing and restriction/prolongation respect the mask
# boundaries, so no CG wrapper is needed.

# %%
print("\nCircle:")
mask_circle_jnp = jnp.array(mask_circle)
rhs_circle = rhs * mask_circle_jnp
A_circle = lambda x: fvx.masked_laplacian(x, mask_circle_jnp, dx, dy, lambda_=lambda_)

# ── CG + spectral preconditioner ──
pc_circle = fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="dst")
sol_cg_circle, info_cg_circle = fvx.solve_cg(
    A_circle, rhs_circle, preconditioner=pc_circle, rtol=1e-10, atol=1e-10
)
sol_cg_circle = sol_cg_circle * mask_circle_jnp
t_cg_circle = time_fn(
    lambda: (
        fvx.solve_cg(
            A_circle, rhs_circle, preconditioner=pc_circle, rtol=1e-10, atol=1e-10
        )[0]
        * mask_circle_jnp
    )
)
rr_cg_circle = rel_residual(
    sol_cg_circle, rhs_circle, A_circle, interior_mask=mask_circle_jnp
)
err_cg_circle = (rhs_circle - A_circle(sol_cg_circle)) * mask_circle_jnp

plot_triplet(
    rhs_circle,
    sol_cg_circle,
    err_cg_circle,
    mask_circle,
    f"Circle — CG + spectral PC ({info_cg_circle.iterations} iters)",
    f"CG, {info_cg_circle.iterations} iters",
    "solver_circle_cg.png",
)

# %% [markdown]
# ![Circle: CG solver](../images/demo_solvers/solver_circle_cg.png)

# %%
# ── Multigrid (8 V-cycles) ──
mg_circle = fvx.build_multigrid_solver(mask_circle, dx, dy, lambda_=lambda_, n_cycles=8)
sol_mg_circle = mg_circle(rhs_circle)
t_mg_circle = time_fn(lambda: mg_circle(rhs_circle))
A_mg_circle = lambda u: _apply_operator(u, mg_circle.levels[0])
rr_mg_circle = rel_residual(
    sol_mg_circle, rhs_circle, A_mg_circle, interior_mask=mask_circle_jnp
)
err_mg_circle = (rhs_circle - A_mg_circle(sol_mg_circle)) * mask_circle_jnp

plot_triplet(
    rhs_circle,
    sol_mg_circle,
    err_mg_circle,
    mask_circle,
    "Circle — Multigrid (8 V-cycles)",
    "MG, 8 V-cycles",
    "solver_circle_mg.png",
)

# %% [markdown]
# ![Circle: Multigrid solver](../images/demo_solvers/solver_circle_mg.png)

# %%
results["Circle"] = {
    "CG": {
        "time_ms": t_cg_circle * 1000,
        "rel_residual": rr_cg_circle,
        "label": f"{info_cg_circle.iterations} iters",
    },
    "Multigrid": {
        "time_ms": t_mg_circle * 1000,
        "rel_residual": rr_mg_circle,
        "label": "8 V-cyc",
    },
}

print(f"  CG: {t_cg_circle * 1000:.2f} ms, {info_cg_circle.iterations} iters")
print(f"  MG: {t_mg_circle * 1000:.2f} ms, rel residual = {rr_mg_circle:.2e}")

# %% [markdown]
# ## 6. Notch Domain with Variable Coefficient — Multigrid, MG+CG
#
# The hardest case: a spatially varying diffusion coefficient $c(x,y)$ on an
# irregular masked domain.  Only multigrid handles both features natively.
#
# We solve the **variable-coefficient Helmholtz equation**:
#
# $$
# \nabla \cdot \bigl(c(x,y)\,\nabla \psi\bigr) - \lambda\,\psi = f
# $$
#
# where $c(x,y) = 1 + 0.8\sin(2\pi x / N_x)$ varies sinusoidally in $x$.
#
# The domain is a rectangle with 8-cell walls on all sides and a rectangular
# notch (rows 20-44, columns 0-20) cut from the left:
#
# ```
#   ░░░░░░░░░░░░░░░░░░░░
#   ░████████████████████░
#   ░████████████████████░
#   ░░░░░░███████████████░
#   ░░░░░░███████████████░
#   ░████████████████████░
#   ░████████████████████░
#   ░░░░░░░░░░░░░░░░░░░░░
# ```
#
# ### Multigrid standalone (10 V-cycles)
#
# The variable coefficient $c(x,y)$ enters the operator through face-averaged
# diffusivities.  Multigrid coarsens both the mask and the coefficient field.
# We use 10 V-cycles (more than the 8 used for constant-coefficient cases)
# because the variable coefficient increases the condition number.
#
# ### MG-preconditioned CG
#
# For maximum accuracy, use the multigrid solver as a preconditioner inside
# CG.  Each CG step applies one multigrid V-cycle as an approximate inverse,
# and CG refines to the requested tolerance.
#
# **Important:** The CG operator must use the same discretisation as the
# multigrid (`_apply_operator`), not `fvx.masked_laplacian`, because the
# multigrid boundary treatment differs.

# %%
print("\nNotch (variable coeff):")
coeff = 1.0 + 0.8 * np.sin(2 * np.pi * X / Nx)
mask_notch_jnp = jnp.array(mask_notch)
rhs_notch = rhs * mask_notch_jnp

mg_notch = fvx.build_multigrid_solver(
    mask_notch, dx, dy, lambda_=lambda_, coeff=coeff, n_cycles=10
)
A_notch = lambda x: _apply_operator(x, mg_notch.levels[0])

# ── Multigrid standalone (10 V-cycles) ──
sol_mg_notch = mg_notch(rhs_notch)
t_mg_notch = time_fn(lambda: mg_notch(rhs_notch))
rr_mg_notch = rel_residual(
    sol_mg_notch, rhs_notch, A_notch, interior_mask=mask_notch_jnp
)
err_mg_notch = (rhs_notch - A_notch(sol_mg_notch)) * mask_notch_jnp

plot_triplet(
    rhs_notch,
    sol_mg_notch,
    err_mg_notch,
    mask_notch,
    "Notch — Multigrid standalone (10 V-cycles)",
    "MG, 10 V-cycles",
    "solver_notch_mg.png",
)

# %% [markdown]
# ![Notch: Multigrid solver](../images/demo_solvers/solver_notch_mg.png)

# %%
# ── MG-preconditioned CG ──
pc_mg_notch = fvx.make_multigrid_preconditioner(mg_notch)
sol_mgcg_notch, info_mgcg_notch = fvx.solve_cg(
    A_notch, rhs_notch, preconditioner=pc_mg_notch, rtol=1e-12, atol=1e-12
)
sol_mgcg_notch = sol_mgcg_notch * mask_notch_jnp
t_mgcg_notch = time_fn(
    lambda: (
        fvx.solve_cg(
            A_notch, rhs_notch, preconditioner=pc_mg_notch, rtol=1e-12, atol=1e-12
        )[0]
        * mask_notch_jnp
    )
)
rr_mgcg_notch = rel_residual(
    sol_mgcg_notch, rhs_notch, A_notch, interior_mask=mask_notch_jnp
)
err_mgcg_notch = (rhs_notch - A_notch(sol_mgcg_notch)) * mask_notch_jnp

plot_triplet(
    rhs_notch,
    sol_mgcg_notch,
    err_mgcg_notch,
    mask_notch,
    f"Notch — MG+CG ({info_mgcg_notch.iterations} iters)",
    f"MG+CG, {info_mgcg_notch.iterations} iters",
    "solver_notch_mgcg.png",
)

# %% [markdown]
# ![Notch: MG+CG solver](../images/demo_solvers/solver_notch_mgcg.png)

# %%
results["Notch\n(variable coeff)"] = {
    "Multigrid": {
        "time_ms": t_mg_notch * 1000,
        "rel_residual": rr_mg_notch,
        "label": "10 V-cyc",
    },
    "MG+CG": {
        "time_ms": t_mgcg_notch * 1000,
        "rel_residual": rr_mgcg_notch,
        "label": f"{info_mgcg_notch.iterations} iters",
    },
}

diff_mg_mgcg = float(
    jnp.linalg.norm(sol_mgcg_notch - sol_mg_notch) / jnp.linalg.norm(sol_mgcg_notch)
)
print(f"  MG:    {t_mg_notch * 1000:.2f} ms, rel residual = {rr_mg_notch:.2e}")
print(
    f"  MG+CG: {t_mgcg_notch * 1000:.2f} ms, {info_mgcg_notch.iterations} iters, "
    f"rel residual = {rr_mgcg_notch:.2e}"
)
print(f"  Relative difference MG vs MG+CG: {diff_mg_mgcg:.2e}")

# %% [markdown]
# ## 7. Accuracy & Timing Comparison
#
# The bar chart below compares **relative residual** (left, log scale) and
# **JIT-compiled solve time** (right) across all geometry / solver
# combinations.  Timing uses the warmup + repeated-call approach: each
# solver is JIT-compiled, called twice to warm up, then timed over 5
# repeated calls with `block_until_ready()`.

# %%
print("\nSummary chart:")
geom_order = list(results.keys())
geom_shorts = [g.split("\n")[0] for g in geom_order]

colors = {
    "Spectral": "#2196F3",
    "Capacitance": "#4CAF50",
    "CG": "#FF9800",
    "Multigrid": "#9C27B0",
    "MG+CG": "#E91E63",
}

x_positions: list[float] = []
bar_colors: list[str] = []
times_list: list[float] = []
residuals_list: list[float] = []
solver_names: list[str] = []
group_width = 0.8

for gi, geom in enumerate(geom_order):
    solvers = list(results[geom].keys())
    n_s = len(solvers)
    bar_w = group_width / n_s
    for si, sname in enumerate(solvers):
        x_positions.append(gi + (si - (n_s - 1) / 2) * bar_w)
        bar_colors.append(colors.get(sname, "#888888"))
        times_list.append(results[geom][sname]["time_ms"])
        residuals_list.append(results[geom][sname]["rel_residual"])
        solver_names.append(sname)

fig, (ax_res, ax_time) = plt.subplots(1, 2, figsize=(16, 5))

# ── Timing bars ──
bars = ax_time.bar(
    x_positions,
    times_list,
    width=group_width / 3.5,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.5,
)
for bar, t in zip(bars, times_list, strict=True):
    ax_time.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{t:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax_time.set_xticks(range(len(geom_shorts)))
ax_time.set_xticklabels(geom_shorts, fontsize=10)
ax_time.set_ylabel("Time (ms)", fontsize=11)
ax_time.set_title("Solve Time (JIT-compiled, 64x64)", fontsize=12)

# ── Residual bars ──
residuals_plot = [max(r, 1e-15) for r in residuals_list]
ax_res.bar(
    x_positions,
    residuals_plot,
    width=group_width / 3.5,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.5,
)
ax_res.set_yscale("log")
ax_res.set_xticks(range(len(geom_shorts)))
ax_res.set_xticklabels(geom_shorts, fontsize=10)
ax_res.set_ylabel("Relative Residual ||r|| / ||b||", fontsize=11)
ax_res.set_title("Accuracy (Relative Residual Norm)", fontsize=12)
ax_res.set_ylim(bottom=1e-16)

# ── Legend ──
legend_entries = []
seen: set[str] = set()
for sname in solver_names:
    if sname not in seen:
        seen.add(sname)
        legend_entries.append(Patch(facecolor=colors.get(sname, "#888"), label=sname))
ax_res.legend(handles=legend_entries, loc="upper left", fontsize=9)
ax_time.legend(handles=legend_entries, loc="upper left", fontsize=9)

plt.tight_layout()
fig.savefig(IMG_DIR / "accuracy_timing.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved accuracy_timing.png")

# %% [markdown]
# ![Accuracy and timing comparison](../images/demo_solvers/accuracy_timing.png)

# %% [markdown]
# ## 8. Inhomogeneous Boundary Conditions (Known Boundary Values)
#
# All solvers above assume **homogeneous** Dirichlet BCs ($\psi = 0$ on the
# boundary).  In practice, boundary values are often **known but non-zero** —
# e.g., prescribed SSH from observations, reanalysis (ERA5), or a coarser
# model providing boundary forcing.
#
# The standard approach is the **lifting trick**: decompose
#
# $$
# \psi = \psi_{\text{lift}} + \psi_{\text{hom}}
# $$
#
# where $\psi_{\text{lift}}$ is *any* function that matches the prescribed
# boundary data $g$, and $\psi_{\text{hom}}$ solves the **corrected** equation
# with **zero** BCs:
#
# $$
# (A - \lambda)\,\psi_{\text{hom}} = f - (A - \lambda)\,\psi_{\text{lift}},
# \qquad \psi_{\text{hom}} = 0 \;\text{on boundary}
# $$
#
# The simplest $\psi_{\text{lift}}$ is just the boundary values themselves,
# placed in the ghost cells (or mask-boundary cells) and zero in the interior.
# The Laplacian of this "shell" provides the correction to the RHS.
#
# ### Example: prescribed sinusoidal SSH on basin walls
#
# We solve the same Helmholtz problem on the basin domain, but now with
# non-zero Dirichlet data: $\psi = g$ on the boundary where
# $g = 0.1 \sin(2\pi y / L_y)$ (mimicking a prescribed SSH gradient from
# a parent model or observations).

# %%
# --- Build the lifting function ---
# The key insight: the discrete Laplacian stencil at a wet cell reads its
# neighbours regardless of wet/dry status.  For homogeneous BCs, land
# neighbours are 0.  For inhomogeneous BCs, we put the prescribed values g
# at the DRY cells adjacent to the ocean — these act as ghost values that
# the stencil reads, just like ghost cells in the C-grid convention.
from scipy.ndimage import binary_dilation

wet = mask_basin.astype(bool)
# Dilate the wet region by 1 cell to find the land cells that touch ocean
wet_dilated = binary_dilation(wet)
# Dry boundary cells = the 1-cell-wide land ring adjacent to ocean
dry_boundary = wet_dilated & ~wet

# Prescribed boundary data: g(y) = 0.1 * sin(2pi * y / Ny)
# This mimics, e.g., a prescribed SSH gradient from a parent model.
g_field = 0.1 * np.sin(2 * np.pi * Y / Ny)

# psi_lift: g at dry boundary cells, zero everywhere else
psi_lift = np.zeros((Ny, Nx))
psi_lift[dry_boundary] = g_field[dry_boundary]
psi_lift_jnp = jnp.array(psi_lift)

print(f"Dry boundary cells: {int(dry_boundary.sum())}")
print(f"Boundary data range: [{psi_lift[dry_boundary].min():.4f}, "
      f"{psi_lift[dry_boundary].max():.4f}]")

# %%
# --- Corrected RHS ---
# f_corrected = f - A(psi_lift), where A is the masked Laplacian operator
A_psi_lift = fvx.masked_laplacian(psi_lift_jnp, mask_basin_jnp, dx, dy, lambda_=lambda_)
rhs_corrected = rhs_basin - A_psi_lift

# Solve with homogeneous BCs on the corrected RHS
sol_cg_inhom, info_inhom = fvx.solve_cg(
    A_basin, rhs_corrected, preconditioner=pc_basin, rtol=1e-10, atol=1e-10
)
sol_cg_inhom = sol_cg_inhom * mask_basin_jnp

# Full solution: psi = psi_lift + psi_hom
# At wet cells: psi = 0 + psi_hom (the solver result)
# At dry boundary cells: psi = g + 0 (the prescribed boundary data)
psi_full_inhom = psi_lift_jnp + sol_cg_inhom

# Verify: residual of the full solution should be small at interior wet cells
residual_full = rhs_basin - A_basin(psi_full_inhom)
# The residual at boundary-adjacent wet cells includes the effect of the
# prescribed dry-cell values — this is the correct inhomogeneous solve.
interior_residual = float(
    jnp.max(jnp.abs(residual_full * jnp.array(_erode_mask(mask_basin))))
)
print(f"CG iters: {info_inhom.iterations}")
print(f"Max interior residual: {interior_residual:.2e}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Show the wet domain + 1-cell dry boundary ring for context
plot_mask = (mask_basin.astype(bool) | dry_boundary).astype(float)

# (a) Prescribed boundary data (visible at dry boundary cells)
lift_np = np.where(plot_mask > 0.5, np.asarray(psi_lift_jnp), np.nan)
im0 = axes[0].imshow(lift_np, origin="lower", cmap="RdBu_r", interpolation="nearest")
axes[0].set_title("$\\psi_{\\mathrm{lift}}$ (boundary data $g$)", fontsize=11)
fig.colorbar(im0, ax=axes[0], shrink=0.8)

# (b) Homogeneous correction (interior wet cells only)
sol_np = np.where(mask_basin > 0.5, np.asarray(sol_cg_inhom), np.nan)
im1 = axes[1].imshow(sol_np, origin="lower", cmap="RdBu_r", interpolation="nearest")
axes[1].set_title("$\\psi_{\\mathrm{hom}}$ (correction, $\\psi=0$ BCs)", fontsize=11)
fig.colorbar(im1, ax=axes[1], shrink=0.8)

# (c) Full solution = lift + hom (show wet + boundary ring)
full_np = np.where(plot_mask > 0.5, np.asarray(psi_full_inhom), np.nan)
im2 = axes[2].imshow(full_np, origin="lower", cmap="RdBu_r", interpolation="nearest")
axes[2].set_title("$\\psi = \\psi_{\\mathrm{lift}} + \\psi_{\\mathrm{hom}}$", fontsize=11)
fig.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.axis("off")

fig.suptitle(
    "Inhomogeneous Dirichlet BCs via lifting trick",
    fontsize=14,
    y=1.02,
)
plt.tight_layout()
fig.savefig(IMG_DIR / "inhomogeneous_bc.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved inhomogeneous_bc.png")

# %% [markdown]
# ![Inhomogeneous Dirichlet BCs via lifting](../images/demo_solvers/inhomogeneous_bc.png)

# %% [markdown]
# The lifting trick works with **any** solver (spectral, CG, capacitance,
# multigrid) — it only modifies the RHS, not the solver itself.  This makes
# it straightforward to incorporate:
#
# - **Observation-derived boundary data** (tide gauges, altimetry)
# - **Reanalysis forcing** (ERA5 SSH or currents at open boundaries)
# - **Nesting** (parent model provides boundary values for a regional child)
#
# | Step | Operation |
# |------|-----------|
# | 1. Build $\psi_{\text{lift}}$ | Place prescribed values at boundary, zero interior |
# | 2. Correct RHS | $f' = f - (A - \lambda)\,\psi_{\text{lift}}$ |
# | 3. Solve homogeneous | $(A - \lambda)\,\psi_{\text{hom}} = f'$ with $\psi = 0$ on boundary |
# | 4. Reconstruct | $\psi = \psi_{\text{lift}} + \psi_{\text{hom}}$ |

# %% [markdown]
# ## 9. Summary Table
#
# | Geometry | Solver | Recommended when | Complexity |
# |----------|--------|-----------------|------------|
# | **Rectangle** | Spectral (DST) | Full grid, constant coeff | $O(N \log N)$, exact |
# | **Near-rectangular mask** | Capacitance matrix | Few boundary points, constant coeff | $O(N \log N + B^2)$ |
# | **Complex mask** | CG + spectral PC | Arbitrary mask, constant coeff | $O(k \cdot N \log N)$ |
# | **Any mask** | Multigrid | Variable coeff, moderate accuracy | $O(N)$ per V-cycle |
# | **Any mask, high accuracy** | MG + CG | Variable coeff, tight tolerance | $O(k \cdot N)$ |
#
# See the [Elliptic Solvers docs](https://jejjohnson.github.io/finitevolX/elliptic_solvers/)
# for full theory and the
# [Preconditioner Guide](https://jejjohnson.github.io/finitevolX/elliptic_solvers/#preconditioners)
# for detailed comparisons.

# %%
print("\n" + "=" * 80)
print(
    f"{'Geometry':<22} {'Solver':<14} {'Time (ms)':>10} {'Rel. Residual':>14} {'Detail'}"
)
print("=" * 80)
for geom in geom_order:
    geom_short = geom.split("\n")[0]
    for sname, data in results[geom].items():
        print(
            f"{geom_short:<22} {sname:<14} {data['time_ms']:>10.2f} "
            f"{data['rel_residual']:>14.1e} {data['label']}"
        )
    print("-" * 80)
