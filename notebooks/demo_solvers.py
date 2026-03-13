# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo: Elliptic Solvers on Different Geometries
#
# This notebook shows how to solve the Helmholtz equation
#
# $$
# (\nabla^2 - \lambda)\,\psi = f
# $$
#
# on four progressively harder geometries using finitevolX's solver stack:
#
# 1. **Rectangle** — spectral solver (exact, fastest)
# 2. **Basin with land border** — capacitance matrix method
# 3. **Circular basin** — preconditioned CG with spectral vs multigrid
# 4. **Variable-coefficient on irregular domain** — multigrid standalone
#
# Each section solves the same manufactured problem and compares the solution
# to the analytical expectation.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import finitevolx as fvx

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Shared setup
#
# We use a 64x64 grid and a smooth RHS field throughout.

# %%
Ny, Nx = 64, 64
Lx, Ly = 1.0, 1.0
dx, dy = Lx / Nx, Ly / Ny
lambda_ = 4.0  # Helmholtz parameter

# Smooth RHS: single-mode sinusoidal
j_idx = jnp.arange(Ny)[:, None]
i_idx = jnp.arange(Nx)[None, :]
rhs = jnp.sin(jnp.pi * (j_idx + 1) / (Ny + 1)) * jnp.sin(
    jnp.pi * (i_idx + 1) / (Nx + 1)
)


# Plotting helper
def plot_comparison(fields, titles, suptitle, masks=None):
    """Plot 2-4 fields side by side with optional mask overlay."""
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, field, title in zip(axes, fields, titles, strict=True):
        f = np.asarray(field)
        im = ax.imshow(f, origin="lower", cmap="RdBu_r", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ---
# ## 1. Rectangle — Spectral Solver (DST)
#
# For a rectangular domain with Dirichlet BCs ($\psi = 0$ on edges), the
# spectral solver gives the **exact** discrete solution in $O(N \log N)$ time.
# No iteration, no precomputation.

# %%
psi_spectral = fvx.solve_helmholtz_dst(rhs, dx, dy, lambda_)

plot_comparison(
    [rhs, psi_spectral],
    ["RHS $f$", "Solution $\\psi$ (spectral DST)"],
    "1. Rectangle — Spectral Solver",
)
print(
    f"Solution range: [{float(psi_spectral.min()):.4f}, {float(psi_spectral.max()):.4f}]"
)

# %% [markdown]
# ---
# ## 2. Basin with Land Border — Capacitance Matrix
#
# A rectangular ocean basin with a 4-cell land border. The capacitance
# method extends the spectral solver to handle masked domains by
# precomputing Green's functions at boundary points.

# %%
# Create basin mask: land on all edges
mask_basin = np.ones((Ny, Nx), dtype=float)
mask_basin[:4, :] = 0.0
mask_basin[-4:, :] = 0.0
mask_basin[:, :4] = 0.0
mask_basin[:, -4:] = 0.0

# Build capacitance solver (offline, one-time cost)
cap_solver = fvx.build_capacitance_solver(
    mask_basin.astype(bool), dx, dy, lambda_=lambda_, base_bc="dst"
)

# Solve
rhs_masked = rhs * jnp.array(mask_basin)
psi_cap = cap_solver(rhs_masked)

plot_comparison(
    [mask_basin, rhs_masked, psi_cap],
    ["Mask (basin)", "Masked RHS", "Solution (capacitance)"],
    "2. Basin with Land Border — Capacitance Matrix",
)
print(f"Boundary points: {len(cap_solver._j_b)}")

# %% [markdown]
# ---
# ## 3. Circular Basin — CG vs Multigrid
#
# A circular ocean basin is too irregular for the capacitance method to be
# efficient (many boundary points). We compare two approaches:
#
# - **CG with spectral preconditioner**: uses `masked_laplacian` as the
#   operator and a rectangular spectral solve as the preconditioner.
#   Simple and effective for constant-coefficient problems.
# - **Multigrid standalone**: the multigrid solver handles masked domains
#   natively — no CG wrapper needed.
#
# A third option, the **Nyström preconditioner**, is also available via
# `make_nystrom_preconditioner` — it works with any operator but requires
# tuning the approximation rank.

# %%
# Create circular mask
Y, X = np.mgrid[:Ny, :Nx]
center_y, center_x = Ny / 2, Nx / 2
radius = 0.4 * Ny
mask_circle = ((X - center_x) ** 2 + (Y - center_y) ** 2 < radius**2).astype(float)

mask_jnp = jnp.array(mask_circle)
rhs_circle = rhs * mask_jnp


# Define the masked Helmholtz operator for CG
def A_circle(x):
    return fvx.masked_laplacian(x, mask_jnp, dx, dy, lambda_=lambda_)


# --- CG with spectral preconditioner ---
pc_spectral = fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="dst")
psi_sp, info_sp = fvx.solve_cg(
    A_circle, rhs_circle, preconditioner=pc_spectral, rtol=1e-8, atol=1e-8
)
psi_sp = psi_sp * mask_jnp

# --- Multigrid standalone (no CG needed) ---
mg_solver = fvx.build_multigrid_solver(mask_circle, dx, dy, lambda_=lambda_, n_cycles=8)
psi_mg = mg_solver(rhs_circle)

from finitevolx._src.solvers.multigrid import _apply_operator

mg_residual = jnp.linalg.norm(rhs_circle - _apply_operator(psi_mg, mg_solver.levels[0]))

print("Solver comparison on circular basin:")
print(
    f"  CG + spectral PC:       {info_sp.iterations:3d} iters, residual = {info_sp.residual_norm:.2e}"
)
print(f"  Multigrid (8 V-cycles):              residual = {float(mg_residual):.2e}")

# %%
plot_comparison(
    [mask_circle, psi_sp, psi_mg],
    [
        "Mask (circle)",
        f"CG+spectral ({info_sp.iterations} iters)",
        "Multigrid (8 V-cycles)",
    ],
    "3. Circular Basin — CG vs Multigrid",
)

# %% [markdown]
# ---
# ## 4. Variable Coefficient on Irregular Domain — Multigrid Standalone
#
# The most challenging case: a spatially varying coefficient $c(x,y)$ on a
# masked domain. Only multigrid can handle this natively.
#
# We solve:
# $$
# \nabla \cdot \bigl(c(x,y)\,\nabla u\bigr) - \lambda\,u = f
# $$
#
# where $c(x,y) = 1 + 0.8\sin(2\pi x/L)$ varies across the domain.

# %%
# Variable coefficient field
coeff = 1.0 + 0.8 * np.sin(2 * np.pi * X / Nx)

# Irregular domain: rectangle with a notch cut out
mask_notch = np.ones((Ny, Nx), dtype=float)
mask_notch[:8, :] = 0.0  # bottom wall
mask_notch[-8:, :] = 0.0  # top wall
mask_notch[:, :8] = 0.0  # left wall
mask_notch[:, -8:] = 0.0  # right wall
mask_notch[20:44, :20] = 0.0  # left notch

mask_notch_jnp = jnp.array(mask_notch)
rhs_notch = rhs * mask_notch_jnp

# --- Multigrid standalone ---
mg_varcoeff = fvx.build_multigrid_solver(
    mask_notch, dx, dy, lambda_=lambda_, coeff=coeff, n_cycles=8
)
psi_mg_standalone = mg_varcoeff(rhs_notch)

# Check residual (reusing _apply_operator imported in section 3)
residual = rhs_notch - _apply_operator(psi_mg_standalone, mg_varcoeff.levels[0])
residual_norm = float(jnp.linalg.norm(residual))
rhs_norm = float(jnp.linalg.norm(rhs_notch))
print("Multigrid standalone (variable coeff, notched domain):")
print(f"  Relative residual: {residual_norm / rhs_norm:.2e}")

# %%
plot_comparison(
    [mask_notch, coeff * mask_notch, rhs_notch, psi_mg_standalone],
    ["Mask (notch)", "Coefficient $c(x,y)$", "Masked RHS", "Solution (multigrid)"],
    "4. Variable Coefficient on Irregular Domain — Multigrid",
)

# %% [markdown]
# ---
# ## 5. Multigrid-Preconditioned CG for Variable Coefficients
#
# For maximum accuracy with variable coefficients, combine multigrid
# preconditioning with CG. The multigrid V-cycle provides a good
# approximate inverse, and CG refines it to machine precision.
#
# **Important:** The CG operator and the multigrid preconditioner must use
# the same discretisation. Use `_apply_operator` with the multigrid's
# level data (not `masked_laplacian`, which uses a different boundary
# treatment).

# %%
# Use the multigrid solver we already built as a preconditioner
pc_mg_varcoeff = fvx.make_multigrid_preconditioner(mg_varcoeff)


# CG operator must match the multigrid discretisation
def A_notch(x):
    return _apply_operator(x, mg_varcoeff.levels[0])


psi_mgcg, info_mgcg = fvx.solve_cg(
    A_notch,
    rhs_notch,
    preconditioner=pc_mg_varcoeff,
    rtol=1e-10,
    atol=1e-10,
)
psi_mgcg = psi_mgcg * mask_notch_jnp

print("Multigrid-preconditioned CG:")
print(f"  Iterations: {info_mgcg.iterations}")
print(f"  Residual:   {info_mgcg.residual_norm:.2e}")

# Compare to standalone multigrid
diff = jnp.linalg.norm(psi_mgcg - psi_mg_standalone) / jnp.linalg.norm(psi_mgcg)
print(f"  Difference from standalone MG: {float(diff):.2e}")

# %%
plot_comparison(
    [psi_mg_standalone, psi_mgcg],
    ["Multigrid standalone", f"MG-preconditioned CG ({info_mgcg.iterations} iters)"],
    "5. Variable Coeff — Multigrid vs MG-Preconditioned CG",
)

# %% [markdown]
# ---
# ## Summary: Which Solver for Which Geometry?
#
# | Geometry | Coefficient | Recommended solver | Why |
# |---|---|---|---|
# | **Rectangle** | Constant | Spectral (DST/DCT/FFT) | Exact, $O(N \log N)$, no iteration |
# | **Near-rectangular mask** | Constant | Capacitance matrix | Nearly as fast as spectral |
# | **Complex mask** | Constant | CG + spectral preconditioner | Simple, effective |
# | **Any mask** | Variable $c(x,y)$ | Multigrid standalone | Handles both natively |
# | **Any mask, high accuracy** | Variable $c(x,y)$ | CG + multigrid preconditioner | Best convergence |
#
# See the [Elliptic Solvers docs](https://jejjohnson.github.io/finitevolX/elliptic_solvers/)
# for full theory and the
# [Preconditioner Guide](https://jejjohnson.github.io/finitevolX/elliptic_solvers/#preconditioners)
# for detailed comparisons.
