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
# # Helmholtz Screening: QG PV Inversion
#
# In quasi-geostrophic (QG) dynamics, the streamfunction $\psi$ is recovered
# from potential vorticity (PV) $q$ via the **Helmholtz equation**:
#
# $$
# (\nabla^2 - \lambda)\,\psi = q, \qquad \lambda = \frac{1}{L_d^2}
# $$
#
# where $L_d = \sqrt{g'H}/f_0$ is the **Rossby deformation radius**.
# The parameter $\lambda$ controls *screening*: how far a PV anomaly's
# influence extends spatially.
#
# | Regime | $L_d$ | $\lambda$ | Character |
# |--------|-------|-----------|-----------|
# | Barotropic | $\infty$ | 0 | Poisson — long-range, fills the basin |
# | Mesoscale | 200–500 km | moderate | Moderate screening |
# | Submesoscale | 10–50 km | large | Strong screening — very local response |
#
# This notebook demonstrates the screening effect using the module-class
# solvers from spectraldiffx, integrated with finitevolX's C-grid operators.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import equinox as eqx

import finitevolx as fvx
from spectraldiffx import (
    StaggeredDirichletHelmholtzSolver2D,
    RegularNeumannHelmholtzSolver2D,
)

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "helmholtz_screening"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Setup: Isolated PV Anomaly
#
# We place a Gaussian PV anomaly in a square basin with no-normal-flow walls
# ($\psi = 0$ on the boundary).
#
# $$
# q(x, y) = A \exp\!\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2\sigma^2}\right)
# $$

# %%
nx, ny = 128, 128
Lx, Ly = 2e6, 2e6  # 2000 km × 2000 km
dx, dy = Lx / nx, Ly / ny

# PV anomaly at domain centre
amplitude = 1e-9  # s⁻²  (typical QG PV magnitude)
sigma = 200e3     # 200 km radius
x0, y0 = Lx / 2, Ly / 2

# T-point coordinates (cell centres, interior grid)
i_idx = jnp.arange(nx)[None, :] + 0.5
j_idx = jnp.arange(ny)[:, None] + 0.5
x_T = i_idx * dx
y_T = j_idx * dy

q = amplitude * jnp.exp(-((x_T - x0) ** 2 + (y_T - y0) ** 2) / (2 * sigma**2))

print(f"PV range: [{float(q.min()):.4e}, {float(q.max()):.4e}] s⁻²")

# %% [markdown]
# ## 2. The Screening Effect
#
# We solve $(∇^2 - \lambda)\psi = q$ for four values of $L_d$, sweeping
# from Poisson ($\lambda = 0$) to strongly screened ($L_d = 50$ km).

# %%
Ld_values = [np.inf, 500e3, 200e3, 50e3]  # metres
Ld_labels = ["$L_d = \\infty$ (Poisson)", "$L_d = 500$ km", "$L_d = 200$ km", "$L_d = 50$ km"]

solutions = []
for Ld in Ld_values:
    lam = 0.0 if np.isinf(Ld) else 1.0 / Ld**2
    solver = StaggeredDirichletHelmholtzSolver2D(dx=dx, dy=dy, alpha=lam)
    psi = eqx.filter_jit(solver)(q)
    solutions.append(psi)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, psi, label in zip(axes.flat, solutions, Ld_labels):
    im = ax.imshow(
        np.asarray(psi), origin="lower", cmap="RdBu_r",
        extent=[0, Lx / 1e6, 0, Ly / 1e6],
    )
    X_plot = np.linspace(0, Lx / 1e6, nx)
    Y_plot = np.linspace(0, Ly / 1e6, ny)
    ax.contour(X_plot, Y_plot, np.asarray(psi), levels=10, colors="k", linewidths=0.5, alpha=0.5)
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("x (×10³ km)")
    ax.set_ylabel("y (×10³ km)")
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Screening effect: ψ from $(∇^2 - λ)ψ = q$", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(IMG_DIR / "screening_effect.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# As $L_d$ decreases (screening increases), the streamfunction response
# becomes more localised around the PV anomaly.  At $L_d = 50$ km, the
# response barely extends beyond the Gaussian core.

# %% [markdown]
# ## 3. Cross-Section Profiles
#
# 1-D slices through the anomaly centre show the spatial extent quantitatively.

# %%
mid_j = ny // 2

fig, ax = plt.subplots(figsize=(10, 5))

x_km = np.asarray(x_T[mid_j, :]) / 1e3
styles = ["-", "--", "-.", ":"]

for psi, label, style in zip(solutions, Ld_labels, styles):
    profile = np.asarray(psi[mid_j, :])
    # Normalise for shape comparison
    profile_norm = profile / np.abs(profile).max() if np.abs(profile).max() > 0 else profile
    ax.plot(x_km, profile_norm, style, label=label, lw=2)

ax.set_xlabel("x (km)")
ax.set_ylabel("ψ / max|ψ| (normalised)")
ax.set_title("Cross-section through PV anomaly centre (y = Ly/2)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(x0 / 1e3, color="0.5", ls=":", lw=1)
fig.savefig(IMG_DIR / "cross_sections.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Dirichlet vs Neumann Boundary Conditions
#
# - **Dirichlet** ($\psi = 0$ on walls): no-normal-flow basin walls
# - **Neumann** ($\partial\psi/\partial n = 0$ on walls): free-slip walls
#
# For Helmholtz with $\lambda > 0$, both are well-posed (no null space).

# %%
Ld_compare = 200e3
lam_compare = 1.0 / Ld_compare**2

solver_dir = StaggeredDirichletHelmholtzSolver2D(dx=dx, dy=dy, alpha=lam_compare)
solver_neu = RegularNeumannHelmholtzSolver2D(dx=dx, dy=dy, alpha=lam_compare)

psi_dir = eqx.filter_jit(solver_dir)(q)
psi_neu = eqx.filter_jit(solver_neu)(q)

diff_bc = jnp.abs(psi_dir - psi_neu)
print(f"max |ψ_Dir - ψ_Neu| = {float(diff_bc.max()):.4e}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

vmax = max(float(jnp.abs(psi_dir).max()), float(jnp.abs(psi_neu).max()))

axes[0].imshow(
    np.asarray(psi_dir), origin="lower", cmap="RdBu_r",
    extent=[0, Lx / 1e6, 0, Ly / 1e6], vmin=-vmax, vmax=vmax,
)
axes[0].set_title("Dirichlet (ψ=0 on walls)")

axes[1].imshow(
    np.asarray(psi_neu), origin="lower", cmap="RdBu_r",
    extent=[0, Lx / 1e6, 0, Ly / 1e6], vmin=-vmax, vmax=vmax,
)
axes[1].set_title("Neumann (∂ψ/∂n=0 on walls)")

im = axes[2].imshow(
    np.asarray(diff_bc), origin="lower", cmap="hot_r",
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[2].set_title("|Dirichlet − Neumann|")
fig.colorbar(im, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xlabel("x (×10³ km)")
    ax.set_ylabel("y (×10³ km)")

fig.suptitle(f"BC comparison at Ld = {Ld_compare/1e3:.0f} km", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(IMG_DIR / "dirichlet_vs_neumann.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# The difference is concentrated near the domain boundaries, where the BCs
# differ.  In the interior (far from walls), both solutions agree — the
# screening localises the response so boundary effects are minimal when
# the anomaly is well inside the domain.

# %% [markdown]
# ## 5. Integration with finitevolX
#
# finitevolX provides convenience wrappers that dispatch to the spectral
# solvers.  `pv_inversion` solves $(∇^2 - \lambda)\psi = q$ and supports
# multi-layer batching.

# %%
grid = fvx.ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)

# Existing API (uses DST-I internally — regular grid)
psi_fvx = fvx.pv_inversion(q, dx, dy, lambda_=lam_compare, bc="dst")

# Direct staggered solver (correct for T-point PV)
psi_direct = eqx.filter_jit(solver_dir)(q)

diff_api = jnp.abs(psi_fvx - psi_direct)
print(f"max |pv_inversion(DST-I) - DST-II| = {float(diff_api.max()):.4e}")
print(f"relative diff = {float(diff_api.max() / jnp.abs(psi_direct).max()):.4e}")

# %% [markdown]
# The $O(dx^2)$ difference arises because `pv_inversion(bc="dst")` dispatches
# to DST-I (regular Dirichlet) while the module class uses DST-II (staggered
# Dirichlet).  For T-point data, DST-II is more appropriate.

# %% [markdown]
# ## 6. JAX Ecosystem: JIT, vmap, grad
#
# The module-class solvers are fully JAX-compatible: JIT-compiled, vmappable,
# and differentiable.

# %%
# JIT compilation
jit_solver = eqx.filter_jit(solver_dir)
_ = jit_solver(q)  # compile
psi_jit = jit_solver(q)  # fast path
print(f"JIT solve: max |ψ| = {float(jnp.abs(psi_jit).max()):.4e}")

# %%
# vmap over an ensemble of PV fields
q_ensemble = q[None, :, :] * jnp.linspace(0.5, 2.0, 5)[:, None, None]
print(f"Ensemble shape: {q_ensemble.shape}")

batch_solve = jax.vmap(jit_solver)
psi_ensemble = batch_solve(q_ensemble)
print(f"Solution shape: {psi_ensemble.shape}")
print(f"Max |ψ| per member: {[f'{float(jnp.abs(p).max()):.4e}' for p in psi_ensemble]}")

# %%
# Differentiable: gradient of energy w.r.t. PV
def energy(q_field):
    psi = solver_dir(q_field)
    return 0.5 * jnp.sum(psi * q_field) * dx * dy

E = energy(q)
dE_dq = jax.grad(energy)(q)

print(f"Energy E = {float(E):.4e}")
print(f"dE/dq range: [{float(dE_dq.min()):.4e}, {float(dE_dq.max()):.4e}]")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(
    np.asarray(psi_jit), origin="lower", cmap="RdBu_r",
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[0].set_title("ψ = solver(q)")
fig.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(
    np.asarray(dE_dq), origin="lower", cmap="RdBu_r",
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[1].set_title("∂E/∂q (energy sensitivity)")
fig.colorbar(im1, ax=axes[1], shrink=0.8)

for ax in axes:
    ax.set_xlabel("x (×10³ km)")
    ax.set_ylabel("y (×10³ km)")

fig.suptitle("JAX transforms: JIT solve and autodiff", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(IMG_DIR / "jax_transforms.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Timing Comparison
#
# We benchmark the module solver vs the functional API at two resolutions.

# %%
import time


def benchmark(fn, *args, n_warmup=3, n_iter=20):
    """Time a JIT-compiled function."""
    for _ in range(n_warmup):
        fn(*args).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args).block_until_ready()
    elapsed = (time.perf_counter() - t0) / n_iter
    return elapsed


results = {}
for N in [128, 256]:
    dx_n, dy_n = Lx / N, Ly / N
    i_n = jnp.arange(N)[None, :] + 0.5
    j_n = jnp.arange(N)[:, None] + 0.5
    x_n = i_n * dx_n
    y_n = j_n * dy_n
    q_n = amplitude * jnp.exp(-((x_n - x0) ** 2 + (y_n - y0) ** 2) / (2 * sigma**2))

    # Module class
    solver_n = StaggeredDirichletHelmholtzSolver2D(dx=dx_n, dy=dy_n, alpha=lam_compare)
    jit_solver_n = eqx.filter_jit(solver_n)
    t_module = benchmark(jit_solver_n, q_n)

    # Functional API
    solve_fn = jax.jit(lambda rhs: fvx.solve_helmholtz_dst(rhs, dx_n, dy_n, lam_compare))
    t_func = benchmark(solve_fn, q_n)

    results[N] = {"module": t_module, "functional": t_func}

print(f"{'Method':<20} {'128×128':>12} {'256×256':>12}")
print("-" * 46)
for method in ["module", "functional"]:
    t128 = results[128][method] * 1000
    t256 = results[256][method] * 1000
    print(f"{method:<20} {t128:>10.2f} ms {t256:>10.2f} ms")

# %% [markdown]
# Both methods have identical performance — the module class is a thin
# wrapper around the functional API.  The spectral solver is
# $O(N \log N)$ and typically takes sub-millisecond time for moderate grids.

# %% [markdown]
# ## Summary
#
# | Concept | Detail |
# |---------|--------|
# | **Equation** | $(∇^2 - \lambda)\psi = q$ where $\lambda = 1/L_d^2$ |
# | **Screening** | Larger $\lambda$ → more localised $\psi$ response |
# | **Dirichlet** | $\psi = 0$ on walls (no-normal-flow basin) |
# | **Neumann** | $\partial\psi/\partial n = 0$ (free-slip walls) |
# | **Module class** | `StaggeredDirichletHelmholtzSolver2D(dx, dy, alpha)` |
# | **Convenience** | `fvx.pv_inversion(q, dx, dy, lambda_)` dispatches to DST-I |
# | **JAX compat** | JIT, vmap, grad all work out of the box |
