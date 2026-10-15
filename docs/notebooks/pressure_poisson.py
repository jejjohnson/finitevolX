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
# # Pressure Poisson Equation on the Arakawa C-Grid
#
# This notebook demonstrates solving the **pressure Poisson equation**
# $\nabla^2 p = \nabla \cdot \mathbf{u}$ on the Arakawa C-grid and using
# the result to project a velocity field to be divergence-free.
#
# ## C-Grid Variable Placement
#
# ```
#     X ── V ── X ── V ── X
#     |         |         |
#     U    T    U    T    U     T = tracer / pressure (cell centre)
#     |         |         |     U = u-velocity (east face)
#     X ── V ── X ── V ── X     V = v-velocity (north face)
#     |         |         |     X = vorticity (corner)
#     U    T    U    T    U
#     |         |         |
#     X ── V ── X ── V ── X
# ```
#
# The C-grid divergence at T-points uses backward differences of face
# velocities, while the pressure gradient uses forward differences.  Their
# composition $\nabla \cdot \nabla p$ gives exactly the 5-point Laplacian,
# which the spectral DST-II solver inverts when $p = 0$ sits on the wall
# faces.

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
from finitevolx import (
    solve_poisson_dst,
    solve_poisson_dst2,
    StaggeredDirichletHelmholtzSolver2D,
)

IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "pressure_poisson"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Grid Construction
#
# We build a `CartesianGrid2D` with a 2-cell ghost ring.  The interior has
# `nx × ny` T-points; the full array shape is `(ny+2) × (nx+2)`.

# %%
nx, ny = 64, 32
Lx, Ly = 4e6, 2e6  # 4000 km × 2000 km basin

grid = fvx.CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
dx, dy = grid.dx, grid.dy
Ny, Nx = ny + 2, nx + 2

print(f"Interior: {nx}×{ny},  dx={dx/1e3:.0f} km,  dy={dy/1e3:.0f} km")
print(f"Full array shape: ({Ny}, {Nx})")

# %% [markdown]
# ### Grid point locations
#
# A small 8×4 grid shows how T, U, and V points are staggered.

# %%
grid_small = fvx.CartesianGrid2D.from_interior(8, 4, Lx, Ly)
dx_s, dy_s = grid_small.dx, grid_small.dy

# T-points at cell centres
t_x = (np.arange(10) + 0.5) * dx_s
t_y = (np.arange(6) + 0.5) * dy_s
Tx, Ty = np.meshgrid(t_x, t_y)

# U-points at east faces
u_x = np.arange(1, 10) * dx_s
u_y = (np.arange(6) + 0.5) * dy_s
Ux, Uy = np.meshgrid(u_x, u_y)

# V-points at north faces
v_x = (np.arange(10) + 0.5) * dx_s
v_y = np.arange(1, 6) * dy_s
Vx, Vy = np.meshgrid(v_x, v_y)

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(Tx / 1e6, Ty / 1e6, c="steelblue", s=50, zorder=3, label="T (pressure)")
ax.scatter(Ux / 1e6, Uy / 1e6, c="crimson", s=30, marker=">", zorder=3, label="U (u-vel)")
ax.scatter(Vx / 1e6, Vy / 1e6, c="forestgreen", s=30, marker="^", zorder=3, label="V (v-vel)")

for i in range(11):
    ax.axvline(i * dx_s / 1e6, color="0.8", lw=0.5)
for j in range(7):
    ax.axhline(j * dy_s / 1e6, color="0.8", lw=0.5)

ax.axvspan(0, dx_s / 1e6, color="0.9", alpha=0.5)
ax.axvspan(9 * dx_s / 1e6, 10 * dx_s / 1e6, color="0.9", alpha=0.5)
ax.axhspan(0, dy_s / 1e6, color="0.9", alpha=0.5)
ax.axhspan(5 * dy_s / 1e6, 6 * dy_s / 1e6, color="0.9", alpha=0.5)
ax.text(
    0.5 * dx_s / 1e6, 3 * dy_s / 1e6, "ghost",
    ha="center", va="center", fontsize=8, color="0.5",
)

ax.set_xlabel("x (×10⁶ m)")
ax.set_ylabel("y (×10⁶ m)")
ax.set_title("Arakawa C-Grid: T, U, V point locations (8×4 interior)")
ax.legend(loc="upper right", fontsize=9)
ax.set_aspect("equal")
fig.savefig(IMG_DIR / "grid_points.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![C-grid point locations](../../images/pressure_poisson/grid_points.png)

# %% [markdown]
# ## 2. Divergence-Free Projection
#
# The pressure-projection method (Chorin splitting) removes divergence from
# a velocity field in three steps:
#
# 1. Compute $\nabla \cdot \mathbf{u}$ at T-points
# 2. Solve $\nabla^2 p = \nabla \cdot \mathbf{u}$ with $p = 0$ on walls
# 3. Correct: $\mathbf{u}^* = \mathbf{u} - \nabla p$
#
# Because the C-grid divergence and gradient operators compose to the exact
# 5-point Laplacian, the corrected velocity is **discretely** divergence-free.
#
# **Where the wall is:** pressure lives at T points (cell centres), and the
# walls are the cell faces half a cell outside the first / last interior
# T point.  A `Dirichlet1D` boundary condition puts $p = 0$ **on the face**
# by mirroring the interior into the ghost cell (`ghost = -interior`), and
# that is exactly the DST-II (staggered Dirichlet) spectral solver, which
# `fvx.pressure_from_divergence(..., bc="dst")` applies to the T interior of
# the full array.  The wall-face normal velocities `u[:, 0]` (west) and
# `v[0, :]` (south) are boundary-owned: no forward operator writes them, so
# we fill them from the same gradient.

# %%
diff_op = fvx.Difference2D(grid)
div_op = fvx.Divergence2D(grid)

# p = 0 on every wall face: Dirichlet1D(value=0) sets ghost = -interior.
wall = fvx.BoundaryConditionSet.closed()


def gradient(f):
    """(∂f/∂x at U, ∂f/∂y at V) on every face, the wall faces included.

    f's ghost ring must be filled.  The forward differences write U[1:-1, 1:-1]
    and V[1:-1, 1:-1]; the west / south wall faces U[:, 0] and V[0, :] are
    boundary-owned, so we add them:
    u[j, 1/2] = (f[j, 1] - f[j, 0]) / dx,  v[1/2, i] = (f[1, i] - f[0, i]) / dy.
    """
    u = diff_op.diff_x_T_to_U(f)  # [Ny, Nx]
    v = diff_op.diff_y_T_to_V(f)  # [Ny, Nx]
    u = u.at[1:-1, 0].set((f[1:-1, 1] - f[1:-1, 0]) / dx)
    v = v.at[0, 1:-1].set((f[1, 1:-1] - f[0, 1:-1]) / dy)
    return u, v


# --- Construct a known divergent velocity field ---
# A divergent potential φ at T-points (x = (i - 1/2) dx) that vanishes on the
# wall faces x = 0, Lx and y = 0, Ly; then u = ∂φ/∂x at U-points and
# v = ∂φ/∂y at V-points.  This guarantees ∇·u = ∇²φ in the discrete sense.
x_t = (jnp.arange(Nx) - 0.5) * dx  # [Nx]
y_t = (jnp.arange(Ny) - 0.5) * dy  # [Ny]
X_t, Y_t = x_t[None, :], y_t[:, None]
phi = jnp.sin(jnp.pi * X_t / Lx) * jnp.sin(jnp.pi * Y_t / Ly)
phi = phi + 0.3 * jnp.sin(2 * jnp.pi * X_t / Lx) * jnp.sin(jnp.pi * Y_t / Ly)
phi = wall(phi, dx, dy)  # the sines already mirror; this makes the wall explicit

# Divergent velocity: u = ∂φ/∂x at U-points, v = ∂φ/∂y at V-points
u, v = gradient(phi)

# Divergence at T-points
div_uv = div_op(u, v)
print(f"max |∇·u| = {float(jnp.abs(div_uv[1:-1, 1:-1]).max()):.4e}")

# %%
# --- Solve ∇²p = ∇·u and project ---
# Pass the full array: the wrapper solves the T interior [1:-1, 1:-1] with
# p = 0 on the faces (DST-II) and returns a zero ghost ring.
p_full = fvx.pressure_from_divergence(div_uv, dx, dy, bc="dst")
p_full = wall(p_full, dx, dy)  # fill the ghost ring: p = 0 on the faces
p_interior = p_full[1:-1, 1:-1]
rhs = div_uv[1:-1, 1:-1]

# Pressure gradient at U/V points (wall faces included)
dp_dx, dp_dy = gradient(p_full)

# Corrected velocity
u_corr = u - dp_dx
v_corr = v - dp_dy

# Verify divergence-free
div_corr = div_op(u_corr, v_corr)
max_div_before = float(jnp.abs(div_uv[1:-1, 1:-1]).max())
max_div_after = float(jnp.abs(div_corr[1:-1, 1:-1]).max())

print(f"max |∇·u| before: {max_div_before:.4e}")
print(f"max |∇·u| after:  {max_div_after:.4e}")

# Verify recovered pressure matches original potential
p_error = float(jnp.abs(p_interior - phi[1:-1, 1:-1]).max())
print(f"max |p - φ|:       {p_error:.4e}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Original divergence
vmax_div = float(jnp.abs(div_uv[1:-1, 1:-1]).max())
axes[0].imshow(
    np.asarray(div_uv[1:-1, 1:-1]),
    origin="lower", cmap="RdBu_r", aspect="auto",
    vmin=-vmax_div, vmax=vmax_div,
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[0].set_title(f"∇·u before (max = {max_div_before:.2e})")

# Pressure field
im1 = axes[1].imshow(
    np.asarray(p_interior), origin="lower", cmap="RdBu_r",
    aspect="auto",
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
X_plot = np.linspace(0, Lx / 1e6, nx)
Y_plot = np.linspace(0, Ly / 1e6, ny)
axes[1].contour(
    X_plot, Y_plot, np.asarray(p_interior),
    levels=12, colors="k", linewidths=0.5, alpha=0.5,
)
axes[1].set_title("Pressure p (= recovered φ)")
fig.colorbar(im1, ax=axes[1], shrink=0.8)

# Corrected divergence
axes[2].imshow(
    np.asarray(div_corr[1:-1, 1:-1]),
    origin="lower", cmap="RdBu_r", aspect="auto",
    vmin=-vmax_div, vmax=vmax_div,
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[2].set_title(f"∇·u after (max = {max_div_after:.2e})")

for ax in axes:
    ax.set_xlabel("x (×10³ km)")
    ax.set_ylabel("y (×10³ km)")

fig.suptitle("Pressure Poisson solve: divergence → pressure → projection", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(IMG_DIR / "projection.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Pressure Poisson: divergence, pressure, and projection](../../images/pressure_poisson/projection.png)

# %% [markdown]
# The projection removes divergence to **machine precision** and recovers
# φ exactly, because the DST-II solve inverts the same discrete Laplacian
# that the C-grid divergence and gradient compose to with $p = 0$ on the
# wall faces.

# %% [markdown]
# ## 3. DST-I vs DST-II: Which Solver?
#
# Both solvers take the T interior as their unknowns; they differ in where
# the zero sits:
#
# * **DST-II** puts $p = 0$ on the cell **faces**, half a cell outside the
#   first / last T point (`ghost = -interior`): the C-grid wall, and what
#   `Dirichlet1D` imposes.
# * **DST-I** puts $p = 0$ on the virtual nodes one step outside, i.e. at the
#   ghost-cell **centres** (`ghost = 0`): a wall half a cell too far out.
#
# Each is the exact inverse of its own discrete operator, so the Laplacian
# residual alone cannot tell them apart; the distance to φ (which vanishes
# on the faces) can.

# %%
p_dst1 = solve_poisson_dst(rhs, dx, dy)    # DST-I: p = 0 at the ghost centres
p_dst2 = solve_poisson_dst2(rhs, dx, dy)   # DST-II: p = 0 on the faces

diff = jnp.abs(p_dst1 - p_dst2)
print(f"max |p_DST-I - p_DST-II| = {float(diff.max()):.4e}")
print(f"relative diff = {float(diff.max() / jnp.abs(p_dst1).max()):.4e}")

# Each inverts the Laplacian with its own ghost convention
laplacian_p1 = diff_op.laplacian(jnp.zeros((Ny, Nx)).at[1:-1, 1:-1].set(p_dst1))
laplacian_p2 = diff_op.laplacian(
    wall(jnp.zeros((Ny, Nx)).at[1:-1, 1:-1].set(p_dst2), dx, dy)
)

err_dst1 = float(jnp.abs(laplacian_p1[1:-1, 1:-1] - rhs).max())
err_dst2 = float(jnp.abs(laplacian_p2[1:-1, 1:-1] - rhs).max())
print(f"Laplacian residual (DST-I, ghost = 0):          {err_dst1:.4e}")
print(f"Laplacian residual (DST-II, ghost = -interior): {err_dst2:.4e}")
print(f"max |p_DST-I  - φ| = {float(jnp.abs(p_dst1 - phi[1:-1, 1:-1]).max()):.4e}")
print(f"max |p_DST-II - φ| = {float(jnp.abs(p_dst2 - phi[1:-1, 1:-1]).max()):.4e}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

vmax = float(jnp.abs(p_dst1).max())

axes[0].imshow(
    np.asarray(p_dst1), origin="lower", cmap="RdBu_r",
    aspect="auto", vmin=-vmax, vmax=vmax,
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[0].set_title("DST-I — p = 0 at the ghost centres")

axes[1].imshow(
    np.asarray(p_dst2), origin="lower", cmap="RdBu_r",
    aspect="auto", vmin=-vmax, vmax=vmax,
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[1].set_title("DST-II — p = 0 on the faces")

im = axes[2].imshow(
    np.asarray(diff), origin="lower", cmap="hot_r",
    aspect="auto",
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[2].set_title("|DST-I − DST-II|")
fig.colorbar(im, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xlabel("x (×10³ km)")
    ax.set_ylabel("y (×10³ km)")

fig.suptitle("Regular vs staggered Dirichlet solver comparison", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(IMG_DIR / "dst1_vs_dst2.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![DST-I vs DST-II comparison](../../images/pressure_poisson/dst1_vs_dst2.png)

# %% [markdown]
# DST-II recovers φ to machine precision: its wall is the C-grid wall.
# DST-I solves a slightly larger basin (the wall half a cell further out),
# a first-order error that does not shrink faster than $\Delta x$.

# %% [markdown]
# ## 4. Module Class Interface
#
# spectraldiffx provides equinox `Module` classes for solver composition.
# `StaggeredDirichletHelmholtzSolver2D` wraps DST-II and is convenient
# for cell-centred workflows and JIT compilation.

# %%
solver = StaggeredDirichletHelmholtzSolver2D(dx=dx, dy=dy, alpha=0.0)
jit_solver = eqx.filter_jit(solver)

p_module = jit_solver(rhs)
print(f"max |module - DST-II| = {float(jnp.abs(p_module - p_dst2).max()):.2e}")

# %% [markdown]
# ## 5. Convenience Wrapper
#
# finitevolX provides one-line convenience functions that dispatch to
# the correct spectral solver based on the `bc` parameter.

# %%
# Pass the full [Ny, Nx] array: the wrapper slices the T interior itself.
p_convenience = fvx.pressure_from_divergence(div_uv, dx, dy, bc="dst")
print(
    "max |convenience - DST-II| = "
    f"{float(jnp.abs(p_convenience[1:-1, 1:-1] - p_dst2).max()):.2e}"
)

# %% [markdown]
# Here we pass `bc="dst"` (Dirichlet) to match the manufactured test where
# $\phi = 0$ on the wall faces.  Note that the library default is `bc="dct"`
# (Neumann, $\partial p / \partial n = 0$), which is the standard choice for
# pressure with solid walls in most ocean models.

# %% [markdown]
# ## Summary
#
# | Concept | Detail |
# |---------|--------|
# | **Pressure location** | T-points (cell centres) on the C-grid |
# | **Correct solver** | DST-II: $p = 0$ on the wall faces (`Dirichlet1D`, ghost = −interior) |
# | **DST-I** | $p = 0$ at the ghost-cell centres (ghost = 0): the wall half a cell too far out |
# | **Projection** | $\mathbf{u}^* = \mathbf{u} - \nabla p$ removes divergence to machine precision |
# | **C-grid advantage** | $\partial p/\partial x$ at U-points, $\partial p/\partial y$ at V-points — no interpolation |
# | **Module class** | `StaggeredDirichletHelmholtzSolver2D` wraps DST-II for equinox workflows |
# | **Convenience** | `fvx.pressure_from_divergence(div_u, dx, dy)` on the full array — defaults to DCT (Neumann, on the faces) |
