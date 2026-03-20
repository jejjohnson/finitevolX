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
# # 2-D Advection: Solid-Body Rotation of a Cosine Bell
#
# This tutorial demonstrates 2-D advection on the Arakawa C-grid using a
# classic benchmark: a **cosine-bell** profile advected by a **solid-body
# rotation** velocity field.  After one full revolution the bell should
# return to its starting position, so any distortion is purely numerical.
#
# By the end you will understand:
#
# 1. How `Advection2D` computes the flux-form tendency
#    $-\nabla \cdot (\mathbf{u}\,q)$ on a staggered grid.
# 2. How to set up a **non-trivial velocity field** on the staggered
#    U/V-points.
# 3. How scheme choice affects **peak erosion** and **shape fidelity** in 2-D.
#
# > **Prerequisites** — read the
# > [Advection Theory](../advection.md) page and the
# > [1-D Advection](advection_1d_schemes.py) tutorial first.

# %% [markdown]
# ## The Test Problem
#
# ### Velocity field: solid-body rotation
#
# A non-divergent solid-body rotation centred at $(x_0, y_0)$ with
# angular velocity $\omega$:
#
# $$
# u(x, y) = -\omega\,(y - y_0), \qquad v(x, y) = \omega\,(x - x_0)
# $$
#
# Every fluid parcel traces a circle around $(x_0, y_0)$.  The period of
# one full revolution is $T = 2\pi / \omega$.
#
# ```
#         ←  ←  ←  ←  ←
#       ↙                  ↖
#     ↓                      ↑
#     ↓        (x₀,y₀)      ↑
#       ↘                  ↗
#         →  →  →  →  →
# ```
#
# ### Initial condition: cosine bell
#
# The cosine bell (Williamson et al., 1992) is a smooth, compactly
# supported bump:
#
# $$
# q_0(x, y) =
# \begin{cases}
#   \displaystyle\frac{h_0}{2}\Bigl(1 + \cos\!\bigl(\pi\,r/R\bigr)\Bigr)
#     & r < R \\[4pt]
#   0 & r \geq R
# \end{cases}
# $$
#
# where $r = \sqrt{(x - x_c)^2 + (y - y_c)^2}$ is the distance from
# the bell centre $(x_c, y_c)$ and $R$ is the bell radius.
#
# We place the bell **off-centre** so it orbits the rotation axis:
#
# ```
#   ┌─────────────────────┐
#   │                     │
#   │  ╭─╮               │
#   │  │●│  ← bell        │
#   │  ╰─╯     (x₀,y₀)  │
#   │             ×       │
#   │                     │
#   └─────────────────────┘
# ```
#
# After one full revolution ($t = T$), the bell should return to its
# starting position.

# %%
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import finitevolx as fvx

IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "advection_2d_rotation"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Grid Setup
#
# We build a 2-D Arakawa C-grid on the doubly-periodic domain
# $[0, 1] \times [0, 1]$.  As in the 1-D tutorial, we add extra ghost
# cells (`ng = 4` per side) so that the advection operator's write region
# `[2:-2, 2:-2]` covers all physical cells.
#
# ### C-grid staggering
#
# Recall the Arakawa C-grid colocation:
#
# ```
#      V[j+1/2, i]
#         ↑
#   ──────┼──────  U[j, i+1/2]
#         │
#     T[j, i]          ← scalar lives here
# ```
#
# - **T-points** (cell centres): scalar $q$ and coordinates $(x_T, y_T)$.
# - **U-points** (east faces): $x$-velocity $u$ at $(x_T + \Delta x/2,\; y_T)$.
# - **V-points** (north faces): $y$-velocity $v$ at $(x_T,\; y_T + \Delta y/2)$.

# %%
# --- physical parameters ---
nx = ny = 128  # physical cells per direction
Lx = Ly = 1.0
omega = 2 * jnp.pi  # angular velocity → period T = 1.0

# --- ghost ring ---
ng = 4  # ghost cells per side

# --- grid ---
dx = Lx / nx
dy = Ly / ny
grid = fvx.ArakawaCGrid2D(
    Nx=nx + 2 * ng, Ny=ny + 2 * ng, Lx=Lx, Ly=Ly, dx=dx, dy=dy
)

# --- coordinate arrays (physical domain only) ---
# T-point centres
x_t = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)  # (nx,)
y_t = jnp.linspace(0.5 * dy, Ly - 0.5 * dy, ny)  # (ny,)
X_T, Y_T = jnp.meshgrid(x_t, y_t)  # (ny, nx) each

print(f"Grid: {nx}×{ny} physical cells, dx = dy = {dx:.4e}")
print(f"Total array shape: ({grid.Ny}, {grid.Nx}) with {ng} ghost cells per side")

# %% [markdown]
# ## Velocity Field
#
# We evaluate the solid-body rotation analytically on the **staggered**
# grid points.  The velocity components must be sampled at U-points and
# V-points respectively — not at cell centres.
#
# The full array (including ghost cells) has shape `(Ny, Nx)` where
# `Ny = ny + 2*ng`.  Index $j$ in the full array corresponds to the
# physical $y$-coordinate $(j - \text{ng} + 0.5)\,\Delta y$ at T-points.

# %%
# Rotation centre = domain centre
x0, y0 = 0.5 * Lx, 0.5 * Ly

# --- Full coordinate arrays (including ghosts) ---
# T-points
x_full_t = (jnp.arange(grid.Nx) - ng + 0.5) * dx  # (Nx,)
y_full_t = (jnp.arange(grid.Ny) - ng + 0.5) * dy  # (Ny,)

# U-points are shifted +dx/2 in x relative to T-points
x_full_u = x_full_t + 0.5 * dx  # (Nx,)
y_full_u = y_full_t  # (Ny,)

# V-points are shifted +dy/2 in y relative to T-points
x_full_v = x_full_t  # (Nx,)
y_full_v = y_full_t + 0.5 * dy  # (Ny,)

# Meshgrids
Xu, Yu = jnp.meshgrid(x_full_u, y_full_u)  # U-point coords (Ny, Nx)
Xv, Yv = jnp.meshgrid(x_full_v, y_full_v)  # V-point coords (Ny, Nx)

# Solid-body rotation velocity
u_field = -omega * (Yu - y0)  # at U-points
v_field = omega * (Xv - x0)  # at V-points

# %% [markdown]
# ## Initial Condition: Cosine Bell
#
# We place the bell at $(x_c, y_c) = (0.25, 0.5)$ with radius $R = 0.15$
# and amplitude $h_0 = 1$.  This puts the bell to the left of the rotation
# centre, so it will orbit clockwise (for our sign convention).

# %%
xc, yc = 0.25, 0.5  # bell centre
R_bell = 0.15  # bell radius
h0 = 1.0  # bell amplitude


def cosine_bell(X, Y, xc, yc, R, h0):
    """Cosine bell initial condition on a 2-D grid."""
    r = jnp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    return jnp.where(r < R, 0.5 * h0 * (1.0 + jnp.cos(jnp.pi * r / R)), 0.0)


# Evaluate on T-point coordinates (physical domain)
q0_phys = cosine_bell(X_T, Y_T, xc, yc, R_bell, h0)

# Embed in the full array
q0 = jnp.zeros((grid.Ny, grid.Nx))
q0 = q0.at[ng:-ng, ng:-ng].set(q0_phys)

# %% [markdown]
# ## Periodic Boundary Conditions (2-D)
#
# Same idea as the 1-D case, but we wrap ghost rows/columns in both
# directions:

# %%
def enforce_periodic_2d(h: jax.Array, ng: int) -> jax.Array:
    """Fill *ng* ghost rows/columns on each side with periodic copies."""
    # Top/bottom (y-direction)
    h = h.at[:ng, :].set(h[-2 * ng : -ng, :])
    h = h.at[-ng:, :].set(h[ng : 2 * ng, :])
    # Left/right (x-direction)
    h = h.at[:, :ng].set(h[:, -2 * ng : -ng])
    h = h.at[:, -ng:].set(h[:, ng : 2 * ng])
    return h


q0 = enforce_periodic_2d(q0, ng)

# %% [markdown]
# ### Visualise the initial condition and velocity field:

# %%
fig, ax = plt.subplots(figsize=(6, 5.5))

# Contour plot of the cosine bell
q_plot = np.asarray(q0[ng:-ng, ng:-ng])
x_np, y_np = np.asarray(x_t), np.asarray(y_t)
cf = ax.contourf(x_np, y_np, q_plot, levels=20, cmap="RdYlBu_r")
fig.colorbar(cf, ax=ax, label="q")

# Velocity quivers (subsample for clarity)
skip = 8
xs = np.asarray(x_t[::skip])
ys = np.asarray(y_t[::skip])
Xq, Yq = np.meshgrid(xs, ys)
Uq = -float(omega) * (Yq - y0)
Vq = float(omega) * (Xq - x0)
ax.quiver(Xq, Yq, Uq, Vq, color="k", alpha=0.4, scale=30)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Initial Condition + Velocity Field")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(IMG_DIR / "initial_condition.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Initial condition](../../images/advection_2d_rotation/initial_condition.png)

# %% [markdown]
# ## Time Integration
#
# We integrate one full revolution ($T = 2\pi / \omega$) using SSP-RK3.
# The CFL condition for 2-D advection is
#
# $$
# \sigma = \left(\frac{|u|_{\max}}{\Delta x}
#        + \frac{|v|_{\max}}{\Delta y}\right) \Delta t \leq 1
# $$
#
# The maximum speed is $\omega \cdot r_{\max}$ where
# $r_{\max} = \sqrt{2}\,L/2 \approx 0.707$ for a unit square, giving
# $|u|_{\max} = |v|_{\max} \approx \omega / \sqrt{2}$.

# %%
T_period = 2 * jnp.pi / omega  # = 1.0 for omega = 2*pi

# CFL estimate
u_max = float(jnp.max(jnp.abs(u_field)))
v_max = float(jnp.max(jnp.abs(v_field)))
cfl = 0.4
dt = cfl / (u_max / dx + v_max / dy)
nsteps = int(jnp.ceil(T_period / dt))
dt = float(T_period) / nsteps  # adjust to land exactly at T

print(f"T = {float(T_period):.4f}, dt = {dt:.4e}, nsteps = {nsteps}")
print(f"|u|_max = {u_max:.2f}, |v|_max = {v_max:.2f}, CFL ≈ {cfl:.2f}")

# %% [markdown]
# ## Running the Simulation
#
# We compare three schemes: `upwind1`, `superbee`, and `weno5`.

# %%
advect = fvx.Advection2D(grid)
methods_2d = ["upwind1", "superbee", "weno5"]
results_2d = {}

# Snapshot times (fractions of period)
snapshot_fracs = [0.0, 0.25, 0.5, 1.0]
snapshots = {m: {} for m in methods_2d}

for method in methods_2d:

    def make_rhs(m):
        def rhs(q):
            q = enforce_periodic_2d(q, ng)
            return advect(q, u_field, v_field, method=m)

        return rhs

    rhs_fn = jax.jit(make_rhs(method))
    q = q0.copy()

    # Store initial snapshot
    snapshots[method][0.0] = np.asarray(q[ng:-ng, ng:-ng])

    for step in range(1, nsteps + 1):
        q = fvx.rk3_ssp_step(q, rhs_fn, dt)
        q = enforce_periodic_2d(q, ng)

        t_frac = step / nsteps
        # Check if we're near a snapshot time
        for sf in snapshot_fracs:
            if sf > 0 and abs(t_frac - sf) < 0.5 / nsteps:
                snapshots[method][sf] = np.asarray(q[ng:-ng, ng:-ng])

    results_2d[method] = np.asarray(q[ng:-ng, ng:-ng])
    peak = float(np.max(results_2d[method]))
    print(f"  {method:<10s}  peak = {peak:.4f}  (exact = {float(h0):.1f})")

# %% [markdown]
# ## Results: Snapshots at Quarter Revolutions
#
# The plots below show the cosine bell at $t = 0$, $T/4$, $T/2$, and $T$
# (one full revolution) for the `weno5` scheme.

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True, constrained_layout=True)
titles = ["t = 0", "t = T/4", "t = T/2", "t = T"]

for ax, sf, title in zip(axes, snapshot_fracs, titles):
    data = snapshots["weno5"].get(sf, results_2d["weno5"])
    # Use pcolormesh — contourf creates ring artifacts from tiny numerical noise.
    im = ax.pcolormesh(x_np, y_np, data, vmin=0, vmax=1, cmap="RdYlBu_r", shading="auto")
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.set_aspect("equal")

axes[0].set_ylabel("y")
fig.colorbar(im, ax=axes, label="q", shrink=0.85)
fig.suptitle("WENO5: Cosine Bell Solid-Body Rotation", fontsize=13)
fig.savefig(IMG_DIR / "weno5_snapshots.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![WENO5 snapshots](../../images/advection_2d_rotation/weno5_snapshots.png)

# %% [markdown]
# ## Scheme Comparison After One Revolution
#
# Comparing the three schemes side by side reveals how much each one
# erodes the peak and distorts the shape.

# %%
exact_2d = np.asarray(q0_phys)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, constrained_layout=True)

for ax, method in zip(axes, methods_2d):
    data = results_2d[method]
    im = ax.pcolormesh(x_np, y_np, data, vmin=0, vmax=1, cmap="RdYlBu_r", shading="auto")
    ax.contour(x_np, y_np, exact_2d, levels=[0.05, 0.5, 0.95], colors="k", linewidths=0.8, linestyles="--")
    peak = float(np.max(data))
    ax.set_title(f"{method}  (peak = {peak:.3f})")
    ax.set_xlabel("x")
    ax.set_aspect("equal")

axes[0].set_ylabel("y")
fig.colorbar(im, ax=axes, label="q", shrink=0.85)
fig.suptitle("Scheme Comparison After One Full Revolution", fontsize=13)
fig.savefig(IMG_DIR / "scheme_comparison.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Scheme comparison](../../images/advection_2d_rotation/scheme_comparison.png)

# %% [markdown]
# ## Error Maps
#
# The difference $q(T) - q_0$ shows the spatial structure of the error.
# Positive values (red) indicate the scheme deposited mass where there
# should be none; negative values (blue) indicate mass was removed.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, constrained_layout=True)

err_max = max(float(np.max(np.abs(results_2d[m] - exact_2d))) for m in methods_2d)

for ax, method in zip(axes, methods_2d):
    err = results_2d[method] - exact_2d
    im = ax.pcolormesh(
        x_np, y_np, err, vmin=-err_max, vmax=err_max, cmap="RdBu_r", shading="auto"
    )
    l2 = float(np.sqrt(np.mean(err**2)))
    ax.set_title(f"{method}  ($L_2$ = {l2:.3e})")
    ax.set_xlabel("x")
    ax.set_aspect("equal")

axes[0].set_ylabel("y")
fig.colorbar(im, ax=axes, label="q(T) - q0", shrink=0.85)
fig.suptitle("Error After One Revolution", fontsize=13)
fig.savefig(IMG_DIR / "error_maps.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Error maps](../../images/advection_2d_rotation/error_maps.png)

# %% [markdown]
# ## Cross-Section Through the Bell Peak
#
# A 1-D slice through the bell centre ($y = 0.5$) gives a clearer
# picture of peak erosion and shape distortion.

# %%
j_centre = ny // 2  # row index for y ≈ 0.5

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_np, exact_2d[j_centre, :], "k--", lw=2, label="Exact")

colours_2d = {"upwind1": "#d62728", "superbee": "#2ca02c", "weno5": "#ff7f0e"}
for method in methods_2d:
    ax.plot(x_np, results_2d[method][j_centre, :], colour := colours_2d[method], lw=1.5, label=method)

ax.set_xlabel("x")
ax.set_ylabel("q")
ax.set_title("Cross-Section at y = 0.5 After One Revolution")
ax.set_xlim(0, 0.5)
ax.legend()
fig.tight_layout()
fig.savefig(IMG_DIR / "cross_section.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Cross-section](../../images/advection_2d_rotation/cross_section.png)

# %% [markdown]
# ## Quantitative Summary
#
# | Scheme | Peak at $t=T$ | $L_2$ error | Character |
# |--------|:---:|:---:|-----------|
# | `upwind1` | low | high | Massive diffusion, bell nearly flat |
# | `superbee` | moderate | moderate | TVD-sharp edges, some steepening |
# | `weno5` | high | low | Best shape preservation overall |
#
# The numbers confirm the 1-D findings: **`weno5`** is the best
# general-purpose choice, while **`superbee`** trades peak accuracy for
# strict monotonicity.
#
# ## Next Steps
#
# - Try **`wenoz5`** (less dissipative WENO variant) or **`van_leer`**
#   (smoother TVD limiter).
# - Increase resolution to see convergence.
# - Add a **masked domain** (island) to test the adaptive stencil
#   selection — see the [Advection Theory](../advection.md#adaptive-stencil-selection-mask-aware)
#   page.
# - Combine advection with a physical model — see the
#   [Shallow Water](shallow_water.py) tutorial.
