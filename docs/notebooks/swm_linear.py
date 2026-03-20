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
# # Linear Shallow-Water Model: Double-Gyre Wind Forcing on a Beta-Plane
#
# This notebook builds a **linear shallow-water model** step by step on the
# Arakawa C-grid using `finitevolx`.  The model simulates wind-driven
# ocean gyres in a closed rectangular basin -- the simplest configuration
# that produces the classic double-gyre circulation (an anticyclonic
# subtropical gyre in the south and a cyclonic subpolar gyre in the north).
#
# ## Governing equations
#
# The linearised shallow-water equations on a beta-plane are:
#
# $$\partial_t \eta = -H \,\nabla \cdot \mathbf{u} + \nu \,\nabla^2 \eta$$
#
# $$\partial_t u = -g \,\partial_x \eta + f\,v - r\,u + \nu \,\nabla^2 u + F_x$$
#
# $$\partial_t v = -g \,\partial_y \eta - f\,u - r\,v + \nu \,\nabla^2 v$$
#
# where:
#
# | Symbol | Meaning |
# |--------|---------|
# | $\eta$ | Free-surface anomaly (deviation from mean depth $H$) |
# | $u, v$ | Zonal and meridional velocity components |
# | $H$ | Mean fluid depth |
# | $g$ | Gravitational acceleration |
# | $f = f_0 + \beta y$ | Coriolis parameter on a beta-plane |
# | $r$ | Linear (Rayleigh) drag coefficient |
# | $\nu$ | Laplacian viscosity / diffusivity |
# | $F_x$ | Zonal wind body-force acceleration |
#
# The **wind forcing** follows the standard double-gyre pattern:
#
# $$F_x = -A \cos\!\left(\frac{2\pi y}{L_y}\right)$$
#
# This produces westward forcing (trade winds) near $y = 0$ and $y = L_y$,
# and eastward forcing (westerlies) near $y = L_y / 2$.  The resulting
# wind-stress curl drives an anticyclonic (subtropical) gyre in the
# southern half and a cyclonic (subpolar) gyre in the northern half.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import finitevolx as fvx

IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "swm_linear"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. C-Grid Variable Placement
#
# On the Arakawa C-grid, each prognostic variable lives at a different
# location within the grid cell:
#
# ```
#     X ── V ── X ── V ── X
#     |         |         |
#     U    T    U    T    U     T = eta (cell centre)
#     |         |         |     U = u-velocity (east face)
#     X ── V ── X ── V ── X     V = v-velocity (north face)
#     |         |         |     X = vorticity (corner)
#     U    T    U    T    U
#     |         |         |
#     X ── V ── X ── V ── X
# ```
#
# The staggering ensures that the discrete pressure gradient and Coriolis
# terms are naturally centred, minimising numerical dispersion.
#
# **Array layout:** for `nx` interior T-points in $x$ and `ny` in $y$,
# the full array shape is `(Ny, Nx) = (ny + 2, nx + 2)` -- one ghost cell
# on each side.  The physical interior occupies `[1:-1, 1:-1]`.

# %% [markdown]
# ## 2. Grid Construction

# %%
# --- Demo grid (small for fast execution in the notebook) ---
nx, ny = 32, 32
Lx = 5.12e6  # Domain length in x [m] (~5120 km)
Ly = 5.12e6  # Domain length in y [m] (~5120 km)

grid = fvx.ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
dx, dy = grid.dx, grid.dy
Ny, Nx = ny + 2, nx + 2

print(f"Interior cells: {nx} x {ny}")
print(f"Full array shape: ({Ny}, {Nx})")
print(f"Grid spacing: dx = {dx / 1e3:.1f} km, dy = {dy / 1e3:.1f} km")

# %% [markdown]
# ## 3. Physical Parameters
#
# We use values representative of a mid-latitude ocean basin on a
# beta-plane.
#
# | Parameter | Value | Units | Meaning |
# |-----------|-------|-------|---------|
# | $L_x, L_y$ | 5.12 x 10^6 | m | Basin dimensions (~5000 km square) |
# | $g$ | 9.81 | m s$^{-2}$ | Gravitational acceleration |
# | $H$ | 500 | m | Mean fluid depth (first baroclinic mode equivalent depth) |
# | $f_0$ | 9.375 x 10$^{-5}$ | s$^{-1}$ | Reference Coriolis parameter (~40 N) |
# | $\beta$ | 1.754 x 10$^{-11}$ | m$^{-1}$ s$^{-1}$ | Meridional Coriolis gradient |
# | $r$ | 5.0 x 10$^{-6}$ | s$^{-1}$ | Linear drag (damping timescale ~2.3 days) |
# | $\nu$ | 5.0 x 10$^5$ | m$^2$ s$^{-1}$ | Laplacian viscosity (smooths grid-scale noise) |
# | $A$ | 2.0 x 10$^{-7}$ | m s$^{-2}$ | Peak wind body-force acceleration |
# | $\Delta t$ | 200 | s | Time step |

# %%
gravity = 9.81  # [m s^-2]
mean_depth = 500.0  # [m]  reference depth H
f0 = 9.375e-5  # [s^-1]  Coriolis at ~40 deg N
beta = 1.754e-11  # [m^-1 s^-1]  df/dy
drag = 5.0e-6  # [s^-1]  linear Rayleigh drag
viscosity = 5.0e5  # [m^2 s^-1]  Laplacian viscosity
wind_amplitude = 2.0e-7  # [m s^-2]  peak wind body-force
dt = 200.0  # [s]  time step

# CFL check: gravity wave speed
c = np.sqrt(gravity * mean_depth)
cfl = c * dt / min(dx, dy)
print(f"Gravity wave speed c = {c:.1f} m/s")
print(f"CFL number = {cfl:.3f}  (must be < 1 for stability)")

# %% [markdown]
# ## 4. Initial Conditions and Forcing
#
# We start from rest ($u = v = 0$) with a small surface perturbation.
# The Coriolis parameter varies linearly with $y$ (beta-plane), and
# the zonal wind forcing has the double-gyre cosine profile.

# %%
# --- Coordinate arrays for the interior (T-points at cell centres) ---
x_interior = (np.arange(nx) + 0.5) * dx  # shape (nx,)
y_interior = (np.arange(ny) + 0.5) * dy  # shape (ny,)
X, Y = np.meshgrid(x_interior, y_interior)  # shape (ny, nx)

# --- Initial free-surface anomaly (small perturbation) ---
eta0_interior = 0.01 * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
# Pad with zero ghost cells -> shape (Ny, Nx)
eta = jnp.pad(jnp.asarray(eta0_interior), pad_width=1, mode="constant")
u = jnp.zeros((Ny, Nx))
v = jnp.zeros((Ny, Nx))
print(f"eta shape: {eta.shape}  (interior + ghost ring)")
print(f"u   shape: {u.shape}")
print(f"v   shape: {v.shape}")

# %%
# --- Coriolis parameter: f = f0 + beta * (y - Ly/2) at T-points ---
coriolis_interior = f0 + beta * (Y - 0.5 * Ly)
coriolis = jnp.pad(jnp.asarray(coriolis_interior), pad_width=1, mode="constant")
# Edge-pad so interpolations near walls see physical values
# instead of zero ghost cells.
coriolis = coriolis.at[0, :].set(coriolis[1, :])
coriolis = coriolis.at[-1, :].set(coriolis[-2, :])
coriolis = coriolis.at[:, 0].set(coriolis[:, 1])
coriolis = coriolis.at[:, -1].set(coriolis[:, -2])
print(
    f"Coriolis f range: [{float(coriolis[1:-1, 1:-1].min()):.4e}, "
    f"{float(coriolis[1:-1, 1:-1].max()):.4e}] s^-1"
)

# %%
# --- Wind forcing: F_x = -A * cos(2*pi*y / Ly) ---
# Negative cosine so that:
#   y ~ 0:      F_x < 0 (westward, trade winds)
#   y ~ Ly/2:   F_x > 0 (eastward, westerlies)
#   y ~ Ly:     F_x < 0 (westward, trade winds)
wind_u_interior = -wind_amplitude * np.cos(2.0 * np.pi * Y / Ly)
wind_u = jnp.pad(jnp.asarray(wind_u_interior), pad_width=1, mode="constant")
# Edge-pad wind forcing for the same reason as Coriolis
wind_u = wind_u.at[0, :].set(wind_u[1, :])
wind_u = wind_u.at[-1, :].set(wind_u[-2, :])
wind_u = wind_u.at[:, 0].set(wind_u[:, 1])
wind_u = wind_u.at[:, -1].set(wind_u[:, -2])

# %%
# --- Plot the wind forcing ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolormesh(
    X / 1e6,
    Y / 1e6,
    wind_u_interior * 1e7,
    cmap="RdBu_r",
    shading="auto",
)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label("$F_x$ [$\\times 10^{-7}$ m s$^{-2}$]")
ax.set_xlabel("x [10$^{6}$ m]")
ax.set_ylabel("y [10$^{6}$ m]")
ax.set_title("Double-gyre wind forcing $F_x = -A\\cos(2\\pi y / L_y)$")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(IMG_DIR / "wind_forcing.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {IMG_DIR / 'wind_forcing.png'}")

# %% [markdown]
# ![Double-gyre wind forcing](../../images/swm_linear/wind_forcing.png)

# %% [markdown]
# ## 5. Operator Construction
#
# finitevolx provides stateless operator modules that know about the grid
# spacing and stencil layout.  We need three operators:
#
# - **`Difference2D`** -- finite differences: `diff_x_T_to_U`, `diff_y_T_to_V`,
#   `divergence`, and `laplacian`.
# - **`Interpolation2D`** -- cross-averaging between staggered locations:
#   `T_to_U`, `T_to_V`, `V_to_U`, `U_to_V`, `U_to_T`, `V_to_T`, `X_to_T`.
# - **`Vorticity2D`** -- `relative_vorticity` at corner (X) points.

# %%
diff_op = fvx.Difference2D(grid)
interp_op = fvx.Interpolation2D(grid)
vort_op = fvx.Vorticity2D(grid)

print("Operators created (stateless equinox Modules, JIT-compatible).")

# %% [markdown]
# ## 6. Tendency Function
#
# The right-hand side (RHS) computes time tendencies for $\eta$, $u$, and
# $v$.  Each line below is annotated with:
# - which equation term it implements
# - the staggered grid location of the result
#
# The key idea: operators move values between staggered locations.  For
# example, `diff_op.diff_x_T_to_U` differentiates a T-point field and
# returns the result at U-points.


# %%
def tendency(eta_field, u_field, v_field):
    """Compute the linear shallow-water tendencies on the C-grid.

    Parameters
    ----------
    eta_field, u_field, v_field : Float[Array, "Ny Nx"]
        Current state (all on the full (Ny, Nx) grid with ghost cells).

    Returns
    -------
    eta_rhs, u_rhs, v_rhs : Float[Array, "Ny Nx"]
        Time tendencies for each prognostic variable.
    """
    # === eta equation: d(eta)/dt = -H * div(u) + nu * laplacian(eta) ===

    # Divergence of velocity at T-points: shape (Ny, Nx)
    # Uses backward differences: div = du/dx + dv/dy
    eta_rhs = -mean_depth * diff_op.divergence(u_field, v_field)

    # Diffusion of eta at T-points: shape (Ny, Nx)
    # 5-point Laplacian stencil
    eta_rhs = eta_rhs + viscosity * diff_op.laplacian(eta_field)

    # === Coriolis cross-terms ===
    # f lives at T-points; we need it at U-points and V-points
    # v lives at V-points; we need it at U-points for f*v in the u-equation
    # u lives at U-points; we need it at V-points for f*u in the v-equation

    v_on_u = interp_op.V_to_U(v_field)  # V-pts -> U-pts, shape (Ny, Nx)
    u_on_v = interp_op.U_to_V(u_field)  # U-pts -> V-pts, shape (Ny, Nx)
    coriolis_on_u = interp_op.T_to_U(coriolis)  # T-pts -> U-pts, shape (Ny, Nx)
    coriolis_on_v = interp_op.T_to_V(coriolis)  # T-pts -> V-pts, shape (Ny, Nx)

    # === u equation: d(u)/dt = -g * d(eta)/dx + f*v - r*u + nu*lap(u) + F_x ===

    # Pressure gradient: -g * d(eta)/dx at U-points
    u_rhs = -gravity * diff_op.diff_x_T_to_U(eta_field)

    # Coriolis: +f*v at U-points (f averaged to U, v averaged to U)
    u_rhs = u_rhs + coriolis_on_u * v_on_u

    # Wind forcing + drag + diffusion, all at U-points
    u_rhs = u_rhs + wind_u - drag * u_field + viscosity * diff_op.laplacian(u_field)

    # === v equation: d(v)/dt = -g * d(eta)/dy - f*u - r*v + nu*lap(v) ===

    # Pressure gradient: -g * d(eta)/dy at V-points
    v_rhs = -gravity * diff_op.diff_y_T_to_V(eta_field)

    # Coriolis: -f*u at V-points (f averaged to V, u averaged to V)
    v_rhs = v_rhs - coriolis_on_v * u_on_v

    # Drag + diffusion, all at V-points (no meridional wind forcing)
    v_rhs = v_rhs - drag * v_field + viscosity * diff_op.laplacian(v_field)

    return eta_rhs, u_rhs, v_rhs


def apply_bc(state):
    """Re-apply wall (zero ghost-cell) boundary conditions.

    After each time step, ghost cells are zeroed out.  This enforces
    no-normal-flow at the basin walls for velocity, and homogeneous
    Dirichlet for eta.
    """
    return tuple(fvx.pad_interior(f, mode="constant") for f in state)


def rhs(state):
    """Tendency with BC enforcement, for use with heun_step."""
    return tendency(*apply_bc(state))


# %% [markdown]
# ## 7. Time Stepping
#
# We use **Heun's method** (also called the modified Euler method or
# explicit trapezoidal rule).  It is a second-order Runge-Kutta scheme:
#
# 1. **Predictor** (forward Euler): $\tilde{y} = y_n + \Delta t \, f(y_n)$
# 2. **Corrector** (trapezoidal average): $y_{n+1} = y_n + \frac{\Delta t}{2}
#    \bigl[f(y_n) + f(\tilde{y})\bigr]$
#
# This is more stable than forward Euler and only requires two RHS
# evaluations per step.
#
# For the demo we run a short integration (500 steps = 100,000 s ~ 1.2 days)
# with **no spin-up** -- just enough to see the wind forcing begin to
# drive circulation.


# %%
@jax.jit
def step(eta_field, u_field, v_field):
    """Advance one Heun step and re-apply wall BCs."""
    state = fvx.heun_step((eta_field, u_field, v_field), rhs, dt)
    return apply_bc(state)


n_steps = 500
print(
    f"Running {n_steps} steps (dt = {dt} s, total = {n_steps * dt / 86400:.2f} days)..."
)

for _i in range(n_steps):
    eta, u, v = step(eta, u, v)

# Force JAX to finish all computation
eta.block_until_ready()
print("Done.")
print(f"max |eta| = {float(jnp.abs(eta[1:-1, 1:-1]).max()):.4e} m")

# %% [markdown]
# ## 8. Results
#
# We plot three diagnostic fields at the final time:
#
# 1. **Free-surface anomaly** $\eta$ -- shows the pressure pattern
# 2. **Speed** $|\mathbf{u}|$ -- magnitude of the velocity field
# 3. **Kinetic energy** $\tfrac{1}{2}|\mathbf{u}|^2$ -- energy in the flow
#
# **Note on boundary slicing:** Interpolated/derived fields (speed, KE)
# use `[2:-2, 2:-2]` instead of `[1:-1, 1:-1]` because the first interior
# row/column is contaminated by ghost-cell averaging (e.g. `U_to_T` at
# T-point [1,1] averages u[1,1] and u[1,0] where the ghost value is 0).

# %%
# --- Compute diagnostics ---
# Interpolate u, v to T-points for plotting
u_center = interp_op.U_to_T(u)  # U-pts -> T-pts
v_center = interp_op.V_to_T(v)  # V-pts -> T-pts

# Relative vorticity at corner (X) points, then average to T-points
zeta_corner = vort_op.relative_vorticity(u, v)
zeta_center = interp_op.X_to_T(zeta_corner)

# --- Coordinate grids ---
# [1:-1, 1:-1] coords for native T-point fields (eta)
x_full, y_full = np.meshgrid(x_interior, y_interior)
# [2:-2, 2:-2] coords for interpolated/derived fields (speed, KE, vorticity)
# The first interior row/column is contaminated by ghost-cell averaging,
# so we trim one extra cell on each side.
x_inner, y_inner = np.meshgrid(x_interior[1:-1], y_interior[1:-1])

# Extract interiors for plotting
# eta lives natively at T-points -- [1:-1, 1:-1] is fine
eta_plot = np.asarray(eta[1:-1, 1:-1])
# Interpolated fields use [2:-2, 2:-2] to avoid ghost-cell boundary artifacts
u_plot = np.asarray(u_center[2:-2, 2:-2])
v_plot = np.asarray(v_center[2:-2, 2:-2])
speed_plot = np.sqrt(u_plot**2 + v_plot**2)
ke_plot = 0.5 * (u_plot**2 + v_plot**2)
zeta_plot = np.asarray(zeta_center[2:-2, 2:-2])

# Energy diagnostics
total_ke = 0.5 * float(jnp.sum(u_center[2:-2, 2:-2] ** 2 + v_center[2:-2, 2:-2] ** 2)) * dx * dy
total_pe = 0.5 * gravity * float(jnp.sum(eta[1:-1, 1:-1] ** 2)) * dx * dy
print(f"eta  range: [{eta_plot.min():.4e}, {eta_plot.max():.4e}] m")
print(f"speed max:  {speed_plot.max():.4e} m/s")
print(f"zeta range: [{zeta_plot.min():.4e}, {zeta_plot.max():.4e}] s^-1")
print(f"Total KE:   {total_ke:.4e} m^4 s^-2")
print(f"Total PE:   {total_pe:.4e} m^4 s^-2")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# (a) Free-surface anomaly -- native T-point, use full interior
vmax_eta = float(np.abs(eta_plot).max())
if vmax_eta == 0:
    vmax_eta = 1.0
im0 = axes[0].pcolormesh(
    x_full / 1e6,
    y_full / 1e6,
    eta_plot,
    cmap="RdBu_r",
    shading="auto",
    vmin=-vmax_eta,
    vmax=vmax_eta,
)
fig.colorbar(im0, ax=axes[0], shrink=0.9)
axes[0].set_title(f"$\\eta$ [m] (max = {vmax_eta:.2e})")

# (b) Speed -- interpolated to T-points, use inner domain [2:-2, 2:-2]
im1 = axes[1].pcolormesh(
    x_inner / 1e6,
    y_inner / 1e6,
    speed_plot,
    cmap="viridis",
    shading="auto",
)
fig.colorbar(im1, ax=axes[1], shrink=0.9)
axes[1].set_title(f"|u| [m/s] (max = {speed_plot.max():.2e})")

# (c) Kinetic energy -- interpolated to T-points, use inner domain [2:-2, 2:-2]
im2 = axes[2].pcolormesh(
    x_inner / 1e6,
    y_inner / 1e6,
    ke_plot,
    cmap="inferno",
    shading="auto",
)
fig.colorbar(im2, ax=axes[2], shrink=0.9)
axes[2].set_title(f"KE [m$^2$ s$^{{-2}}$] (max = {ke_plot.max():.2e})")

for ax in axes:
    ax.set_xlabel("x [10$^{6}$ m]")
    ax.set_ylabel("y [10$^{6}$ m]")
    ax.set_aspect("equal")

fig.suptitle(
    f"Linear SWM after {n_steps} steps ({n_steps * dt / 86400:.1f} days)",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()
fig.savefig(IMG_DIR / "demo_results.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {IMG_DIR / 'demo_results.png'}")

# %% [markdown]
# ![Linear SWM results](../../images/swm_linear/demo_results.png)

# %% [markdown]
# At this early stage the solution is dominated by gravity-wave adjustment
# of the initial perturbation and the beginning of wind-driven flow.
# A much longer integration (with spin-up) is needed to reach the
# steady-state double-gyre circulation.

# %% [markdown]
# ## 9. Full-Resolution Simulation
#
# The production run uses a finer grid and long spin-up to reach
# quasi-steady state.  These are the parameters used by the script
# `scripts/swm_linear.py`:
#
# ```python
# # Full-resolution configuration
# nx, ny = 64, 64
# Lx, Ly = 5.12e6, 5.12e6
# gravity = 9.81
# mean_depth = 500.0
# f0 = 9.375e-5
# beta = 1.754e-11
# drag = 5.0e-6
# viscosity = 5.0e5
# wind_acceleration = 2.0e-7
# dt = 200.0
# spinup_steps = 150_000    # ~347 days silent spin-up
# steps = 15_000            # ~35 days of recorded snapshots
# snapshot_interval = 1_500  # save every ~3.5 days
# ```
#
# Run it with:
# ```bash
# uv run python scripts/swm_linear.py
# ```
#
# The pre-rendered animation shows the spun-up double-gyre circulation:
#
# ![Linear shallow-water double gyre](../../images/swm_linear/linear_shallow_water_double_gyre.gif)

# %% [markdown]
# ## 10. Summary
#
# | Concept | finitevolx API | Detail |
# |---------|---------------|--------|
# | **Grid** | `ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)` | Creates the C-grid with 2-cell ghost ring |
# | **Differences** | `Difference2D(grid)` | `diff_x_T_to_U`, `diff_y_T_to_V`, `divergence`, `laplacian` |
# | **Interpolation** | `Interpolation2D(grid)` | `T_to_U`, `T_to_V`, `V_to_U`, `U_to_V`, `U_to_T`, `V_to_T`, `X_to_T` |
# | **Vorticity** | `Vorticity2D(grid)` | `relative_vorticity` at corner (X) points |
# | **Time stepping** | `heun_step(state, rhs, dt)` | Heun / modified Euler (RK2) |
# | **Boundary conditions** | `pad_interior(field, mode="constant")` | Zero ghost cells = solid walls |
# | **Variable placement** | T-points (centres), U-points (east faces), V-points (north faces) | Standard Arakawa C-grid staggering |
# | **Coriolis coupling** | Interpolate $f$ to U/V, cross-average $v$ to U / $u$ to V | Ensures energy-conserving discretisation |
