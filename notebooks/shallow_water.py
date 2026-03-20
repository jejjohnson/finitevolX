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
# # Nonlinear Shallow Water Equations on the Arakawa C-Grid
#
# This notebook extends the **linear** shallow-water model to the **nonlinear**
# case.  The two key additions are:
#
# 1. The **continuity equation** uses the full layer depth $H + \eta$ instead
#    of the constant reference depth $H$.
# 2. The **momentum equations** include nonlinear advection $-(\mathbf{u}
#    \cdot \nabla) \mathbf{u}$.
#
# ## Governing equations
#
# $$
# \frac{\partial \eta}{\partial t}
#   = -\nabla \cdot \bigl[(H + \eta)\,\mathbf{u}\bigr]
#     + \nu\,\nabla^2 \eta
# $$
#
# $$
# \frac{\partial u}{\partial t}
#   = -g\,\frac{\partial \eta}{\partial x}
#     - (\mathbf{u} \cdot \nabla) u
#     + f\,v - r\,u
#     + \nu\,\nabla^2 u + F_x
# $$
#
# $$
# \frac{\partial v}{\partial t}
#   = -g\,\frac{\partial \eta}{\partial y}
#     - (\mathbf{u} \cdot \nabla) v
#     - f\,u - r\,v
#     + \nu\,\nabla^2 v
# $$
#
# Compared to the linear version:
#
# - **Continuity**: $-H\,\nabla \cdot \mathbf{u}$ becomes
#   $-\nabla \cdot [(H + \eta)\,\mathbf{u}]$, coupling the free surface
#   into the mass flux.
# - **Momentum**: the advection terms $-(\mathbf{u} \cdot \nabla) u$ and
#   $-(\mathbf{u} \cdot \nabla) v$ are absent in the linear model.
#
# We integrate these on a closed-basin Arakawa C-grid driven by a
# double-gyre wind stress, using `finitevolx.heun_step` (Heun/RK2).

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

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "shallow_water"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. What's New vs the Linear Model
#
# | Term | Linear | Nonlinear |
# |------|--------|-----------|
# | **Continuity** | $-H\,\nabla \cdot \mathbf{u}$ | $-\nabla \cdot [(H+\eta)\,\mathbf{u}]$ |
# | **Momentum advection** | absent | $-(\mathbf{u} \cdot \nabla) u$, $-(\mathbf{u} \cdot \nabla) v$ |
# | **Pressure gradient** | $-g\,\partial_x \eta$ | $-g\,\partial_x \eta$ (same -- see note below) |
# | **Coriolis, drag, diffusion** | unchanged | unchanged |
# | **New operator** | -- | `Advection2D` (upwind flux reconstruction) |
#
# **Note on the pressure gradient**: the nonlinear model uses the same
# $-g\,\partial_x \eta$ as the linear model, *not* the full Bernoulli
# function $B = g\eta + \tfrac{1}{2}|\mathbf{u}|^2$.  Using the full
# Bernoulli together with the separate advection $-(\mathbf{u}\cdot\nabla)
# \mathbf{u}$ would double-count the kinetic-energy gradient
# $\nabla(\tfrac{1}{2}|\mathbf{u}|^2)$, which is already contained in
# the advection term.

# %% [markdown]
# ## 2. Grid and Parameters
#
# We use the same domain and physics as the linear shallow-water notebook.
# The grid is a closed-basin Arakawa C-grid with a 2-cell ghost ring
# (zero = solid walls).

# %%
# --- Domain ---
nx, ny = 32, 32
Lx, Ly = 5.12e6, 5.12e6  # 5120 km square basin

grid = fvx.ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
dx, dy = grid.dx, grid.dy
Ny, Nx = ny + 2, nx + 2

print(f"Interior: {nx} x {ny},  dx = {dx / 1e3:.0f} km,  dy = {dy / 1e3:.0f} km")
print(f"Full array shape: ({Ny}, {Nx})")

# %%
# --- Physical parameters ---
gravity = 9.81  # gravitational acceleration  [m s-2]
mean_depth = 500.0  # reference layer depth H     [m]
f0 = 9.375e-5  # Coriolis parameter          [s-1]
beta = 1.754e-11  # beta-plane gradient          [m-1 s-1]
drag = 5.0e-6  # Rayleigh friction            [s-1]
viscosity = 5.0e5  # Laplacian viscosity           [m2 s-1]
wind_amp = 2.0e-7  # peak wind acceleration        [m s-2]
dt = 200.0  # time step                     [s]

# | Parameter | Symbol | Value | Units |
# |-----------|--------|-------|-------|
# | Gravity | $g$ | 9.81 | m s$^{-2}$ |
# | Mean depth | $H$ | 500 | m |
# | Coriolis | $f_0$ | 9.375e-5 | s$^{-1}$ |
# | Beta | $\beta$ | 1.754e-11 | m$^{-1}$ s$^{-1}$ |
# | Drag | $r$ | 5e-6 | s$^{-1}$ |
# | Viscosity | $\nu$ | 5e5 | m$^2$ s$^{-1}$ |
# | Wind | $A$ | 2e-7 | m s$^{-2}$ |
# | Time step | $\Delta t$ | 200 | s |

# %% [markdown]
# ## 3. Initial Conditions and Forcing
#
# The setup is identical to the linear shallow-water notebook:
#
# - **Initial $\eta$**: a gentle sinusoidal bump.
# - **Coriolis**: beta-plane $f = f_0 + \beta (y - L_y/2)$.
# - **Wind**: double-gyre pattern $F_x = -A\cos(2\pi y / L_y)$.
#
# We pad interior fields with a zero ghost ring (solid-wall BCs), then
# edge-pad the Coriolis and wind fields so that interpolations near the
# walls see physical values rather than zeros.

# %%
# Interior coordinate arrays
x_1d = (np.arange(nx) + 0.5) * dx
y_1d = (np.arange(ny) + 0.5) * dy
x_2d, y_2d = np.meshgrid(x_1d, y_1d)  # shape (ny, nx)

# Initial free-surface anomaly [m]
eta0_interior = 0.01 * np.sin(np.pi * x_2d / Lx) * np.sin(np.pi * y_2d / Ly)

# Coriolis parameter on interior T-points [s-1]
coriolis_interior = f0 + beta * (y_2d - 0.5 * Ly)

# Double-gyre zonal wind forcing [m s-2]
wind_u_interior = -wind_amp * np.cos(2.0 * np.pi * y_2d / Ly)


def pad_to_full(interior: np.ndarray) -> jnp.ndarray:
    """Pad interior array with a 1-cell ghost ring of zeros."""
    return jnp.pad(jnp.asarray(interior), pad_width=1, mode="constant")


def edge_pad(field: jnp.ndarray) -> jnp.ndarray:
    """Copy nearest interior values into ghost cells (for static fields)."""
    field = field.at[0, :].set(field[1, :])
    field = field.at[-1, :].set(field[-2, :])
    field = field.at[:, 0].set(field[:, 1])
    field = field.at[:, -1].set(field[:, -2])
    return field


# Build full-grid fields
eta = pad_to_full(eta0_interior)
u = jnp.zeros_like(eta)
v = jnp.zeros_like(eta)
coriolis = edge_pad(pad_to_full(coriolis_interior))
wind_u = edge_pad(pad_to_full(wind_u_interior))

print(
    f"eta shape: {eta.shape},  min/max: {float(eta.min()):.4f} / {float(eta.max()):.4f}"
)

# %% [markdown]
# ## 4. The Advection Operator
#
# The key new ingredient is `Advection2D`.  It computes the advective flux
# divergence $-\nabla \cdot (h\,\mathbf{u})$ at T-points using upwind
# flux reconstruction.
#
# ```python
# adv_op = fvx.Advection2D(grid=grid)
# tendency = adv_op(h, u, v, method="upwind1")
# ```
#
# - `h` is the scalar field being advected (at T-points).
# - `u`, `v` are the advecting velocities (at U- and V-points).
# - `method="upwind1"` selects first-order upwind reconstruction; higher
#   orders (`"weno3"`, `"weno5"`, etc.) are also available.
#
# The operator returns the tendency $-\nabla \cdot (h\,\mathbf{u})$,
# which is exactly what appears in the continuity equation.

# %%
diff = fvx.Difference2D(grid=grid)
interp = fvx.Interpolation2D(grid=grid)
adv = fvx.Advection2D(grid=grid)
vort = fvx.Vorticity2D(grid=grid)

# Quick demonstration: advect a test field
test_h = pad_to_full(np.sin(np.pi * x_2d / Lx) * np.sin(np.pi * y_2d / Ly))
test_u = pad_to_full(0.01 * np.ones_like(x_2d))
test_v = jnp.zeros_like(test_h)

adv_tendency = adv(test_h, test_u, test_v, method="upwind1")
print(f"Advection tendency shape: {adv_tendency.shape}")
print(f"max |tendency|: {float(jnp.abs(adv_tendency[1:-1, 1:-1]).max()):.4e}")

# %% [markdown]
# ## 5. The Nonlinear Continuity Equation
#
# In the linear model, the continuity equation is:
#
# $$
# \frac{\partial \eta}{\partial t} = -H\,\nabla \cdot \mathbf{u}
#   + \nu\,\nabla^2 \eta
# $$
#
# In the nonlinear model, $H$ is replaced by the **full layer depth**
# $H + \eta$:
#
# $$
# \frac{\partial \eta}{\partial t}
#   = -\nabla \cdot \bigl[(H + \eta)\,\mathbf{u}\bigr]
#     + \nu\,\nabla^2 \eta
# $$
#
# This means we advect the *total depth* $H + \eta$ rather than just
# using a constant $H$ times the velocity divergence.  In code:
#
# ```python
# layer_depth = mean_depth + eta
# eta_rhs = adv(layer_depth, u, v, method="upwind1")  # -div((H+eta)*u)
# eta_rhs = eta_rhs + viscosity * diff.laplacian(eta)  # + nu * lap(eta)
# ```
#
# **Stability note**: when $\eta$ becomes large and negative, $H + \eta$
# can approach zero or go negative, causing the model to blow up.  In
# practice we monitor the minimum depth to ensure it stays positive.

# %%
# Demonstrate the nonlinear continuity tendency
layer_depth = mean_depth + eta
eta_rhs_nonlinear = adv(layer_depth, u, v, method="upwind1")
eta_rhs_nonlinear = eta_rhs_nonlinear + viscosity * diff.laplacian(eta)

# Compare with the linear version
eta_rhs_linear = -mean_depth * diff.divergence(u, v) + viscosity * diff.laplacian(eta)

print(
    f"Nonlinear eta_rhs max: {float(jnp.abs(eta_rhs_nonlinear[1:-1, 1:-1]).max()):.4e}"
)
print(f"Linear    eta_rhs max: {float(jnp.abs(eta_rhs_linear[1:-1, 1:-1]).max()):.4e}")

# %% [markdown]
# ## 6. Momentum Advection
#
# The nonlinear momentum equations add the terms:
#
# $$
# -(\mathbf{u} \cdot \nabla) u
#   = -\left(u\,\frac{\partial u}{\partial x}
#     + v\,\frac{\partial u}{\partial y}\right)
# $$
#
# $$
# -(\mathbf{u} \cdot \nabla) v
#   = -\left(u\,\frac{\partial v}{\partial x}
#     + v\,\frac{\partial v}{\partial y}\right)
# $$
#
# **Implementation**: we interpolate $u$ and $v$ to T-points, compute
# centred gradients there, form the dot product, then interpolate back
# to the correct staggered position (U or V).
#
# **Why not use the Bernoulli pressure gradient?**
# The full Bernoulli function is
# $B = g\eta + \tfrac{1}{2}(u^2 + v^2)$.  Its gradient includes
# $\nabla(\tfrac{1}{2}|\mathbf{u}|^2)$, which is *already contained*
# in the advection term $(\mathbf{u} \cdot \nabla)\mathbf{u}$ via the
# vector identity:
#
# $$
# (\mathbf{u} \cdot \nabla)\mathbf{u}
#   = \nabla\!\left(\tfrac{1}{2}|\mathbf{u}|^2\right)
#     + (\nabla \times \mathbf{u}) \times \mathbf{u}
# $$
#
# Using $-\nabla B$ together with $-(\mathbf{u}\cdot\nabla)\mathbf{u}$
# would double-count the kinetic-energy gradient.  So we keep the
# pressure gradient as simply $-g\,\nabla\eta$.

# %%
# Demonstrate momentum advection computation


def centered_gradients(field):
    """Centred gradients at T-points via difference + re-averaging."""
    df_dx = interp.U_to_T(diff.diff_x_T_to_U(field))
    df_dy = interp.V_to_T(diff.diff_y_T_to_V(field))
    return df_dx, df_dy


# Interpolate velocities to T-points
u_on_t = interp.U_to_T(u)
v_on_t = interp.V_to_T(v)

# Gradients of u and v at T-points
du_dx, du_dy = centered_gradients(u_on_t)
dv_dx, dv_dy = centered_gradients(v_on_t)

# Advection: -(u . grad) u at U-points, -(u . grad) v at V-points
u_adv = interp.T_to_U(u_on_t * du_dx + v_on_t * du_dy)  # -> U-points
v_adv = interp.T_to_V(u_on_t * dv_dx + v_on_t * dv_dy)  # -> V-points

print(f"u_adv shape: {u_adv.shape}  (at U-points)")
print(f"v_adv shape: {v_adv.shape}  (at V-points)")

# %% [markdown]
# ## 7. Tendency Function
#
# The full right-hand side assembles all terms.  Each line is annotated
# with the corresponding equation term.

# %%


def tendency(eta_field, u_field, v_field):
    """Compute the nonlinear shallow-water tendencies on the C-grid.

    Returns (d_eta/dt, du/dt, dv/dt), all at their native stagger points.
    """
    # --- Continuity: d(eta)/dt = -div((H+eta)*u) + nu*lap(eta) ---
    layer_depth = mean_depth + eta_field  # H + eta         [Ny, Nx]
    eta_rhs = adv(
        layer_depth,
        u_field,
        v_field,  # -div((H+eta)*u) [Ny, Nx]
        method="upwind1",
    )
    eta_rhs = eta_rhs + viscosity * diff.laplacian(eta_field)  # + nu*lap(eta)

    # --- Momentum advection: -(u . grad) u, -(u . grad) v ---
    u_on_t = interp.U_to_T(u_field)  # u at T-points   [Ny, Nx]
    v_on_t = interp.V_to_T(v_field)  # v at T-points   [Ny, Nx]

    du_dx, du_dy = centered_gradients(u_on_t)  # grad(u) at T    [Ny, Nx]
    dv_dx, dv_dy = centered_gradients(v_on_t)  # grad(v) at T    [Ny, Nx]
    u_adv = interp.T_to_U(u_on_t * du_dx + v_on_t * du_dy)  # -> U  [Ny, Nx]
    v_adv = interp.T_to_V(u_on_t * dv_dx + v_on_t * dv_dy)  # -> V  [Ny, Nx]

    # --- Coriolis: f*v at U-points, -f*u at V-points ---
    v_on_u = interp.V_to_U(v_field)  # v at U-points   [Ny, Nx]
    u_on_v = interp.U_to_V(u_field)  # u at V-points   [Ny, Nx]
    coriolis_on_u = interp.T_to_U(coriolis)  # f at U-points   [Ny, Nx]
    coriolis_on_v = interp.T_to_V(coriolis)  # f at V-points   [Ny, Nx]

    # --- u-momentum: -g*d(eta)/dx - adv + f*v + F_x - r*u + nu*lap(u) ---
    u_rhs = -gravity * diff.diff_x_T_to_U(eta_field)  # pressure grad
    u_rhs = u_rhs - u_adv  # - (u . grad) u
    u_rhs = u_rhs + coriolis_on_u * v_on_u  # + f*v
    u_rhs = u_rhs + wind_u  # + F_x
    u_rhs = u_rhs - drag * u_field  # - r*u
    u_rhs = u_rhs + viscosity * diff.laplacian(u_field)  # + nu*lap(u)

    # --- v-momentum: -g*d(eta)/dy - adv - f*u - r*v + nu*lap(v) ---
    v_rhs = -gravity * diff.diff_y_T_to_V(eta_field)  # pressure grad
    v_rhs = v_rhs - v_adv  # - (u . grad) v
    v_rhs = v_rhs - coriolis_on_v * u_on_v  # - f*u
    v_rhs = v_rhs - drag * v_field  # - r*v
    v_rhs = v_rhs + viscosity * diff.laplacian(v_field)  # + nu*lap(v)

    return eta_rhs, u_rhs, v_rhs


# Quick check: the tendency should be well-defined
d_eta, d_u, d_v = tendency(eta, u, v)
print(f"d_eta max: {float(jnp.abs(d_eta[1:-1, 1:-1]).max()):.4e}")
print(f"d_u   max: {float(jnp.abs(d_u[1:-1, 1:-1]).max()):.4e}")
print(f"d_v   max: {float(jnp.abs(d_v[1:-1, 1:-1]).max()):.4e}")

# %% [markdown]
# ## 8. Time Stepping
#
# We use `finitevolx.heun_step` (Heun/RK2 predictor-corrector) and
# run a short integration of 500 steps.  After each step, we re-apply
# wall boundary conditions (zero ghost cells) using `fvx.pad_interior`.

# %%


def apply_bc(state):
    """Re-apply solid-wall BCs (zero ghost cells)."""
    return tuple(fvx.pad_interior(f, mode="constant") for f in state)


def rhs(state):
    """Tendency wrapper for heun_step (expects/returns tuples)."""
    return tendency(*apply_bc(state))


@jax.jit
def step(eta_f, u_f, v_f):
    """One Heun step + boundary condition reset."""
    state = fvx.heun_step((eta_f, u_f, v_f), rhs, dt)
    return apply_bc(state)


# --- Short integration: 500 steps ---
n_steps = 500
print(f"Integrating {n_steps} steps ({n_steps * dt / 86400:.2f} days) ...")

for _i in range(n_steps):
    eta, u, v = step(eta, u, v)

# Block until computation is done
eta.block_until_ready()
print("Done.")

# Check minimum total depth (stability diagnostic)
min_depth = float((mean_depth + eta[1:-1, 1:-1]).min())
print(f"Minimum total depth: {min_depth:.1f} m  (must stay > 0)")

# %% [markdown]
# ## 9. Results
#
# We plot four diagnostics:
#
# 1. **Free-surface anomaly** $\eta$ -- the sea-surface height perturbation.
# 2. **Speed** $|\mathbf{u}|$ -- flow magnitude at T-points.
# 3. **Relative vorticity** $\zeta = \partial v/\partial x - \partial u/\partial y$.
# 4. **Total depth** $H + \eta$ -- must remain positive everywhere.
#
# **Boundary slicing:** Native T-point fields ($\eta$, total depth) use
# `[1:-1, 1:-1]`.  Interpolated/derived fields (speed, vorticity) use
# `[2:-2, 2:-2]` to avoid ghost-cell contamination at the boundary.

# %%
# Compute diagnostics on the interior
u_center = interp.U_to_T(u)
v_center = interp.V_to_T(v)

zeta_corner = vort.relative_vorticity(u, v)
zeta_center = interp.X_to_T(zeta_corner)

# --- Coordinate grids ---
# Full interior [1:-1, 1:-1] for native T-point fields (eta, total depth)
x_full_1d, y_full_1d = x_1d, y_1d
# Inner domain [2:-2, 2:-2] for interpolated/derived fields (speed, vorticity)
# Trims the first interior row/column where ghost-cell averaging contaminates values.
x_inner_1d, y_inner_1d = x_1d[1:-1], y_1d[1:-1]

# Native T-point fields: [1:-1, 1:-1]
eta_np = np.asarray(eta[1:-1, 1:-1])
depth_np = mean_depth + eta_np

# Interpolated fields: [2:-2, 2:-2] to avoid boundary artifacts
u_np = np.asarray(u_center[2:-2, 2:-2])
v_np = np.asarray(v_center[2:-2, 2:-2])
speed_np = np.sqrt(u_np**2 + v_np**2)
zeta_np = np.asarray(zeta_center[2:-2, 2:-2])

# Energy diagnostics
total_ke = 0.5 * float(jnp.sum(u_center[2:-2, 2:-2] ** 2 + v_center[2:-2, 2:-2] ** 2)) * dx * dy
total_pe = 0.5 * gravity * float(jnp.sum(eta[1:-1, 1:-1] ** 2)) * dx * dy
print(f"Total KE:   {total_ke:.4e} m^4 s^-2")
print(f"Total PE:   {total_pe:.4e} m^4 s^-2")

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Free-surface anomaly -- native T-point, full interior
vmax_eta = float(np.abs(eta_np).max())
if vmax_eta == 0.0:
    vmax_eta = 1.0
im0 = axes[0, 0].pcolormesh(
    x_full_1d / 1e6,
    y_full_1d / 1e6,
    eta_np,
    shading="auto",
    cmap="RdBu_r",
    vmin=-vmax_eta,
    vmax=vmax_eta,
)
axes[0, 0].set_title(
    f"Free-surface anomaly $\\eta$ [m]\nmax |$\\eta$| = {vmax_eta:.4f}"
)
fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

# (b) Speed -- interpolated, inner domain [2:-2, 2:-2]
im1 = axes[0, 1].pcolormesh(
    x_inner_1d / 1e6,
    y_inner_1d / 1e6,
    speed_np,
    shading="auto",
    cmap="viridis",
)
axes[0, 1].set_title(f"Speed |u| [m/s]\nmax = {float(speed_np.max()):.4e}")
fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

# (c) Relative vorticity -- interpolated, inner domain [2:-2, 2:-2]
vmax_z = float(np.abs(zeta_np).max())
if vmax_z == 0.0:
    vmax_z = 1.0
im2 = axes[1, 0].pcolormesh(
    x_inner_1d / 1e6,
    y_inner_1d / 1e6,
    zeta_np,
    shading="auto",
    cmap="RdBu_r",
    vmin=-vmax_z,
    vmax=vmax_z,
)
axes[1, 0].set_title(f"Relative vorticity $\\zeta$ [s$^{{-1}}$]\nmax = {vmax_z:.4e}")
fig.colorbar(im2, ax=axes[1, 0], shrink=0.8)

# (d) Total depth -- native T-point, full interior
im3 = axes[1, 1].pcolormesh(
    x_full_1d / 1e6,
    y_full_1d / 1e6,
    depth_np,
    shading="auto",
    cmap="cividis",
)
axes[1, 1].set_title(f"Total depth $H + \\eta$ [m]\nmin = {float(depth_np.min()):.1f}")
fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)

for ax in axes.flat:
    ax.set_xlabel("x [$\\times 10^6$ m]")
    ax.set_ylabel("y [$\\times 10^6$ m]")
    ax.set_aspect("equal")

t_days = n_steps * dt / 86400.0
fig.suptitle(
    f"Nonlinear Shallow Water -- {nx}$\\times${ny} grid, t = {t_days:.1f} days",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()
fig.savefig(IMG_DIR / "shallow_water_results.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Full Simulation
#
# The production script at `scripts/shallow_water.py` runs a longer
# integration (150 000 spin-up steps + 15 000 recording steps) on a
# 64 x 64 grid and saves an animated GIF of the free-surface evolution.
#
# ```python
# # Full nonlinear shallow-water configuration:
# #   nx, ny = 64, 64
# #   Lx, Ly = 5.12e6, 5.12e6
# #   dt = 200 s, spinup = 150000 steps, record = 15000 steps
# #   Same physics: g=9.81, H=500, f0=9.375e-5, beta=1.754e-11
# #   New: Advection2D for continuity, centered advection for momentum
# ```
#
# ![Nonlinear shallow-water double gyre](../docs/images/shallow_water/shallow_water_double_gyre.gif)

# %% [markdown]
# ## Summary
#
# ### Linear vs Nonlinear Comparison
#
# | Feature | Linear SWE | Nonlinear SWE |
# |---------|-----------|---------------|
# | Continuity | $-H\,\nabla\cdot\mathbf{u}$ | $-\nabla\cdot[(H+\eta)\,\mathbf{u}]$ |
# | Momentum advection | none | $-(\mathbf{u}\cdot\nabla)\mathbf{u}$ |
# | Pressure gradient | $-g\,\nabla\eta$ | $-g\,\nabla\eta$ |
# | Stability concern | unconditional (linear) | min depth $H+\eta > 0$ required |
# | New operator | -- | `Advection2D` |
# | Computational cost | lower | higher (advection + gradients) |
#
# ### finitevolx API Reference
#
# | Class / Function | Role in this notebook |
# |-----------------|----------------------|
# | `ArakawaCGrid2D.from_interior` | Build the C-grid with ghost ring |
# | `Difference2D` | Finite differences, Laplacian, divergence |
# | `Interpolation2D` | Stagger-point averaging (T/U/V/X) |
# | `Advection2D` | Upwind flux advection $-\nabla\cdot(h\,\mathbf{u})$ |
# | `Vorticity2D` | Relative vorticity at corner (X) points |
# | `heun_step` | Heun/RK2 time integrator |
# | `pad_interior` | Re-apply ghost-cell boundary conditions |
