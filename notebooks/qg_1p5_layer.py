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
# # 1.5-Layer Quasi-Geostrophic Double-Gyre Model
#
# This notebook builds a **1.5-layer quasi-geostrophic (QG) model** step by
# step using finitevolx operators on the Arakawa C-grid.  The physical setup
# is an active upper layer of depth $H$ over an infinitely deep, resting
# abyssal layer.  The density difference between the two layers defines a
# **reduced gravity** $g'$, which sets the internal Rossby deformation
# radius $L_d$.
#
# The model is forced by a classical **double-gyre wind-curl** pattern and
# dissipated by bottom drag and lateral viscosity.  This is the simplest
# system that produces the western-intensified gyres, inertial recirculation,
# and mesoscale eddies that characterise mid-latitude ocean basins.
#
# ## Governing equations (Formulation B -- PV anomaly)
#
# The prognostic variable is the **PV anomaly**
#
# $$q_a = \zeta - \frac{\psi}{L_d^2}$$
#
# where $\zeta = \nabla^2 \psi$ is the relative vorticity and $\psi$ is the
# streamfunction.  The resting state $\psi = 0$ gives $q_a = 0$.  The
# evolution equation is
#
# $$\partial_t q_a = -\mathbf{u} \cdot \nabla q_a \;-\; \beta\,v \;+\; F
#     \;-\; r\,\zeta \;+\; \nu\,\nabla^2 q_a$$
#
# The streamfunction is recovered from the PV anomaly by inverting the
# **Helmholtz equation**
#
# $$\left(\nabla^2 - \frac{1}{L_d^2}\right)\psi = q_a$$
#
# and the geostrophic velocity follows from
#
# $$u = -\frac{\partial \psi}{\partial y}, \qquad
#   v = +\frac{\partial \psi}{\partial x}$$
#
# ### Term-by-term breakdown
#
# | Term | Expression | Physical meaning |
# |------|-----------|------------------|
# | **Advection** | $-\mathbf{u} \cdot \nabla q_a$ | Nonlinear transport of PV by the geostrophic flow |
# | **Beta effect** | $-\beta\,v$ | Planetary-vorticity advection (Rossby-wave restoring) |
# | **Wind forcing** | $F(y)$ | Double-gyre wind-curl input of PV |
# | **Bottom drag** | $-r\,\zeta$ | Linear friction on relative vorticity (Ekman spin-down) |
# | **Diffusion** | $\nu\,\nabla^2 q_a$ | Laplacian viscosity for grid-scale dissipation |

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

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "qg_1p5_layer"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. The Rossby Deformation Radius
#
# The **first baroclinic Rossby radius of deformation** sets the length
# scale at which rotation and stratification compete:
#
# $$L_d = \frac{\sqrt{g' H}}{f_0}$$
#
# where $g' = g\,\Delta\rho/\rho_0$ is the reduced gravity across the
# interface, $H$ is the active-layer thickness, and $f_0$ is the reference
# Coriolis parameter.  Motions at scales $\gg L_d$ are dominated by
# rotation and behave barotropically; motions at scales $\sim L_d$ feel the
# stratification and develop baroclinic instability.
#
# For mid-latitude ocean basins, $L_d \approx 30$--$50$ km.  We use
# $L_d = 40$ km here.
#
# | Parameter | Symbol | Value | Units | Meaning |
# |-----------|--------|-------|-------|---------|
# | Domain length x | $L_x$ | $5.12 \times 10^6$ | m | Zonal basin extent (~5000 km) |
# | Domain length y | $L_y$ | $5.12 \times 10^6$ | m | Meridional basin extent |
# | Coriolis parameter | $f_0$ | $9.375 \times 10^{-5}$ | s$^{-1}$ | Reference rotation rate (~30 N) |
# | Beta | $\beta$ | $1.754 \times 10^{-11}$ | m$^{-1}$ s$^{-1}$ | Meridional Coriolis gradient |
# | Rossby radius | $L_d$ | $4 \times 10^4$ | m | Deformation radius (40 km) |
# | Bottom drag | $r$ | $5 \times 10^{-8}$ | s$^{-1}$ | Ekman spin-down rate |
# | Viscosity | $\nu$ | $5 \times 10^4$ | m$^2$ s$^{-1}$ | Laplacian diffusivity |
# | Wind amplitude | $F_0$ | $2 \times 10^{-12}$ | s$^{-2}$ | Peak PV forcing |
# | Time step | $\Delta t$ | 4000 | s | ~1.1 hours |
#
# ### 1.5-layer setup (ASCII diagram)
#
# ```
#   ~~~ wind stress tau(y) ~~~
#   ===========================  sea surface (z = 0)
#   |                         |
#   |   Active layer (H)      |  density rho_1
#   |   psi, q_a, u, v        |
#   |                         |
#   ---------------------------  interface (z = -H)
#   |                         |
#   |   Deep resting layer    |  density rho_2 > rho_1
#   |   (infinitely deep,     |
#   |    no motion)            |
#   |                         |
#   ===========================  ocean floor
#
#   Reduced gravity: g' = g * (rho_2 - rho_1) / rho_0
#   Deformation radius: Ld = sqrt(g'*H) / f0
# ```

# %% [markdown]
# ## 2. C-Grid Placement for QG
#
# On the Arakawa C-grid, the QG variables live at specific stagger
# locations.  The PV anomaly $q_a$ and streamfunction $\psi$ are
# **T-point** (cell-centre) quantities, while the velocity components
# sit on the appropriate faces:
#
# ```
#     X ── V ── X ── V ── X
#     |         |         |
#     U   T,q   U   T,q   U     T = psi, q_a (cell centre)
#     |   psi   |   psi   |     U = u-velocity (east/west face)
#     X ── V ── X ── V ── X     V = v-velocity (north/south face)
#     |         |         |     X = relative vorticity zeta (corner)
#     U   T,q   U   T,q   U
#     |   psi   |   psi   |
#     X ── V ── X ── V ── X
# ```
#
# The relative vorticity $\zeta = \partial v/\partial x - \partial u/\partial y$
# naturally lives at X-points (corners), and must be interpolated to T-points
# for the drag term.  The full grid array has shape `(ny+2, nx+2)` with
# a one-cell ghost ring encoding the solid-wall boundary condition
# $\psi = 0$ on all four walls.

# %% [markdown]
# ## 3. Grid and Parameters

# %%
# --- Grid ---
nx, ny = 32, 32
Lx, Ly = 5.12e6, 5.12e6  # 5120 km square basin

grid = fvx.ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
dx, dy = grid.dx, grid.dy
Ny, Nx = ny + 2, nx + 2

print(f"Interior: {nx} x {ny}")
print(f"Full array: ({Ny}, {Nx})")
print(f"dx = {dx/1e3:.1f} km,  dy = {dy/1e3:.1f} km")

# --- Physical parameters ---
f0 = 9.375e-5       # Coriolis parameter [s^-1]
beta = 1.754e-11     # beta-plane gradient [m^-1 s^-1]
Ld = 4.0e4           # Rossby deformation radius [m] (40 km)
drag = 5.0e-8        # bottom drag [s^-1]
viscosity = 5.0e4    # Laplacian viscosity [m^2 s^-1]
wind_amp = 2.0e-12   # peak wind-curl PV forcing [s^-2]
dt = 4000.0          # time step [s]

# Helmholtz parameter: lambda = 1/Ld^2
lambda_helmholtz = 1.0 / Ld**2

print(f"\nLd = {Ld/1e3:.0f} km")
print(f"Ld / dx = {Ld / dx:.2f}  (grid points per deformation radius)")
print(f"lambda = 1/Ld^2 = {lambda_helmholtz:.2e} m^-2")
print(f"dt = {dt:.0f} s = {dt/3600:.1f} hours")

# %% [markdown]
# ## 4. Wind-Curl Forcing
#
# The double-gyre wind pattern drives an anticyclonic (clockwise)
# **subtropical gyre** in the southern half and a cyclonic
# (counterclockwise) **subpolar gyre** in the northern half:
#
# $$F(y) = -F_0 \sin\!\left(\frac{2\pi\,y}{L_y}\right)$$
#
# This is negative in the southern half ($0 < y < L_y/2$, driving
# anticyclonic circulation) and positive in the northern half
# ($L_y/2 < y < L_y$, driving cyclonic circulation).  The two gyres
# share a boundary at the mid-basin latitude, where the Gulf-Stream-like
# western boundary current separates from the coast.

# %%
# Interior cell-centre coordinates
x_centers = (jnp.arange(nx) + 0.5) * dx
y_centers = (jnp.arange(ny) + 0.5) * dy
_, y2d = jnp.meshgrid(x_centers, y_centers)

# Double-gyre wind-curl forcing at interior T-points
wind_curl_interior = -wind_amp * jnp.sin(2.0 * jnp.pi * y2d / Ly)

# Pad to full grid with zero ghost ring (wall BCs)
wind_curl = jnp.pad(wind_curl_interior, pad_width=1, mode="constant")

print(f"Wind curl shape (interior): {wind_curl_interior.shape}")
print(f"Wind curl shape (full):     {wind_curl.shape}")
print(f"Wind curl range: [{float(wind_curl_interior.min()):.2e}, "
      f"{float(wind_curl_interior.max()):.2e}] s^-2")

# %%
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolormesh(
    np.asarray(x_centers / 1e6),
    np.asarray(y_centers / 1e6),
    np.asarray(wind_curl_interior),
    cmap="RdBu_r",
    shading="auto",
)
ax.set_xlabel("x [10$^3$ km]")
ax.set_ylabel("y [10$^3$ km]")
ax.set_title("Double-gyre wind-curl forcing $F(y)$")
ax.axhline(Ly / 2 / 1e6, color="k", ls="--", lw=0.8, label="gyre boundary")
ax.legend(fontsize=9)
fig.colorbar(im, ax=ax, label="PV forcing [s$^{-2}$]")
fig.savefig(IMG_DIR / "wind_curl_forcing.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# The forcing is antisymmetric about the basin midline.  The southern
# (blue) half receives negative PV input (anticyclonic), and the northern
# (red) half receives positive PV input (cyclonic).

# %% [markdown]
# ## 5. Streamfunction Inversion
#
# The key diagnostic step in QG is **inverting the PV anomaly for the
# streamfunction**.  Given $q_a$ at T-points, we solve the Helmholtz
# equation
#
# $$\left(\nabla^2 - \frac{1}{L_d^2}\right)\psi = q_a$$
#
# with homogeneous Dirichlet boundary conditions $\psi = 0$ on all four
# basin walls (no-normal-flow).
#
# finitevolx provides `solve_helmholtz_dst` for exactly this problem.
# The DST-I (Type I discrete sine transform) is the correct choice because
# the ghost-cell convention places $\psi = 0$ at the grid vertices -- the
# ghost cells themselves encode the Dirichlet condition.
#
# The `lambda_` parameter is $1/L_d^2$, which enters the Helmholtz
# equation as the screening term that couples the streamfunction to the
# interface displacement.

# %%
diff_op = fvx.Difference2D(grid)
interp_op = fvx.Interpolation2D(grid)


def invert_streamfunction(q_field):
    """Solve (nabla^2 - 1/Ld^2) psi = q_a with psi=0 on walls."""
    rhs = q_field[1:-1, 1:-1]
    psi_interior = fvx.solve_helmholtz_dst(rhs, dx, dy, lambda_=lambda_helmholtz)
    psi_full = jnp.zeros_like(q_field)
    return psi_full.at[1:-1, 1:-1].set(psi_interior)

# %%
# --- Demonstrate: create a test PV anomaly, invert, and plot ---
x2d_full, y2d_full = jnp.meshgrid(
    (jnp.arange(Nx) + 0.5) * dx,
    (jnp.arange(Ny) + 0.5) * dy,
)
q_test = 5.0e-9 * jnp.sin(2.0 * jnp.pi * x2d_full / Lx) * jnp.sin(jnp.pi * y2d_full / Ly)
# Zero out ghost ring
q_test = q_test.at[0, :].set(0.0).at[-1, :].set(0.0)
q_test = q_test.at[:, 0].set(0.0).at[:, -1].set(0.0)

psi_test = invert_streamfunction(q_test)

print(f"q_a  range: [{float(q_test.min()):.2e}, {float(q_test.max()):.2e}]")
print(f"psi  range: [{float(psi_test.min()):.2e}, {float(psi_test.max()):.2e}]")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

vmax_q = float(jnp.abs(q_test[1:-1, 1:-1]).max())
axes[0].imshow(
    np.asarray(q_test[1:-1, 1:-1]), origin="lower", cmap="RdBu_r",
    aspect="auto", vmin=-vmax_q, vmax=vmax_q,
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[0].set_title("Test PV anomaly $q_a$")

vmax_p = float(jnp.abs(psi_test[1:-1, 1:-1]).max())
axes[1].imshow(
    np.asarray(psi_test[1:-1, 1:-1]), origin="lower", cmap="RdBu_r",
    aspect="auto", vmin=-vmax_p, vmax=vmax_p,
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
axes[1].set_title(r"Recovered $\psi$ from Helmholtz inversion")

for ax in axes:
    ax.set_xlabel("x [10$^3$ km]")
    ax.set_ylabel("y [10$^3$ km]")

fig.tight_layout()
fig.savefig(IMG_DIR / "helmholtz_inversion_demo.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# The Helmholtz inversion smears the PV anomaly pattern slightly due to
# the screening term $-\psi/L_d^2$: the streamfunction is smoother and
# weaker than it would be for a pure Poisson ($L_d \to \infty$) inversion.

# %% [markdown]
# ## 6. Geostrophic Velocity Recovery
#
# From the streamfunction $\psi$ at T-points, we recover the geostrophic
# velocity:
#
# $$u = -\frac{\partial \psi}{\partial y}, \qquad
#   v = +\frac{\partial \psi}{\partial x}$$
#
# On the C-grid, $\psi$ is first averaged to X-points (corners), and then
# the orthogonal derivatives place $u$ on U-points and $v$ on V-points:
#
# ```
# Step 1: psi (T-point) --[avg to corners]--> psi_X (X-point)
# Step 2: u = -d(psi_X)/dy  at U-points    shape: (Ny, Nx)
#          v = +d(psi_X)/dx  at V-points    shape: (Ny, Nx)
# ```

# %%
def geostrophic_velocity(psi_field):
    """Recover (u, v) from streamfunction at T-points."""
    psi_on_x = interp_op.T_to_X(psi_field)
    u_field = -diff_op.diff_y_X_to_U(psi_on_x)
    v_field = diff_op.diff_x_X_to_V(psi_on_x)
    return u_field, v_field


u_test, v_test = geostrophic_velocity(psi_test)

print(f"psi shape:  {psi_test.shape}  (full grid, T-points)")
print(f"u shape:    {u_test.shape}    (full grid, U-points)")
print(f"v shape:    {v_test.shape}    (full grid, V-points)")
print(f"max |u| = {float(jnp.abs(u_test).max()):.4e} m/s")
print(f"max |v| = {float(jnp.abs(v_test).max()):.4e} m/s")

# %% [markdown]
# ## 7. PV Tendency Function
#
# We now assemble the full PV tendency, line by line.  Each physical
# process maps onto a specific finitevolx operation.

# %%
adv_op = fvx.Advection2D(grid=grid)
vort_op = fvx.Vorticity2D(grid=grid)


def tendency(q_field):
    """Compute dq_a/dt and return (tendency, psi, u, v).

    Steps:
      1. Invert q_a -> psi  (Helmholtz)
      2. psi -> (u, v)      (geostrophic balance)
      3. PV advection:      -u . grad(q_a)
      4. Beta term:         -beta * v  (at T-points)
      5. Wind forcing:      F(y)       (constant in time)
      6. Bottom drag:       -r * zeta  (drag on VORTICITY, not PV)
      7. Diffusion:         nu * laplacian(q_a)
    """
    # 1. Invert PV -> streamfunction
    psi_field = invert_streamfunction(q_field)

    # 2. Geostrophic velocity
    u_field, v_field = geostrophic_velocity(psi_field)

    # 3. PV advection: -u . grad(q_a)  (writes to interior [2:-2, 2:-2])
    q_rhs = adv_op(q_field, u_field, v_field, method="upwind1")

    # 4. Beta term: -beta * v at interior T-points
    #    We interpolate v from V-points to T-points first.
    v_center = interp_op.V_to_T(v_field)
    q_rhs = q_rhs.at[1:-1, 1:-1].add(-beta * v_center[1:-1, 1:-1])

    # 5. Wind forcing (already on full grid with zero ghosts)
    q_rhs = q_rhs + wind_curl

    # 6. Bottom drag: -r * zeta (NOT -r * q_a!)
    #    Drag acts on relative vorticity zeta = laplacian(psi).
    #    If we dragged q_a instead, we would get an extra +r*psi/Ld^2 term
    #    that acts as anti-damping on the interface displacement.
    zeta = diff_op.laplacian(psi_field)
    q_rhs = q_rhs - drag * zeta

    # 7. Diffusion: nu * laplacian(q_a)
    q_rhs = q_rhs + viscosity * diff_op.laplacian(q_field)

    return q_rhs, psi_field, u_field, v_field


def apply_bc(q_field):
    """Re-enforce wall BCs: zero ghost ring (psi = 0 on walls)."""
    return fvx.pad_interior(q_field, mode="constant")


def pv_tendency(q_field):
    """Tendency function compatible with heun_step: q -> dq/dt."""
    q_rhs, _, _, _ = tendency(apply_bc(q_field))
    return q_rhs

# %% [markdown]
# **Why drag on $\zeta$, not $q_a$?**
#
# The PV anomaly is $q_a = \zeta - \psi/L_d^2$.  If we write the drag
# as $-r\,q_a$, we get $-r\,\zeta + r\,\psi/L_d^2$.  The second term
# is a positive feedback: it *amplifies* the interface displacement
# instead of damping it.  Physical bottom drag acts on the flow velocity
# (and hence on $\zeta = \nabla^2 \psi$), not on the full PV.

# %% [markdown]
# ## 8. Time Stepping
#
# We use the **Heun method** (explicit RK2 predictor-corrector) from
# `finitevolx.heun_step`.  This is second-order accurate and stable for
# advection-dominated problems with a suitable CFL constraint.

# %%
@jax.jit
def step(q_field):
    """Advance one Heun time step and diagnose the balanced state."""
    q_next = apply_bc(fvx.heun_step(q_field, pv_tendency, dt))
    psi_next = invert_streamfunction(q_next)
    u_next, v_next = geostrophic_velocity(psi_next)
    return q_next, psi_next, u_next, v_next

# %% [markdown]
# ### Short demo run
#
# We run a brief integration (~500 steps) from a small sinusoidal PV
# perturbation to demonstrate the model mechanics.  This is too short
# for full spin-up, but enough to see the wind forcing begin to organise
# the circulation into the double-gyre pattern.

# %%
# --- Initial condition: small sinusoidal PV anomaly ---
q0_interior = 5.0e-9 * jnp.sin(
    2.0 * jnp.pi * x_centers[None, :] / Lx
) * jnp.sin(
    jnp.pi * y_centers[:, None] / Ly
)
q = jnp.pad(q0_interior, pad_width=1, mode="constant")

psi = invert_streamfunction(q)
u, v = geostrophic_velocity(psi)

print(f"Initial max |q_a| = {float(jnp.abs(q).max()):.2e}")
print(f"Initial max |psi| = {float(jnp.abs(psi).max()):.2e}")

# %%
n_steps = 500

print(f"Running {n_steps} steps  ({n_steps * dt / 86400:.1f} days) ...")
for i in range(n_steps):
    q, psi, u, v = step(q)
    if (i + 1) % 100 == 0:
        max_q = float(jnp.abs(q[1:-1, 1:-1]).max())
        max_psi = float(jnp.abs(psi[1:-1, 1:-1]).max())
        print(f"  step {i+1:4d}:  max|q_a| = {max_q:.3e},  max|psi| = {max_psi:.3e}")

print("Done.")

# %% [markdown]
# ## 9. Results
#
# We plot the final state as a 4-panel figure: PV anomaly, SSH (height),
# speed, and relative vorticity.
#
# **Boundary slicing:** Native T-point fields ($q_a$, $\eta$) use
# `[1:-1, 1:-1]`.  Interpolated/derived fields (speed, vorticity) use
# `[2:-2, 2:-2]` to avoid ghost-cell contamination at the boundary.

# %%
# Diagnose fields for plotting
u_center = interp_op.U_to_T(u)
v_center = interp_op.V_to_T(v)
zeta_corner = vort_op.relative_vorticity(u, v)
zeta_center = interp_op.X_to_T(zeta_corner)

# --- Coordinate extents ---
# Full interior [1:-1, 1:-1] for native T-point fields (q, eta/SSH)
extent_full = [0, Lx / 1e6, 0, Ly / 1e6]
# Inner domain [2:-2, 2:-2] for interpolated/derived fields (speed, vorticity)
# Trims the first interior row/column where ghost-cell averaging contaminates values.
x_inner_lo = float(x_centers[1]) - 0.5 * dx
x_inner_hi = float(x_centers[-2]) + 0.5 * dx
y_inner_lo = float(y_centers[1]) - 0.5 * dy
y_inner_hi = float(y_centers[-2]) + 0.5 * dy
extent_inner = [x_inner_lo / 1e6, x_inner_hi / 1e6, y_inner_lo / 1e6, y_inner_hi / 1e6]

# Native T-point fields: [1:-1, 1:-1]
q_plot = np.asarray(q[1:-1, 1:-1])
psi_plot = np.asarray(psi[1:-1, 1:-1])
# Sea surface height: eta = (f0/g') * psi, using g' = (f0*Ld)^2 / H ~ f0^2 * Ld^2
# From QG: interface displacement eta = f0*psi / g' = psi / (f0 * Ld^2)
eta_plot = psi_plot / (f0 * Ld**2)

# Interpolated fields: [2:-2, 2:-2] to avoid boundary artifacts
speed_plot = np.asarray(jnp.sqrt(u_center[2:-2, 2:-2]**2 + v_center[2:-2, 2:-2]**2))
zeta_plot = np.asarray(zeta_center[2:-2, 2:-2])

# Energy diagnostics
total_ke = 0.5 * float(jnp.sum(u_center[2:-2, 2:-2] ** 2 + v_center[2:-2, 2:-2] ** 2)) * dx * dy
total_ape = 0.5 * float(jnp.sum(psi[1:-1, 1:-1] ** 2)) * dx * dy / Ld**2
total_enstrophy = 0.5 * float(jnp.sum(q[1:-1, 1:-1] ** 2)) * dx * dy
print(f"Total KE:        {total_ke:.4e} m^4 s^-2")
print(f"Total APE:       {total_ape:.4e} m^4 s^-2")
print(f"Total enstrophy: {total_enstrophy:.4e} m^2 s^-2")

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# (a) PV anomaly -- native T-point, full interior
vmax = float(np.abs(q_plot).max())
if vmax == 0:
    vmax = 1.0
axes[0, 0].imshow(
    q_plot, origin="lower", cmap="RdBu_r", aspect="auto",
    vmin=-vmax, vmax=vmax, extent=extent_full,
)
axes[0, 0].set_title(f"PV anomaly $q_a$ (max = {vmax:.2e})")

# (b) SSH eta = psi / (f0 * Ld^2) -- native T-point, full interior
vmax_eta = float(np.abs(eta_plot).max())
if vmax_eta == 0:
    vmax_eta = 1.0
im_eta = axes[0, 1].imshow(
    eta_plot, origin="lower", cmap="RdBu_r", aspect="auto",
    vmin=-vmax_eta, vmax=vmax_eta, extent=extent_full,
)
axes[0, 1].set_title(f"SSH $\\eta = \\psi / (f_0 L_d^2)$ [m]\nmax = {vmax_eta:.2e}")
fig.colorbar(im_eta, ax=axes[0, 1], shrink=0.8, label="m")

# (c) Speed -- interpolated, inner domain [2:-2, 2:-2]
im_spd = axes[1, 0].imshow(
    speed_plot, origin="lower", cmap="viridis", aspect="auto", extent=extent_inner,
)
axes[1, 0].set_title(f"Speed |u| (max = {float(speed_plot.max()):.3e} m/s)")
fig.colorbar(im_spd, ax=axes[1, 0], shrink=0.8, label="m s$^{-1}$")

# (d) Relative vorticity -- interpolated, inner domain [2:-2, 2:-2]
vmax = float(np.abs(zeta_plot).max())
if vmax == 0:
    vmax = 1.0
axes[1, 1].imshow(
    zeta_plot, origin="lower", cmap="RdBu_r", aspect="auto",
    vmin=-vmax, vmax=vmax, extent=extent_inner,
)
axes[1, 1].set_title(r"Relative vorticity $\zeta$")

for ax in axes.flat:
    ax.set_xlabel("x [10$^3$ km]")
    ax.set_ylabel("y [10$^3$ km]")

fig.suptitle(
    f"1.5-layer QG double gyre  |  t = {n_steps * dt / 86400:.1f} days  |  "
    f"{nx}x{ny} grid",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(IMG_DIR / "qg_1p5_layer_results.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# Even in this short run the double-gyre pattern is visible: the wind
# forcing has begun to spin up anticyclonic (negative $\psi$) circulation
# in the south and cyclonic (positive $\psi$) circulation in the north.

# %% [markdown]
# ## 10. Full Simulation
#
# A production run uses higher resolution and a long spin-up to reach
# statistical equilibrium.  Typical parameters:
#
# - **Grid**: 64 x 64 interior points
# - **Spin-up**: 5000 steps (~230 days of silent integration)
# - **Recording**: 10000 steps (~460 days), sampling every 1000 steps
#
# After spin-up, the western boundary currents intensify, the gyre
# separation develops inertial overshoot, and mesoscale eddies populate
# the inter-gyre region.
#
# ![1.5-layer QG double gyre](../docs/images/qg_1p5_layer/qg_1p5_layer_double_gyre.gif)

# %% [markdown]
# ## 11. Summary
#
# | Component | finitevolx API | Purpose |
# |-----------|---------------|---------|
# | **Grid** | `ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)` | C-grid with ghost ring |
# | **Helmholtz inversion** | `solve_helmholtz_dst(rhs, dx, dy, lambda_)` | $(\\nabla^2 - \\lambda)\\psi = q_a$ |
# | **Differences** | `Difference2D(grid)` | Finite differences, Laplacian |
# | **Interpolation** | `Interpolation2D(grid)` | T-to-X, V-to-T, U-to-T, X-to-T |
# | **Advection** | `Advection2D(grid)(q, u, v, method="upwind1")` | PV transport |
# | **Vorticity** | `Vorticity2D(grid).relative_vorticity(u, v)` | $\\zeta$ at X-points |
# | **Boundary conditions** | `pad_interior(q, mode="constant")` | Zero ghost ring (solid walls) |
# | **Time stepping** | `heun_step(q, tendency_fn, dt)` | Heun / RK2 predictor-corrector |
#
# ### Key modelling choices
#
# 1. **Formulation B** (PV anomaly): the prognostic variable $q_a = \zeta - \psi/L_d^2$
#    keeps the resting state at zero, simplifying initial and boundary conditions.
# 2. **Explicit beta term** ($-\beta v$): planetary-vorticity advection is handled
#    as a source term rather than absorbed into $q_a$.
# 3. **Drag on vorticity** ($-r\zeta$, not $-r q_a$): avoids spurious anti-damping
#    of the interface displacement.
# 4. **DST-I solver**: matches finitevolx's ghost-cell convention where $\psi = 0$
#    at the array boundary (vertex Dirichlet).
