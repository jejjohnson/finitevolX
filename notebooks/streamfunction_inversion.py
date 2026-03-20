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
# # Streamfunction Inversion: T-point vs X-point Placement
#
# The vorticity–streamfunction relation $\nabla^2 \psi = \zeta$ is the
# canonical elliptic inversion in geophysical fluid dynamics.  On the
# Arakawa C-grid, the **same physical equation** requires **different
# spectral solvers** depending on where $\psi$ lives:
#
# | Placement | Grid type | Solver | finitevolX API |
# |-----------|-----------|--------|----------------|
# | X-points (corners) | Regular (vertex-centred) | DST-I | `streamfunction_from_vorticity(bc="dst")` |
# | T-points (cell centres) | Staggered (cell-centred) | DST-II | `solve_poisson_dst2` (direct) |
#
# ## Where does $\psi$ live?
#
# ```
#   Option A: ψ at X-points (corners)     Option B: ψ at T-points (centres)
#   ─────────────────────────────────      ──────────────────────────────────
#
#   ψ ── · ── ψ ── · ── ψ                · ── · ── · ── · ── ·
#   |         |         |                  |         |         |
#   ·    ζ    ·    ζ    ·                  ·    ψ    ·    ψ    ·
#   |         |         |                  |    ζ    |    ζ    |
#   ψ ── · ── ψ ── · ── ψ                · ── · ── · ── · ── ·
#   |         |         |                  |         |         |
#   ·    ζ    ·    ζ    ·                  ·    ψ    ·    ψ    ·
#   |         |         |                  |    ζ    |    ζ    |
#   ψ ── · ── ψ ── · ── ψ                · ── · ── · ── · ── ·
#
#   ψ at corners, ζ at centres             ψ and ζ co-located at centres
#   Solver: DST-I (regular Dirichlet)      Solver: DST-II (staggered Dirichlet)
# ```
#
# This notebook compares both placements on a Lamb–Oseen vortex test case.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

import finitevolx as fvx
from spectraldiffx import solve_poisson_dst, solve_poisson_dst2

IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "images" / "streamfunction_inversion"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Analytical Vortex (Lamb–Oseen)
#
# We use a Gaussian vorticity field centred in a rectangular basin with
# $\psi = 0$ on the walls:
#
# $$
# \zeta(x, y) = \frac{\Gamma}{\pi r_0^2} \exp\!\left(-\frac{r^2}{r_0^2}\right)
# $$
#
# where $r^2 = (x - x_0)^2 + (y - y_0)^2$.

# %%
nx, ny = 128, 128
Lx, Ly = 2e6, 2e6  # 2000 km × 2000 km basin

grid = fvx.ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
dx, dy = grid.dx, grid.dy

# Vortex parameters
Gamma = 1e5       # circulation (m²/s)
r0 = 300e3        # vortex radius (300 km)
x0, y0 = Lx / 2, Ly / 2  # centre of domain

# T-point coordinates (cell centres, interior only)
i_idx = jnp.arange(nx)[None, :] + 0.5
j_idx = jnp.arange(ny)[:, None] + 0.5
x_T = i_idx * dx
y_T = j_idx * dy

r2 = (x_T - x0) ** 2 + (y_T - y0) ** 2
zeta = (Gamma / (jnp.pi * r0**2)) * jnp.exp(-r2 / r0**2)

print(f"max |ζ| = {float(jnp.abs(zeta).max()):.4e} s⁻¹")

# %%
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(
    np.asarray(zeta), origin="lower", cmap="RdBu_r",
    extent=[0, Lx / 1e6, 0, Ly / 1e6],
)
ax.set_xlabel("x (×10³ km)")
ax.set_ylabel("y (×10³ km)")
ax.set_title("Vorticity ζ (Lamb–Oseen vortex)")
fig.colorbar(im, ax=ax, shrink=0.8, label="s⁻¹")
fig.savefig(IMG_DIR / "vorticity.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. X-point Inversion (DST-I)
#
# The finitevolX convenience wrapper `streamfunction_from_vorticity`
# uses DST-I by default, which assumes $\psi$ lives at grid vertices
# (X-points/corners).

# %%
psi_X = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")

print(f"ψ_X range: [{float(psi_X.min()):.2e}, {float(psi_X.max()):.2e}]")

# %% [markdown]
# ## 3. T-point Inversion (DST-II)
#
# For $\psi$ co-located with $\zeta$ at T-points (cell centres), we use
# the staggered Dirichlet solver directly.

# %%
psi_T = solve_poisson_dst2(zeta, dx, dy)

print(f"ψ_T range: [{float(psi_T.min()):.2e}, {float(psi_T.max()):.2e}]")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

vmax = max(float(jnp.abs(psi_X).max()), float(jnp.abs(psi_T).max()))

for ax, psi, title in zip(
    axes,
    [psi_X, psi_T],
    ["ψ at X-points (DST-I)", "ψ at T-points (DST-II)"],
):
    im = ax.imshow(
        np.asarray(psi), origin="lower", cmap="RdBu_r",
        extent=[0, Lx / 1e6, 0, Ly / 1e6],
        vmin=-vmax, vmax=vmax,
    )
    X_plot = np.linspace(0, Lx / 1e6, nx)
    Y_plot = np.linspace(0, Ly / 1e6, ny)
    ax.contour(X_plot, Y_plot, np.asarray(psi), levels=12, colors="k", linewidths=0.5)
    ax.set_xlabel("x (×10³ km)")
    ax.set_ylabel("y (×10³ km)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Streamfunction from vorticity inversion", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(IMG_DIR / "streamfunction_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Velocity Recovery
#
# From each $\psi$ placement, we recover $(u, v)$ using C-grid differences.
#
# **From $\psi$ at X-points:**
# - $u = -\partial\psi/\partial y$ at U-points (X→U: backward y-difference)
# - $v = +\partial\psi/\partial x$ at V-points (X→V: backward x-difference)
#
# **From $\psi$ at T-points:**
# - $u = -\partial\psi/\partial y$ at U-points (T→U needs y-interpolation + difference)
# - $v = +\partial\psi/\partial x$ at V-points (T→V needs x-interpolation + difference)
#
# For simplicity, we use the full-grid Difference2D operator for both.

# %%
# Pad psi_X into full grid (with ghost ring)
psi_X_full = jnp.zeros((ny + 2, nx + 2))
psi_X_full = psi_X_full.at[1:-1, 1:-1].set(psi_X)

psi_T_full = jnp.zeros((ny + 2, nx + 2))
psi_T_full = psi_T_full.at[1:-1, 1:-1].set(psi_T)

diff_op = fvx.Difference2D(grid)

# From ψ at X-points: u = -dψ/dy, v = +dψ/dx
# For X-point ψ, the natural differences are X→U and X→V
# but Difference2D operates T→U and T→V, so this is approximate
u_from_X = -diff_op.diff_y_T_to_V(psi_X_full)
v_from_X = diff_op.diff_x_T_to_U(psi_X_full)

# From ψ at T-points: same T→U, T→V differences
u_from_T = -diff_op.diff_y_T_to_V(psi_T_full)
v_from_T = diff_op.diff_x_T_to_U(psi_T_full)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

fields = [
    (u_from_X, "u from ψ_X (DST-I)"),
    (u_from_T, "u from ψ_T (DST-II)"),
    (v_from_X, "v from ψ_X (DST-I)"),
    (v_from_T, "v from ψ_T (DST-II)"),
]

for ax, (field, title) in zip(axes.flat, fields):
    f = np.asarray(field[1:-1, 1:-1])
    vmax_f = np.abs(f).max()
    im = ax.imshow(
        f, origin="lower", cmap="RdBu_r",
        extent=[0, Lx / 1e6, 0, Ly / 1e6],
        vmin=-vmax_f, vmax=vmax_f,
    )
    ax.set_title(title)
    ax.set_xlabel("x (×10³ km)")
    ax.set_ylabel("y (×10³ km)")
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Velocity recovery from streamfunction", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(IMG_DIR / "velocity_recovery.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Error Analysis
#
# The difference between the two placements is $O(dx^2)$ in the interior,
# arising from the half-grid offset between vertex-centred and cell-centred
# points.

# %%
diff_psi = jnp.abs(psi_X - psi_T)
print(f"max |ψ_X - ψ_T| = {float(diff_psi.max()):.4e}")
print(f"relative diff   = {float(diff_psi.max() / jnp.abs(psi_T).max()):.4e}")

# %% [markdown]
# ### Convergence test
#
# We verify that both solvers converge at $O(dx^2)$ against a manufactured
# solution $\psi_{\text{exact}} = \sin(\pi x / L_x) \sin(\pi y / L_y)$
# which satisfies $\psi = 0$ on the boundary.

# %%
resolutions = [16, 32, 64, 128, 256]
errors_dst1 = []
errors_dst2 = []

for N in resolutions:
    dx_n = Lx / N
    dy_n = Ly / N

    # Manufactured solution: sin(πx/Lx)·sin(πy/Ly) satisfies Dirichlet BCs
    # Laplacian: -(π/Lx)² - (π/Ly)² times ψ
    i_n = jnp.arange(N)[None, :] + 0.5
    j_n = jnp.arange(N)[:, None] + 0.5
    x_n = i_n * dx_n
    y_n = j_n * dy_n

    psi_exact = jnp.sin(jnp.pi * x_n / Lx) * jnp.sin(jnp.pi * y_n / Ly)

    # Discrete FD Laplacian as RHS (what the spectral solvers actually invert)
    # For convergence against the continuous solution, use the continuous Laplacian
    k2 = (jnp.pi / Lx) ** 2 + (jnp.pi / Ly) ** 2
    rhs_n = -k2 * psi_exact

    psi_dst1 = solve_poisson_dst(rhs_n, dx_n, dy_n)
    psi_dst2 = solve_poisson_dst2(rhs_n, dx_n, dy_n)

    errors_dst1.append(float(jnp.abs(psi_dst1 - psi_exact).max()))
    errors_dst2.append(float(jnp.abs(psi_dst2 - psi_exact).max()))

# %%
fig, ax = plt.subplots(figsize=(8, 5.5))

dxs = [Lx / N for N in resolutions]
ax.loglog(dxs, errors_dst1, "o-", label="DST-I (regular)", lw=2)
ax.loglog(dxs, errors_dst2, "s--", label="DST-II (staggered)", lw=2)

# Reference O(dx²) line
dx_ref = np.array(dxs)
ref = errors_dst1[0] * (dx_ref / dx_ref[0]) ** 2
ax.loglog(dx_ref, ref, "k:", alpha=0.5, label="$O(dx^2)$ reference")

ax.set_xlabel("dx (m)")
ax.set_ylabel("max |ψ − ψ_exact|")
ax.set_title("Convergence: DST-I vs DST-II against continuous solution")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.savefig(IMG_DIR / "convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# Both solvers converge at **second order** ($O(dx^2)$).  This is expected:
# the spectral solvers invert the discrete $[1,-2,1]/dx^2$ Laplacian
# exactly, but the RHS was computed from the *continuous* $\nabla^2$.  The
# $O(dx^2)$ gap between discrete and continuous operators dominates.

# %% [markdown]
# ## 6. When to Use Which
#
# | | ψ at X-points (DST-I) | ψ at T-points (DST-II) |
# |---|---|---|
# | **Natural for** | Vorticity–streamfunction formulation | Pressure-like formulations |
# | **Velocity recovery** | Direct difference X→U, X→V | T→U, T→V (same as pressure gradient) |
# | **BC enforcement** | ψ=0 at grid vertices (on boundary) | ψ=0 half-spacing outside (off grid) |
# | **finitevolX API** | `streamfunction_from_vorticity(bc="dst")` | Direct `solve_poisson_dst2` |
# | **Typical use case** | QG models, barotropic vorticity | Projection methods, SSH diagnostics |
