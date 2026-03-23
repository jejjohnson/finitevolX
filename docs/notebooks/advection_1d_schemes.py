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
# # 1-D Advection: Comparing Reconstruction Schemes
#
# This tutorial demonstrates how different **reconstruction schemes** handle
# the advection of a square wave on a periodic 1-D domain.  It is a classic
# computational-fluid-dynamics benchmark that reveals the trade-offs between
# *numerical diffusion* (smearing) and *dispersive oscillations* (ringing).
#
# By the end you will understand:
#
# 1. How the **flux-form advection equation** is discretised on the Arakawa
#    C-grid.
# 2. How **upwind**, **TVD**, and **WENO** reconstruction schemes differ in
#    practice.
# 3. How to use `finitevolx.Advection1D` with different `method` options.
#
# > **Prerequisites** — read the
# > [Advection Theory](../advection.md) page for the mathematical
# > background on flux-form advection and reconstruction schemes.

# %% [markdown]
# ## The Governing Equation
#
# We solve the 1-D **linear advection equation** in flux (conservative) form:
#
# $$
# \frac{\partial q}{\partial t} + \frac{\partial (u\,q)}{\partial x} = 0
# $$
#
# where $q(x,t)$ is a scalar field (e.g. a tracer concentration) and $u$ is a
# constant advection velocity.  With periodic boundary conditions on
# $[0, L]$ and constant $u > 0$, the exact solution is simply the initial
# condition translated to the right:
#
# $$
# q(x, t) = q_0\!\bigl((x - u\,t) \bmod L\bigr)
# $$
#
# After one full period $T = L / u$, the solution returns exactly to its
# initial shape.  Any deviation from the initial condition is **purely
# numerical error**, making this an ideal benchmark.
#
# ### Discrete form on the C-grid
#
# On the Arakawa C-grid, the scalar $q$ lives at **T-points** (cell centres)
# and the velocity $u$ lives at **U-points** (cell faces).  The
# semi-discrete tendency at T-point $i$ is
#
# $$
# \left.\frac{\partial q}{\partial t}\right|_i
# = -\frac{f_{i+\frac{1}{2}} - f_{i-\frac{1}{2}}}{\Delta x}
# $$
#
# where $f_{i+1/2} = u_{i+1/2}\,\hat{q}_{i+1/2}$ is the **numerical flux**
# and $\hat{q}_{i+1/2}$ is the **reconstructed face value**.  The choice of
# reconstruction scheme determines the accuracy and character of the solution.

# %% [markdown]
# ## The Four Schemes We Compare
#
# We test six schemes spanning the library's range:
#
# | Scheme | Type | Order | Character |
# |--------|------|-------|-----------|
# | `upwind1` | Upwind | 1 | Maximum diffusion — smears everything |
# | `upwind3` | Upwind | 3 | Less diffusion, mild Gibbs-like oscillations |
# | `superbee` | TVD limiter | 2 | Sharpest monotone scheme — compresses fronts |
# | `weno5` | WENO | 5 | High-order smooth + non-oscillatory near shocks |
# | `weno7` | WENO | 7 | Higher-order, sharper resolution of fine features |
# | `weno9` | WENO | 9 | Highest-order, best for smooth flows |
#
# **Upwind 1st-order** reconstructs the face value as the upwind cell value:
#
# $$
# \hat{q}_{i+1/2} = \begin{cases} q_i & u_{i+1/2} \geq 0 \\ q_{i+1} & u_{i+1/2} < 0 \end{cases}
# $$
#
# This is maximally diffusive but unconditionally monotone.
#
# **Upwind 3rd-order** uses a 4-cell stencil to build a cubic reconstruction,
# trading some monotonicity for much less diffusion.
#
# **TVD/Superbee** adds a flux limiter $\phi(r)$ to a 1st-order base:
#
# $$
# \hat{q}_{i+1/2} = q_i + \tfrac{1}{2}\,\phi(r)\,(q_i - q_{i-1}),
# \qquad r = \frac{q_{i+1} - q_i}{q_i - q_{i-1}}
# $$
#
# Superbee: $\phi(r) = \max\!\bigl(0,\;\min(2r, 1),\;\min(r, 2)\bigr)$.
# It is the **sharpest** TVD limiter — fronts stay steep, but smooth
# profiles can develop staircase artefacts.
#
# **WENO5** blends three 3-point candidate stencils with nonlinear weights
# that automatically reduce to low order near discontinuities while
# maintaining 5th-order accuracy in smooth regions.  See the
# [Advection Theory](../advection.md#weighted-essentially-non-oscillatory-weno)
# page for the full formulation.

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

IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "advection_1d_schemes"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Grid Setup
#
# We construct a 1-D Arakawa C-grid on the periodic domain $[0, 1]$.
#
# **Ghost cells** — The `Advection1D` operator writes its tendency to the
# interior region `[2:-2]`, meaning the outermost 2 cells on each side
# receive zero tendency.  For periodic advection we need the stencil to
# "see through" the boundary, so we pad the domain with extra ghost cells
# and fill them with periodic copies each timestep.
#
# ```
# Ghost zone          Physical domain           Ghost zone
# ┌─┬─┬─┬─┐ ┌─┬─┬─┬─┬─ ··· ─┬─┬─┬─┬─┐ ┌─┬─┬─┬─┐
# │g│g│g│g│ │0│1│2│3│4  ···  │ │ │ │nx│ │g│g│g│g│
# └─┴─┴─┴─┘ └─┴─┴─┴─┴─ ··· ─┴─┴─┴─┴─┘ └─┴─┴─┴─┘
#   ← ng →                                 ← ng →
# ```
#
# We use `ng = 5` ghost cells per side, which is enough for all schemes
# up to WENO9.

# %%
# --- physical parameters ---
nx = 256  # physical cells
Lx = 1.0  # domain length
c = 1.0  # advection velocity

# --- ghost ring ---
ng = 5  # ghost cells per side (≥ 5 for WENO9)

# --- grid ---
dx = Lx / nx
grid = fvx.ArakawaCGrid1D(Nx=nx + 2 * ng, Lx=Lx, dx=dx)

# cell-centre coordinates for the physical domain
x_phys = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)

print(f"Grid: {nx} physical cells, dx = {dx:.4e}, Nx (with ghosts) = {grid.Nx}")

# %% [markdown]
# ## Periodic Boundary Conditions
#
# Before each call to `Advection1D` we must fill the ghost cells with
# periodic copies from the opposite end of the physical domain.
#
# ```
# Before:  [g g g g | q0 q1 q2 ... q_{n-1} | g g g g]
#           ↑               copy from →          ↑
# After:   [q_{n-4} ... q_{n-1} | q0 q1 ... q_{n-1} | q0 q1 q2 q3]
# ```

# %%
def enforce_periodic_1d(h: jax.Array, ng: int) -> jax.Array:
    """Fill *ng* ghost cells on each side with periodic copies."""
    # left ghosts ← rightmost physical cells
    h = h.at[:ng].set(h[-2 * ng : -ng])
    # right ghosts ← leftmost physical cells
    h = h.at[-ng:].set(h[ng : 2 * ng])
    return h

# %% [markdown]
# ## Initial Condition: Square Wave
#
# A square wave (top-hat) is the classic test because it has two
# **discontinuities** — a rising edge and a falling edge.  The exact
# solution never changes shape, so any smearing or ringing at time $T = L/u$
# is purely numerical.
#
# $$
# q_0(x) =
# \begin{cases}
#   1 & 0.25 \leq x \leq 0.75 \\
#   0 & \text{otherwise}
# \end{cases}
# $$

# %%
def square_wave(x, x_left=0.25, x_right=0.75):
    """Square wave (top-hat) initial condition."""
    return jnp.where((x >= x_left) & (x <= x_right), 1.0, 0.0)


# Build the full array (physical + ghost cells)
q0_phys = square_wave(x_phys)
q0 = jnp.zeros(grid.Nx)
q0 = q0.at[ng:-ng].set(q0_phys)
q0 = enforce_periodic_1d(q0, ng)

# Constant velocity at all points (including ghosts)
u_field = c * jnp.ones(grid.Nx)

# %% [markdown]
# Let's visualise the initial condition:

# %%
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(x_phys, q0_phys, "k-", lw=2, label="Initial condition")
ax.set_xlabel("x")
ax.set_ylabel("q")
ax.set_title("Square Wave Initial Condition")
ax.set_xlim(0, 1)
ax.set_ylim(-0.15, 1.25)
ax.legend()
fig.tight_layout()
fig.savefig(IMG_DIR / "initial_condition.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Initial condition](../../images/advection_1d_schemes/initial_condition.png)

# %% [markdown]
# ## Time Integration
#
# We integrate using the **SSP-RK3** (Strong Stability Preserving
# Runge–Kutta, 3rd-order) time stepper.  SSP schemes preserve the
# monotonicity and TVD properties of the spatial discretisation, making
# them the natural partner for TVD and WENO advection.
#
# The SSP-RK3 update (Shu–Osher form) is:
#
# $$
# q^{(1)} = q^n + \Delta t \, \mathcal{L}(q^n)
# $$
# $$
# q^{(2)} = \tfrac{3}{4}\,q^n + \tfrac{1}{4}\,q^{(1)} + \tfrac{1}{4}\,\Delta t \, \mathcal{L}(q^{(1)})
# $$
# $$
# q^{n+1} = \tfrac{1}{3}\,q^n + \tfrac{2}{3}\,q^{(2)} + \tfrac{2}{3}\,\Delta t \, \mathcal{L}(q^{(2)})
# $$
#
# where $\mathcal{L}$ is the advection tendency operator.
#
# ### CFL condition
#
# The timestep is set via the **CFL number** $\sigma = u\,\Delta t / \Delta x$.
# For SSP-RK3 with upwind-type schemes, $\sigma \leq 1$ is stable.
# We use $\sigma = 0.5$ for safety.

# %%
cfl = 0.5
dt = cfl * dx / abs(c)
T_final = Lx / abs(c)  # one full period
nsteps = int(T_final / dt)
dt = T_final / nsteps  # adjust so we land exactly at T

print(f"CFL = {cfl}, dt = {dt:.4e}, nsteps = {nsteps}")

# %% [markdown]
# ## Running the Four Schemes
#
# For each scheme we:
# 1. Create an `Advection1D` operator.
# 2. Define a right-hand-side function that applies periodic BCs then
#    computes the tendency.
# 3. Advance `nsteps` timesteps with `rk3_ssp_step`.
# 4. Extract the physical cells and compare to the exact solution.

# %%
advect = fvx.Advection1D(grid)
methods = ["upwind1", "upwind3", "superbee", "weno5", "weno7", "weno9"]
results = {}

for method in methods:

    def make_rhs(m):
        """Closure to capture the method string."""

        def rhs(q):
            q = enforce_periodic_1d(q, ng)
            return advect(q, u_field, method=m)

        return rhs

    rhs_fn = jax.jit(make_rhs(method))
    q = q0.copy()

    for _ in range(nsteps):
        q = fvx.rk3_ssp_step(q, rhs_fn, dt)
        q = enforce_periodic_1d(q, ng)

    results[method] = np.asarray(q[ng:-ng])

print("All schemes completed.")

# %% [markdown]
# ## Results: Full Domain Comparison
#
# The plot below overlays all four solutions after one full period
# against the exact (initial) profile.

# %%
exact = np.asarray(q0_phys)
x_np = np.asarray(x_phys)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_np, exact, "k--", lw=2, label="Exact", zorder=5)

colours = {
    "upwind1": "#d62728",
    "upwind3": "#1f77b4",
    "superbee": "#2ca02c",
    "weno5": "#ff7f0e",
    "weno7": "#9467bd",
    "weno9": "#17becf",
}

for method in methods:
    ax.plot(x_np, results[method], color=colours[method], lw=1.5, label=method)

ax.set_xlabel("x")
ax.set_ylabel("q")
ax.set_title("Advection of a Square Wave — One Full Period")
ax.set_xlim(0, 1)
ax.set_ylim(-0.15, 1.25)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(IMG_DIR / "scheme_comparison.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Scheme comparison](../../images/advection_1d_schemes/scheme_comparison.png)

# %% [markdown]
# ## Zoom: Rising and Falling Edges
#
# Zooming in on the discontinuities reveals the character of each scheme:
#
# - **upwind1** smears the edges over many cells (numerical diffusion).
# - **upwind3** has less diffusion but shows overshoots and undershoots
#   (dispersive oscillations — Gibbs-like ringing).
# - **superbee** keeps the fronts steep and never overshoots (TVD
#   property), but can flatten the top slightly.
# - **weno5** achieves a crisp edge with only tiny oscillations, combining
#   the best of both worlds.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

zoom_regions = [(0.15, 0.40, "Rising edge"), (0.60, 0.85, "Falling edge")]

for ax, (xl, xr, title) in zip(axes, zoom_regions):
    ax.plot(x_np, exact, "k--", lw=2, label="Exact", zorder=5)
    for method in methods:
        ax.plot(x_np, results[method], colours[method], lw=1.5, label=method)
    ax.set_xlim(xl, xr)
    ax.set_ylim(-0.15, 1.25)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend(fontsize=9)

axes[0].set_ylabel("q")
fig.suptitle("Zoom on Discontinuities", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(IMG_DIR / "edge_zoom.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Edge zoom](../../images/advection_1d_schemes/edge_zoom.png)

# %% [markdown]
# ## Error Norms
#
# We quantify the error with three standard norms:
#
# $$
# L_1 = \frac{1}{N}\sum_i |q_i - q_i^{\text{exact}}|, \qquad
# L_2 = \sqrt{\frac{1}{N}\sum_i (q_i - q_i^{\text{exact}})^2}, \qquad
# L_\infty = \max_i |q_i - q_i^{\text{exact}}|
# $$

# %%
print(f"{'Method':<12} {'L1':>10} {'L2':>10} {'Linf':>10}")
print("-" * 44)
for method in methods:
    err = results[method] - exact
    l1 = np.mean(np.abs(err))
    l2 = np.sqrt(np.mean(err**2))
    linf = np.max(np.abs(err))
    print(f"{method:<12} {l1:10.4e} {l2:10.4e} {linf:10.4e}")

# %% [markdown]
# ## Convergence Study
#
# To verify that each scheme achieves its theoretical order of accuracy,
# we repeat the experiment at multiple resolutions and plot the $L_2$ error
# against $\Delta x$ on a log-log scale.
#
# For a smooth initial condition, the expected slopes are:
#
# | Scheme | Expected order |
# |--------|---------------|
# | `upwind1` | 1 |
# | `upwind3` | 3 |
# | `superbee` | 2 |
# | `weno5` | 5 |
# | `weno7` | 7 (capped at ~3 by RK3 time integrator) |
# | `weno9` | 9 (capped at ~3 by RK3 time integrator) |
#
# > **Note:** for a discontinuous profile (our square wave), all schemes
# > degrade to 1st-order at the discontinuities.  We therefore also run
# > this study with a smooth **Gaussian** profile to see the clean
# > convergence rates.

# %%
def gaussian(x, x0=0.5, sigma=0.1):
    """Smooth Gaussian initial condition for convergence testing."""
    return jnp.exp(-((x - x0) ** 2) / (2 * sigma**2))


resolutions = [32, 64, 128, 256, 512]
convergence = {m: [] for m in methods}

for nx_test in resolutions:
    dx_test = Lx / nx_test
    dt_test = cfl * dx_test / abs(c)
    nsteps_test = int(T_final / dt_test)
    dt_test = T_final / nsteps_test

    grid_test = fvx.ArakawaCGrid1D(Nx=nx_test + 2 * ng, Lx=Lx, dx=dx_test)
    advect_test = fvx.Advection1D(grid_test)
    x_test = jnp.linspace(0.5 * dx_test, Lx - 0.5 * dx_test, nx_test)
    u_test = c * jnp.ones(grid_test.Nx)

    q0_test = jnp.zeros(grid_test.Nx)
    q0_test = q0_test.at[ng:-ng].set(gaussian(x_test))
    q0_test = enforce_periodic_1d(q0_test, ng)
    exact_test = np.asarray(gaussian(x_test))

    for method in methods:

        def make_rhs_conv(m, adv, u):
            def rhs(q):
                q = enforce_periodic_1d(q, ng)
                return adv(q, u, method=m)

            return rhs

        rhs_fn = jax.jit(make_rhs_conv(method, advect_test, u_test))
        q = q0_test.copy()
        for _ in range(nsteps_test):
            q = fvx.rk3_ssp_step(q, rhs_fn, dt_test)
            q = enforce_periodic_1d(q, ng)

        err = np.asarray(q[ng:-ng]) - exact_test
        l2 = float(np.sqrt(np.mean(err**2)))
        convergence[method].append(l2)

    print(f"  nx = {nx_test:>4d} done")

# %%
dx_arr = np.array([Lx / n for n in resolutions])

fig, ax = plt.subplots(figsize=(7, 5))
for method in methods:
    ax.loglog(dx_arr, convergence[method], "o-", color=colours[method], lw=1.5, label=method)

# Reference slopes
for order, ls in [(1, ":"), (2, "-."), (3, "--"), (5, "-"), (7, (0, (3, 1, 1, 1)))]:
    ref = convergence["upwind1"][0] * (dx_arr / dx_arr[0]) ** order
    ax.loglog(dx_arr, ref, color="grey", ls=ls, lw=0.8, alpha=0.5, label=f"O(Δx$^{order}$)")

ax.set_xlabel("Δx")
ax.set_ylabel("$L_2$ error")
ax.set_title("Convergence: Gaussian Advection (1 period)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(IMG_DIR / "convergence.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Convergence](../../images/advection_1d_schemes/convergence.png)

# %% [markdown]
# ## Summary
#
# | Scheme | Pros | Cons |
# |--------|------|------|
# | **upwind1** | Simple, guaranteed monotone | Extremely diffusive |
# | **upwind3** | Much less diffusion than upwind1 | Oscillations near discontinuities |
# | **superbee** | Sharpest monotone (TVD) scheme | Can create staircase artefacts on smooth profiles |
# | **weno5** | High-order smooth, low oscillation | Slightly more expensive per cell |
# | **weno7** | Sharper features than weno5 | Wider stencil (8 pts), higher cost |
# | **weno9** | Highest accuracy on smooth flows | Widest stencil (10 pts), highest cost |
#
# **Recommendation:** Use **`weno5`** as the default.  Switch to **`weno7`**
# or **`weno9`** when resolving fine-scale structure in smooth flows.  Use a
# TVD limiter (`superbee` or `van_leer`) when strict monotonicity is required
# (e.g., positive-definite tracers).  Use `upwind1` only as a fallback near
# boundaries.
#
# See the [Advection Theory](../advection.md) page for the full
# mathematical background and the
# [decision guide](../advection.md#decision-guide) for choosing a scheme.
