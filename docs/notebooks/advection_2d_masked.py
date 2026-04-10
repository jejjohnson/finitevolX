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
# # 2-D Advection on Masked Domains: Stencil Cascades in Practice
#
# The previous advection tutorials used **fully periodic** domains where
# every cell is ocean.  Real-world problems have coastlines, islands, and
# channels.  When a high-order stencil (say WENO5, 6 points wide) reaches
# across a land cell, it produces garbage.  finitevolX solves this with
# **adaptive stencil cascading**: each cell automatically falls back to the
# widest stencil that fits within contiguous ocean.
#
# ```
# WENO5 stencil (6 pts)         Near coast — only 4 pts fit
# ┌─┬─┬─┬─┬─┬─┐                ┌─┬─┬─┬─┐
# │ │ │ │ │ │ │  → full WENO5  │▓│ │ │ │  → fall back to WENO3
# └─┴─┴─┴─┴─┴─┘                └─┴─┴─┴─┘
#                                ▓ = land
#
# At a tight corner — only 2 pts
# ┌─┬─┐
# │▓│ │  → fall back to upwind1
# └─┴─┘
# ```
#
# This tutorial walks through the practical details:
#
# 1. Build a **weird-geometry domain** with islands and channels.
# 2. Inspect the **stencil capability** and **adaptive masks** to see
#    *exactly* which scheme runs at each cell.
# 3. Run a cosine-bell advection and compare masked WENO5 against
#    unmasked (naïve) WENO5 and upwind1.
# 4. Tips and tricks for getting masked advection to work smoothly.
#
# > **Prerequisites** — read the
# > [1-D Advection](advection_1d_schemes.py) and
# > [2-D Rotation](advection_2d_rotation.py) tutorials for the basics,
# > and the [Advection Theory](../advection.md#adaptive-stencil-selection-mask-aware)
# > page for the mathematical background.

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

IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "advection_2d_masked"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1: Build a Weird Domain
#
# We construct an "archipelago" domain on a 64x64 grid:
#
# - A rectangular basin (land border around the edges).
# - A **C-shaped island** in the centre-right, creating a narrow channel.
# - A small **circular island** in the lower-left quadrant.
# - A **diagonal peninsula** jutting in from the north-west wall.
#
# ```
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# ▓▓▓▓▓▓                                   ▓
# ▓ ▓▓▓                                    ▓
# ▓  ▓▓     peninsula                      ▓
# ▓   ▓                ┌──────┐             ▓
# ▓                    │      │             ▓
# ▓                    │  C-  │             ▓
# ▓       ○            │island│             ▓
# ▓    (circle)        │      └──┐          ▓
# ▓                    │         │          ▓
# ▓                    └─────────┘          ▓
# ▓                                         ▓
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# ```

# %%
nx = ny = 64
Lx = Ly = 1.0
grid = fvx.CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
dx, dy = grid.dx, grid.dy
Nx, Ny = grid.Nx, grid.Ny

# Start with all-ocean, then carve out land.
ocean = np.ones((Ny, Nx), dtype=bool)

# --- Basin walls (2-cell border = ghost ring + 1 land cell) ---
ocean[:2, :] = False  # south
ocean[-2:, :] = False  # north
ocean[:, :2] = False  # west
ocean[:, -2:] = False  # east

# --- C-shaped island (centre-right) ---
# Main rectangle
ocean[22:42, 34:44] = False
# Open the channel on the south side (remove part of the rectangle)
ocean[22:28, 40:44] = True  # channel gap

# --- Small circular island (lower-left) ---
jj, ii = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
cx_circle, cy_circle = 16, 18  # centre (col, row)
r_circle = 4
ocean[(jj - cy_circle) ** 2 + (ii - cx_circle) ** 2 < r_circle**2] = False

# --- Diagonal peninsula from north-west ---
for k in range(8):
    ocean[Ny - 3 - k, 2 + k : 4 + k] = False

print(f"Domain: {Nx}x{Ny}, ocean cells: {ocean.sum()}/{ocean.size}")

# %% [markdown]
# ### Visualise the domain
#
# Quick sanity check — land in dark, ocean in light.

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(ocean.astype(float), origin="lower", cmap="Blues", vmin=0, vmax=1.5)
ax.set_title("Domain: ocean (blue) / land (dark)")
ax.set_xlabel("i")
ax.set_ylabel("j")
fig.tight_layout()
fig.savefig(IMG_DIR / "domain.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Domain](../../images/advection_2d_masked/domain.png)

# %% [markdown]
# ## Step 2: Build the Mask and Inspect Stencil Capabilities
#
# `Mask2D.from_mask()` takes the boolean ocean array and
# automatically computes:
#
# - **Staggered masks** for U, V, and corner points.
# - **Stencil capability**: how many contiguous wet cells exist in each
#   direction from every point.
# - **Land/coast classification** (land → coast → near-coast → ocean).
#
# > **Tip:** Always build the mask from the **full** array (including ghost
# > cells), not just the interior.  The ghost cells should be land for a
# > closed basin, or you can mark them wet for periodic domains.

# %%
mask = fvx.Mask2D.from_mask(ocean)

# The stencil capability tells us how far we can "see" in each direction.
sc = mask.stencil_capability
print(f"Stencil capability shape: {sc.x_pos.shape}")
print(f"  x_pos range: {int(sc.x_pos.min())} -{int(sc.x_pos.max())}")
print(f"  y_pos range: {int(sc.y_pos.min())} -{int(sc.y_pos.max())}")

# %% [markdown]
# ### What does stencil capability look like?
#
# `x_pos[j, i]` counts how many consecutive wet cells lie to the **east**
# of cell (j, i), including itself.  A value of 1 means "I'm wet but my
# eastern neighbour is land."  A value of 6+ means WENO5 can use its full
# stencil here.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, arr, title in [
    (axes[0], np.asarray(sc.x_pos), "x_pos (contiguous wet → east)"),
    (axes[1], np.asarray(sc.y_pos), "y_pos (contiguous wet → north)"),
]:
    im = ax.imshow(arr, origin="lower", cmap="YlGnBu", vmin=0, vmax=10)
    ax.contour(ocean.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    fig.colorbar(im, ax=ax, shrink=0.75, label="# cells")

fig.tight_layout()
fig.savefig(IMG_DIR / "stencil_capability.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Stencil capability](../../images/advection_2d_masked/stencil_capability.png)
#
# Notice how the values **drop to 1-2** right next to coastlines.  WENO5
# needs at least 3 in *both* directions to use the full 6-point stencil.

# %% [markdown]
# ## Step 3: Adaptive Masks — Which Scheme Runs Where?
#
# `get_adaptive_masks()` builds **mutually-exclusive** boolean masks that
# assign exactly one reconstruction scheme to each wet cell:
#
# - **Size 6** → WENO5 (5th-order, needs 3 wet cells on each side)
# - **Size 4** → WENO3 (3rd-order, needs 2 wet cells on each side)
# - **Size 2** → Upwind1 (1st-order, just needs 1 neighbour)
#
# The largest stencil that fits wins.

# %%
# Build adaptive masks for x-direction
adaptive_x = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
adaptive_y = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))

for label, adaptive in [("x", adaptive_x), ("y", adaptive_y)]:
    for sz, msk in sorted(adaptive.items()):
        n = int(jnp.sum(msk))
        names = {2: "upwind1", 4: "WENO3", 6: "WENO5"}
        print(f"  {label}-dir  size {sz} ({names[sz]}): {n:5d} cells")


# %%
# Combine into a single array for plotting:
#   0 = land, 2 = upwind1, 4 = WENO3, 6 = WENO5
def combine_adaptive(adaptive):
    """Merge adaptive masks into one integer array for plotting."""
    out = np.zeros_like(np.asarray(adaptive[2]), dtype=int)
    for sz, msk in adaptive.items():
        out += sz * np.asarray(msk)
    return out


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
cmap = matplotlib.colors.ListedColormap(["#2d2d2d", "#d62728", "#ff7f0e", "#1f77b4"])
bounds = [-0.5, 1, 3, 5, 7]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

for ax, adaptive, title in [
    (axes[0], adaptive_x, "x-direction stencil assignment"),
    (axes[1], adaptive_y, "y-direction stencil assignment"),
]:
    combined = combine_adaptive(adaptive)
    im = ax.imshow(combined, origin="lower", cmap=cmap, norm=norm)
    ax.contour(ocean.astype(float), levels=[0.5], colors="w", linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")

cbar = fig.colorbar(im, ax=axes, shrink=0.75, ticks=[0, 2, 4, 6])
cbar.ax.set_yticklabels(["land", "upwind1", "WENO3", "WENO5"])
fig.savefig(IMG_DIR / "adaptive_stencils.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Adaptive stencils](../../images/advection_2d_masked/adaptive_stencils.png)
#
# Key things to notice:
#
# - The **bulk of the ocean** (blue) uses WENO5 — full 5th-order accuracy.
# - A thin **orange fringe** of WENO3 cells surrounds every coast.
# - The **narrowest channels** and tight corners fall back to upwind1 (red).
# - The x- and y-direction maps differ — a cell may use WENO5 in x but
#   WENO3 in y if the coastline is oriented differently.

# %% [markdown]
# ## Step 4: Advection on the Masked Domain
#
# We set up a cosine-bell tracer in the open-ocean region and advect it
# with a prescribed velocity field that steers it around the islands.
#
# ### Velocity field
#
# We use a uniform diagonal flow $u = v = 1$ — simple enough to understand,
# but the islands force the flow to interact with the stencil cascade.
#
# > **Tip:** Set velocity to zero at land faces.  The mask handles the
# > reconstruction, but zeroing land velocities avoids fluxes through walls.

# %%
# Coordinates for T-points (interior: [1:-1, 1:-1])
x_t = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
y_t = jnp.linspace(0.5 * dy, Ly - 0.5 * dy, ny)
X_T, Y_T = jnp.meshgrid(x_t, y_t)

# Uniform diagonal flow, zeroed on land
ocean_jnp = jnp.array(ocean, dtype=jnp.float64)
u_field = 1.0 * ocean_jnp  # u at U-points
v_field = 1.0 * ocean_jnp  # v at V-points

# Cosine bell in the lower-left ocean area
xc_bell, yc_bell = 0.2, 0.25
R_bell = 0.1
r_full = jnp.zeros((Ny, Nx))
r_int = jnp.sqrt((X_T - xc_bell) ** 2 + (Y_T - yc_bell) ** 2)
r_full = r_full.at[1:-1, 1:-1].set(r_int)
q0 = jnp.where(
    (r_full < R_bell) & (ocean_jnp > 0.5),
    0.5 * (1.0 + jnp.cos(jnp.pi * r_full / R_bell)),
    0.0,
)

# %% [markdown]
# ### Run three experiments
#
# 1. **Masked WENO5** — the correct approach.  Pass `mask=mask` so the
#    operator automatically cascades near coasts.
# 2. **Unmasked WENO5** — what happens if you *forget* the mask.  The
#    stencil reads land cells (zeros) as real data.
# 3. **Upwind1** — the safe but diffusive baseline.

# %%
advect = fvx.Advection2D(grid)

cfl = 0.4
u_max = float(jnp.max(jnp.abs(u_field)))
v_max = float(jnp.max(jnp.abs(v_field)))
dt = cfl / (max(u_max, 1e-10) / dx + max(v_max, 1e-10) / dy)
T_final = 0.35  # long enough for the bell to hit the C-island
nsteps = int(jnp.ceil(T_final / dt))
dt = T_final / nsteps

print(f"dt = {dt:.4e}, nsteps = {nsteps}, T = {T_final}")

experiments = {
    "WENO5 + mask": {"method": "weno5", "mask": mask},
    "WENO5 (no mask)": {"method": "weno5", "mask": None},
    "upwind1": {"method": "upwind1", "mask": None},
}

results = {}
for name, cfg in experiments.items():

    def make_rhs(method, msk):
        def rhs(q):
            # Zero out land cells each stage — keeps the solution clean.
            q = q * ocean_jnp
            return advect(q, u_field, v_field, method=method, mask=msk)

        return rhs

    rhs_fn = jax.jit(make_rhs(cfg["method"], cfg["mask"]))
    q = q0.copy()

    for _step in range(nsteps):
        q = fvx.rk3_ssp_step(q, rhs_fn, dt)
        q = q * ocean_jnp  # enforce land = 0

    peak = float(jnp.max(q))
    results[name] = np.asarray(q)
    print(f"  {name:<20s}  peak = {peak:.4f}")

# %% [markdown]
# ## Results
#
# ### Side-by-side comparison

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)

for ax, (name, data) in zip(axes, results.items(), strict=False):
    im = ax.imshow(
        data,
        origin="lower",
        cmap="RdYlBu_r",
        vmin=-0.05,
        vmax=1.0,
    )
    ax.contour(ocean.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    peak = float(np.max(data))
    ax.set_title(f"{name}\npeak = {peak:.3f}")
    ax.set_xlabel("i")

axes[0].set_ylabel("j")
fig.colorbar(im, ax=axes, shrink=0.85, label="q")
fig.savefig(IMG_DIR / "comparison.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Comparison](../../images/advection_2d_masked/comparison.png)

# %% [markdown]
# ### What to look for
#
# - **WENO5 + mask** (left): The bell is crisp.  As it approaches the
#   C-island, the stencil silently cascades: WENO5 in the open, WENO3
#   in the fringe, upwind1 right at the wall.  No artefacts.
# - **WENO5 no mask** (centre): The stencil reads **land zeros** as real
#   data.  This creates spurious gradients at every coastline — you'll see
#   ringing and possibly negative values near the islands.
# - **Upwind1** (right): Stable everywhere, but the bell is smeared by
#   numerical diffusion.

# %% [markdown]
# ## Practical Tips and Tricks
#
# ### Tip 1: Zero land cells every RK stage
#
# The RK intermediate states can leak small values into land cells
# (because the tendency is only computed on ocean cells, but the linear
# combination in RK touches everything).  Multiply by the ocean mask
# after each stage:
#
# ```python
# def rhs(q):
#     q = q * ocean_jnp       # ← clean up land before reconstruction
#     return advect(q, u, v, method='weno5', mask=mask)
#
# for _ in range(nsteps):
#     q = fvx.rk3_ssp_step(q, rhs, dt)
#     q = q * ocean_jnp       # ← clean up land after the full step
# ```
#
# ### Tip 2: Zero velocity at land faces
#
# `Advection2D` multiplies the reconstructed face value by the face
# velocity.  If the velocity at a land face is nonzero, you get a flux
# through the wall.  Safest:
#
# ```python
# u_field = velocity_x * ocean_jnp   # zero on dry cells
# v_field = velocity_y * ocean_jnp
# ```
#
# ### Tip 3: Choose the right stencil_sizes
#
# By default `get_adaptive_masks` checks sizes `(2, 4, 6, 8, 10)`.  If
# you only use WENO5 (size 6), pass `stencil_sizes=(2, 4, 6)` to skip
# unnecessary work:
#
# ```python
# mask.get_adaptive_masks(direction='x', stencil_sizes=(2, 4, 6))
# ```
#
# When you call `Advection2D(q, u, v, method='weno5', mask=mask)`, this
# is handled automatically — the operator selects `(2, 4, 6)` for WENO5.
#
# ### Tip 4: Inspect stencil assignment before running
#
# Always visualise the adaptive masks (as we did above) before running a
# long simulation.  Look for:
#
# - **Channels narrower than 6 cells** — these will fall back from WENO5
#   to WENO3 or even upwind1.  If accuracy matters there, refine the grid.
# - **Isolated single-cell bays** — these may have no valid stencil at
#   all.  Consider filling them in.
# - **Asymmetric x vs y maps** — expected for elongated features, but
#   worth sanity-checking.
#
# ### Tip 5: Use the classification for diagnostics
#
# The mask provides a 4-level classification that's handy for computing
# coast-only diagnostics:
#
# ```python
# coast_cells = mask.ind_coast       # boolean, True at coastline
# ocean_cells = mask.ind_ocean       # True in the interior
# near_coast  = mask.ind_near_coast  # one cell away from coast
# ```

# %% [markdown]
# ## Bonus: Zoom on the Cascade in Action
#
# Let's zoom in on the C-island channel to see the cascade close up.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

# Zoom region around the C-island channel
r_slice = slice(18, 46)
c_slice = slice(30, 52)

# Panel 1: stencil assignment (x-direction)
combined_x = combine_adaptive(adaptive_x)
ax = axes[0]
ax.imshow(combined_x[r_slice, c_slice], origin="lower", cmap=cmap, norm=norm)
ax.set_title("Stencil (x-dir)")

# Panel 2: stencil assignment (y-direction)
combined_y = combine_adaptive(adaptive_y)
ax = axes[1]
ax.imshow(combined_y[r_slice, c_slice], origin="lower", cmap=cmap, norm=norm)
ax.set_title("Stencil (y-dir)")

# Panel 3: masked WENO5 solution
ax = axes[2]
im = ax.imshow(
    results["WENO5 + mask"][r_slice, c_slice],
    origin="lower",
    cmap="RdYlBu_r",
    vmin=-0.05,
    vmax=1.0,
)
ax.contour(
    ocean[r_slice, c_slice].astype(float), levels=[0.5], colors="k", linewidths=0.8
)
ax.set_title("WENO5 + mask (solution)")
fig.colorbar(im, ax=axes[2], shrink=0.85)

for ax in axes:
    ax.set_xlabel("i (zoomed)")
axes[0].set_ylabel("j (zoomed)")
fig.suptitle("Zoom: C-Island Channel", fontsize=13)
fig.savefig(IMG_DIR / "channel_zoom.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ![Channel zoom](../../images/advection_2d_masked/channel_zoom.png)
#
# In the narrow channel opening at the bottom of the C-island, the
# stencil transitions smoothly:
#
# ```
# open ocean → WENO5 → WENO3 → upwind1 → coast
# ```
#
# The tracer passes through cleanly without ringing or artificial
# diffusion beyond what the local scheme order dictates.

# %% [markdown]
# ## Next Steps
#
# - Combine masked advection with a full dynamical model — see the
#   [Shallow Water](shallow_water.py) tutorial.
# - Explore the mask infrastructure in detail — see the
#   [Masks Demo](demo_masks.py) notebook.
# - For the mathematical background on the cascade, see
#   [Advection Theory: Adaptive Stencil Selection](../advection.md#adaptive-stencil-selection-mask-aware).
