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
# # Demo: Arakawa C-Grid Masks
#
# This notebook demonstrates how `ArakawaCGridMask` builds staggered masks
# from a cell-centre wet/dry field and how they relate to the Arakawa C-grid
# layout used in ocean models.
#
# ## Grid layout
#
# ```
# y
# ^
# :           :
# w-----v-----w..
# |           |
# |           |
# u     h     u
# |           |
# |           |
# w-----v-----w..   > x
# ```
#
# - **h** — cell centre (tracers, height, pressure)
# - **u** — y-face (zonal velocity)
# - **v** — x-face (meridional velocity)
# - **w / psi** — SW corner (vorticity / streamfunction)

# %%
import jax
import matplotlib.pyplot as plt
import numpy as np

from finitevolx import ArakawaCGrid2D, ArakawaCGridMask

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 1. Creating masks from different domain topologies
#
# `ArakawaCGridMask.from_mask` takes a binary h-grid mask and derives all
# staggered masks (u, v, w, psi), boundary classification, and stencil
# capability arrays automatically.

# %%
# Rectangular basin: all ocean interior, land on boundary
n = 10
h_rect = np.ones((n, n), dtype=bool)
h_rect[0, :] = False
h_rect[-1, :] = False
h_rect[:, 0] = False
h_rect[:, -1] = False

masks_rect = ArakawaCGridMask.from_mask(h_rect)
print(f"Rectangular basin: h shape = {masks_rect.h.shape}")
print(f"  Wet h-cells: {int(masks_rect.h.sum())}")
print(f"  Wet u-cells: {int(masks_rect.u.sum())}")
print(f"  Wet v-cells: {int(masks_rect.v.sum())}")
print(f"  Wet psi-cells: {int(masks_rect.psi.sum())}")

# %%
# Island: rectangular basin with a hole in the middle
h_island = np.ones((n, n), dtype=bool)
h_island[0, :] = False
h_island[-1, :] = False
h_island[:, 0] = False
h_island[:, -1] = False
h_island[4:7, 4:7] = False  # island

masks_island = ArakawaCGridMask.from_mask(h_island)
print(f"Island domain: Wet h-cells = {int(masks_island.h.sum())}")

# %%
# Channel: periodic in x, walls in y
h_channel = np.ones((n, 3 * n // 4), dtype=bool)
h_channel[:, 0] = False
h_channel[:, -1] = False

masks_channel = ArakawaCGridMask.from_mask(h_channel)
print(
    f"Channel domain: shape = {masks_channel.h.shape}, "
    f"Wet h-cells = {int(masks_channel.h.sum())}"
)

# %%
# Irregular coastline
h_irreg = np.ones((n, n), dtype=bool)
h_irreg[1, 0] = False
h_irreg[n - 1, 2] = False
h_irreg[0, n - 2] = False
h_irreg[1, n - 2] = False
h_irreg[0, n - 1] = False
h_irreg[1, n - 1] = False
h_irreg[2, n - 1] = False

masks_irreg = ArakawaCGridMask.from_mask(h_irreg)
print(f"Irregular domain: Wet h-cells = {int(masks_irreg.h.sum())}")
print(f"  Irregular boundary psi indices: {len(masks_irreg.psi_irrbound_xids)} cells")

# %% [markdown]
# ## 2. Visualising the staggered masks
#
# The `ArakawaCGridMask` stores five staggered masks plus a 4-level
# land/coast classification (0 = land, 1 = coast, 2 = near-coast, 3 = ocean).

# %%
from finitevolx._src.grid.cgrid_mask import visualize_masks

visualize_masks(masks_island)

# %% [markdown]
# ## 3. Stencil capability and adaptive WENO masks
#
# Each cell stores how many contiguous wet neighbours it has in each direction.
# This drives the adaptive stencil selection for WENO reconstruction near
# coastlines.

# %%
sc = masks_island.stencil_capability
print("Stencil capability (x_pos) — contiguous wet cells to the right:")
print(np.asarray(sc.x_pos))

# %%
# Get mutually-exclusive adaptive masks for x-direction reconstruction
adaptive = masks_island.get_adaptive_masks(direction="x", source="h")
print("Adaptive stencil masks (x-direction):")
for size, mask in sorted(adaptive.items()):
    count = int(mask.sum())
    if count > 0:
        print(f"  stencil size {size:2d}: {count} cells")

# %% [markdown]
# ## 4. Vorticity boundary classification
#
# At vorticity (w) points, the mask classifies cells into:
# - **w_valid**: interior — all 4 adjacent velocity faces are wet
# - **w_vertical_bound**: on a vertical (y-direction) boundary
# - **w_horizontal_bound**: on a horizontal (x-direction) boundary
# - **w_cornerout_bound**: at convex corners (both boundary types)

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ["w_valid", "w_vertical_bound", "w_horizontal_bound", "w_cornerout_bound"]
fields = [
    masks_island.w_valid,
    masks_island.w_vertical_bound,
    masks_island.w_horizontal_bound,
    masks_island.w_cornerout_bound,
]
for ax, title, field in zip(axes, titles, fields, strict=True):
    ax.imshow(np.asarray(field), origin="lower", cmap="Blues", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle("Vorticity boundary classification (island domain)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. All-ocean domain shortcut
#
# For simple rectangular domains without land, use `from_dimensions`:

# %%
masks_simple = ArakawaCGridMask.from_dimensions(ny=12, nx=12)
print(
    f"All-ocean: shape = {masks_simple.h.shape}, all wet = {bool(masks_simple.h.all())}"
)

# %% [markdown]
# ## 6. Physical context: QG and Shallow Water equations
#
# ### Quasi-Geostrophic equations
#
# $$
# \begin{aligned}
# \partial_t \omega &= -\vec{\boldsymbol{u}} \cdot \nabla\omega \\
# \omega &= \nabla^2 \psi \\
# \vec{\boldsymbol{u}} &= \left[-\partial_y \psi,\; \partial_x \psi \right]^\top
# \end{aligned}
# $$
#
# On the C-grid: $\omega$ lives at **h-points** (cell centres), $\psi$ at
# **psi-points** (corners), and $u, v$ at **face points**.
#
# ### Shallow Water equations (vector-invariant form)
#
# $$
# \begin{aligned}
# \partial_t h &= -\nabla \cdot (\vec{\boldsymbol{u}} h) \\
# \partial_t u &= qhv - \partial_x p + F_x \\
# \partial_t v &= -qhu - \partial_y p + F_y
# \end{aligned}
# $$
#
# where $q = (\zeta + f) / h$ is the potential vorticity and
# $p = g(h + \eta_b) + \frac{1}{2}(u^2 + v^2)$ is the Bernoulli potential.
#
# On the C-grid: $h$ at **h-points**, $u$ at **u-points**, $v$ at **v-points**,
# and $q$ at **w/psi-points**.

# %%
# Demonstrate with a grid + masks for a small basin
grid = ArakawaCGrid2D.from_interior(8, 8, dx=1e4, dy=1e4)
masks = ArakawaCGridMask.from_dimensions(ny=grid.Ny, nx=grid.Nx)
print(f"Grid: {grid.Ny}x{grid.Nx} (8 interior + ghost ring)")
print(f"  dx = {grid.dx:.0f} m, dy = {grid.dy:.0f} m")
print(
    f"Masks: h={int(masks.h.sum())} wet, "
    f"u={int(masks.u.sum())} wet, "
    f"v={int(masks.v.sum())} wet"
)
