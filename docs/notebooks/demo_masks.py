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
# # Demo: Arakawa C-Grid Masks
#
# This notebook demonstrates how `Mask2D` builds staggered masks
# from a cell-centre wet/dry field and how they relate to the Arakawa C-grid
# layout used in ocean models.
#
# ## C-Grid Variable Placement
#
# ```
#     X ── V ── X ── V ── X
#     |         |         |
#     U    T    U    T    U     T = tracer / height / pressure (cell centre)
#     |         |         |     U = u-velocity (east face, perpendicular to x)
#     X ── V ── X ── V ── X     V = v-velocity (north face, perpendicular to y)
#     |         |         |     X = vorticity / streamfunction (NE corner)
#     U    T    U    T    U
#     |         |         |
#     X ── V ── X ── V ── X
# ```
#
# The **h-mask** (cell centres) is the primary input.  `Mask2D`
# derives all staggered masks, a 4-level land/coast classification,
# directional stencil capability arrays, and vorticity boundary flags
# automatically.  All staggered masks use the **same-index positive
# half-step** convention (matching the grid module): ``mask.u[j, i]``
# is at ``(j, i + 1/2)``, ``mask.v[j, i]`` at ``(j + 1/2, i)``,
# ``mask.xy_corner[j, i]`` at ``(j + 1/2, i + 1/2)``.
#
# ### Mask derivation rules
#
# | Variable | Location | Rule |
# |----------|----------|------|
# | **h** | cell centre | input mask (1 = wet, 0 = dry) |
# | **u** | east face | wet iff `h[j, i]` and `h[j, i+1]` both wet |
# | **v** | north face | wet iff `h[j, i]` and `h[j+1, i]` both wet |
# | **xy_corner** | NE corner (lenient) | wet iff *any* of the 4 NE-corner h-cells wet |
# | **xy_corner_strict** | NE corner (strict) | wet iff *all four* NE-corner h-cells wet |

# %%
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from finitevolx import CartesianGrid2D, Mask2D

IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "demo_masks"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Creating masks from different domain topologies
#
# `Mask2D.from_mask` takes a binary h-grid mask and derives all
# staggered masks (u, v, xy_corner, xy_corner_strict), boundary classification, and stencil
# capability arrays automatically.
#
# We build four common topologies:
#
# | Topology | Description |
# |----------|-------------|
# | **Basin** | Rectangular ocean interior with land on all boundaries |
# | **Island** | Basin with a 3x3 island in the centre |
# | **Channel** | Periodic in x, solid walls in y |
# | **Irregular** | Arbitrary coastline with convex/concave features |

# %%
# --- Basin: rectangular ocean interior, land on all boundaries ---
n = 10
h_basin = np.ones((n, n), dtype=bool)
h_basin[0, :] = False  # south wall
h_basin[-1, :] = False  # north wall
h_basin[:, 0] = False  # west wall
h_basin[:, -1] = False  # east wall

masks_basin = Mask2D.from_mask(h_basin)
print(f"Basin: h shape = {masks_basin.h.shape}")
print(f"  Wet h-cells: {int(masks_basin.h.sum())}")
print(f"  Wet u-cells: {int(masks_basin.u.sum())}")
print(f"  Wet v-cells: {int(masks_basin.v.sum())}")
print(f"  Wet xy_corner_strict cells: {int(masks_basin.xy_corner_strict.sum())}")

# %%
# --- Island: basin with a 3x3 island in the centre ---
h_island = np.ones((n, n), dtype=bool)
h_island[0, :] = False
h_island[-1, :] = False
h_island[:, 0] = False
h_island[:, -1] = False
h_island[4:7, 4:7] = False  # island

masks_island = Mask2D.from_mask(h_island)
print(
    f"Island: h shape = {masks_island.h.shape}, Wet h-cells = {int(masks_island.h.sum())}"
)

# %%
# --- Channel: periodic in x, solid walls in y ---
# Width is 3/4 of height to mimic a zonal channel (e.g. Drake Passage).
h_channel = np.ones((n, 3 * n // 4), dtype=bool)
h_channel[:, 0] = False  # south wall
h_channel[:, -1] = False  # north wall

masks_channel = Mask2D.from_mask(h_channel)
print(
    f"Channel: h shape = {masks_channel.h.shape}, "
    f"Wet h-cells = {int(masks_channel.h.sum())}"
)

# %%
# --- Irregular coastline ---
# Scattered land cells create convex/concave boundary features.
h_irregular = np.ones((n, n), dtype=bool)
h_irregular[1, 0] = False
h_irregular[n - 1, 2] = False
h_irregular[0, n - 2] = False
h_irregular[1, n - 2] = False
h_irregular[0, n - 1] = False
h_irregular[1, n - 1] = False
h_irregular[2, n - 1] = False

masks_irregular = Mask2D.from_mask(h_irregular)
print(
    f"Irregular: h shape = {masks_irregular.h.shape}, Wet h-cells = {int(masks_irregular.h.sum())}"
)
print(
    f"  Irregular boundary xy_corner_strict indices: {len(masks_irregular.xy_corner_strict_irrbound_cols)} cells"
)

# %% [markdown]
# ## 2. Visualising the staggered masks
#
# The `Mask2D` stores five staggered masks plus a 4-level
# land/coast classification:
#
# | Value | Meaning |
# |-------|---------|
# | 0 | land |
# | 1 | coast (adjacent to land) |
# | 2 | near-coast (one cell from coast) |
# | 3 | open ocean (interior) |
#
# Below we plot **staggered masks** (2x3 grid) and the **classification**
# (single panel with colorbar) for each of the four topologies.


# %%
def plot_staggered_masks(masks, name, img_dir):
    """Plot the 5 staggered masks + classification in a 2x3 grid."""
    fields = {
        "h (centre)": masks.h,
        "u (east face)": masks.u,
        "v (north face)": masks.v,
        "xy_corner (lenient)": masks.xy_corner,
        "xy_corner_strict": masks.xy_corner_strict,
        "classification": masks.classification,
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    for ax, (title, data) in zip(axes.ravel(), fields.items(), strict=True):
        ax.imshow(
            np.asarray(data), origin="lower", interpolation="nearest", cmap="viridis"
        )
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Staggered masks: {name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(img_dir / f"staggered_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_classification(masks, name, img_dir):
    """Plot the 4-level classification with a discrete colorbar."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        np.asarray(masks.classification),
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=3,
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.85)
    cbar.ax.set_yticklabels(["0: land", "1: coast", "2: near-coast", "3: ocean"])
    ax.set_title(f"Land/coast classification: {name}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(img_dir / f"classification_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# Generate staggered + classification figures for all 4 topologies
all_masks = {
    "basin": masks_basin,
    "island": masks_island,
    "channel": masks_channel,
    "irregular": masks_irregular,
}

for topo_name, topo_masks in all_masks.items():
    print(f"--- {topo_name} ---")
    print(f"  h: {topo_masks.h.shape}, wet = {int(topo_masks.h.sum())}")
    print(f"  u: {topo_masks.u.shape}, wet = {int(topo_masks.u.sum())}")
    print(f"  v: {topo_masks.v.shape}, wet = {int(topo_masks.v.sum())}")
    print(f"  xy_corner_strict: {topo_masks.xy_corner_strict.shape}, wet = {int(topo_masks.xy_corner_strict.sum())}")
    plot_staggered_masks(topo_masks, topo_name, IMG_DIR)
    plot_classification(topo_masks, topo_name, IMG_DIR)

# %% [markdown]
# ![Staggered masks: basin](../../images/demo_masks/staggered_basin.png)
#
# ![Classification: basin](../../images/demo_masks/classification_basin.png)
#
# ![Staggered masks: island](../../images/demo_masks/staggered_island.png)
#
# ![Classification: island](../../images/demo_masks/classification_island.png)
#
# ![Staggered masks: channel](../../images/demo_masks/staggered_channel.png)
#
# ![Classification: channel](../../images/demo_masks/classification_channel.png)
#
# ![Staggered masks: irregular](../../images/demo_masks/staggered_irregular.png)
#
# ![Classification: irregular](../../images/demo_masks/classification_irregular.png)

# %% [markdown]
# ## 3. Stencil capability and adaptive WENO masks
#
# Each cell stores how many contiguous wet neighbours it has in each
# direction ($x_+$, $x_-$, $y_+$, $y_-$).  This drives the adaptive
# stencil selection for WENO reconstruction near coastlines:
#
# - **5-point WENO** where the stencil capability $\ge 3$ in both directions
# - **3-point TVD** where the capability is 2
# - **1st-order upwind** at the coast (capability = 1)
#
# The masks are mutually exclusive: every wet cell belongs to exactly one
# stencil category.

# %%
sc = masks_island.stencil_capability
print("Stencil capability (x_pos) -- contiguous wet cells to the right:")
print(f"  shape: {sc.x_pos.shape}")
print(np.asarray(sc.x_pos))

# %%
# Get mutually-exclusive adaptive masks for x-direction reconstruction
adaptive = masks_island.get_adaptive_masks(direction="x", source="h")
print("Adaptive stencil masks (x-direction, island domain):")
for size, mask in sorted(adaptive.items()):
    count = int(mask.sum())
    if count > 0:
        print(f"  stencil size {size:2d}: {count} cells  (shape {mask.shape})")

# %% [markdown]
# ## 4. Vorticity boundary classification
#
# At xy-corner points the mask classifies cells into four
# categories based on the configuration of adjacent velocity faces:
#
# | Flag | Meaning | Boundary condition |
# |------|---------|--------------------|
# | `xy_corner_valid` | interior -- all 4 adjacent faces wet | standard curl stencil |
# | `xy_corner_y_wall` | on a vertical (y-direction) boundary | one-sided $\partial v/\partial x$ |
# | `xy_corner_x_wall` | on a horizontal (x-direction) boundary | one-sided $\partial u/\partial y$ |
# | `xy_corner_convex` | convex corner (both boundary types) | diagonal extrapolation |
#
# These flags are essential for correctly computing vorticity
# $\zeta = \partial v / \partial x - \partial u / \partial y$ at domain
# boundaries in both QG and SW models.

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ["xy_corner_valid", "xy_corner_y_wall", "xy_corner_x_wall", "xy_corner_convex"]
fields = [
    masks_island.xy_corner_valid,
    masks_island.xy_corner_y_wall,
    masks_island.xy_corner_x_wall,
    masks_island.xy_corner_convex,
]
for ax, title, field in zip(axes, titles, fields, strict=True):
    ax.imshow(np.asarray(field), origin="lower", cmap="Blues", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle("Vorticity boundary classification (island domain)", fontsize=13)
plt.tight_layout()
fig.savefig(IMG_DIR / "vorticity_boundary.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Vorticity boundary classification](../../images/demo_masks/vorticity_boundary.png)

# %% [markdown]
# ## 5. All-ocean domain shortcut
#
# For simple rectangular domains without land, use `from_dimensions`.
# This creates a mask where every cell is wet (classification = 3
# everywhere) and all stencil capabilities are maximal.

# %%
masks_simple = Mask2D.from_dimensions(ny=12, nx=12)
print(
    f"All-ocean: h shape = {masks_simple.h.shape}, "
    f"all wet = {bool(masks_simple.h.all())}, "
    f"classification range = [{int(masks_simple.classification.min())}, "
    f"{int(masks_simple.classification.max())}]"
)

# %% [markdown]
# ## 6. Physical context: QG and Shallow Water equations
#
# ### Quasi-Geostrophic (QG) equations
#
# The barotropic QG system evolves relative vorticity $\omega$ via the
# streamfunction $\psi$:
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
# **xy-corner points**, and $u, v$ at **face points**.  The mask
# ensures that $\nabla^2 \psi$ is only computed where all four corner
# values exist (i.e. where `xy_corner_strict` is True).
#
# ### Shallow Water (SW) equations (vector-invariant form)
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
# and $q$ at **xy-corner points**.  The vorticity boundary flags (section 4)
# control how $\zeta$ is computed at the domain edges.

# %%
# Demonstrate with a grid + masks for a small basin
grid = CartesianGrid2D.from_interior(8, 8, Lx=8e4, Ly=8e4)
masks = Mask2D.from_dimensions(ny=grid.Ny, nx=grid.Nx)
print(f"Grid: {grid.Ny}x{grid.Nx} (8 interior + 2 ghost = 10 per side)")
print(f"  dx = {grid.dx:.0f} m, dy = {grid.dy:.0f} m")
print(
    f"Masks: h={int(masks.h.sum())} wet, "
    f"u={int(masks.u.sum())} wet, "
    f"v={int(masks.v.sum())} wet, "
    f"xy_corner_strict={int(masks.xy_corner_strict.sum())} wet"
)

# %% [markdown]
# ## 7. Summary
#
# | Concept | Detail |
# |---------|--------|
# | **Primary input** | Binary h-mask (cell centres): 1 = wet, 0 = dry |
# | **Derived masks** | u (east face), v (north face), xy_corner (lenient), xy_corner_strict |
# | **Classification** | 4-level: land (0), coast (1), near-coast (2), ocean (3) |
# | **Stencil capability** | Per-cell count of contiguous wet neighbours in each direction |
# | **Adaptive stencils** | Mutually-exclusive masks for WENO-5 / TVD-3 / upwind-1 |
# | **Vorticity flags** | `xy_corner_valid`, `xy_corner_y_wall`, `xy_corner_x_wall`, `xy_corner_convex` |
# | **All-ocean shortcut** | `Mask2D.from_dimensions(ny, nx)` |
# | **Factory** | `Mask2D.from_mask(h_bool_array)` |
