# Arakawa C-Grid Masks

`Mask2D` builds all staggered masks from a single cell-centre
wet/dry field, following the Arakawa & Lamb (1977) grid layout:

```
y
^
:           :
w-----v-----w..
|           |
|           |
u     h     u
|           |
|           |
w-----v-----w..   > x
```

| Point                | Location                  | Variable                         |
|----------------------|---------------------------|----------------------------------|
| **h**                | cell centre               | tracers, height, pressure        |
| **u**                | east face                 | zonal velocity                   |
| **v**                | north face                | meridional velocity              |
| **xy_corner**        | NE corner (lenient)       | vorticity                        |
| **xy_corner_strict** | NE corner (strict)        | streamfunction                   |

All staggered points use the *same-index, positive half-step* convention:
``u[j, i]`` is the east face of ``h[j, i]``, ``v[j, i]`` is the north face,
and ``xy_corner[j, i]`` is the NE corner — matching ``CartesianGrid2D``.
See [the Arakawa C-grid discretization notes](cgrid_discretization.md) for
the underlying convention and a comparison with negative half-step
alternatives.

## Creating masks

All you need is a binary h-grid mask (True = ocean, False = land).
The factory method derives everything else:

```python
import numpy as np
from finitevolx import Mask2D

# Rectangular basin with land boundaries
n = 10
h_mask = np.ones((n, n), dtype=bool)
h_mask[0, :] = h_mask[-1, :] = False
h_mask[:, 0] = h_mask[:, -1] = False

# With an island
h_mask[4:7, 4:7] = False

masks = Mask2D.from_mask(h_mask)
```

For an all-ocean domain, use the shortcut:

```python
masks = Mask2D.from_dimensions(ny=12, nx=12)
```

Or construct from any float field at a known grid position, where
`NaN` marks dry cells:

```python
# From a field at cell centres (T-points): SSH, temperature, pressure, …
masks = Mask2D.from_center(ssh_field)

# From a field at u-faces, v-faces, or xy-corners.  Pass `mode=` to
# choose the inversion strategy when the inverse mapping back to the
# h-grid is non-unique:
#   mode='permissive'  → h wet iff *any* surrounding face/corner wet
#   mode='conservative' → h wet iff *all* surrounding face/corner wet
masks = Mask2D.from_u_face(u_field, mode="permissive")
masks = Mask2D.from_v_face(v_field, mode="permissive")
masks = Mask2D.from_corner(vorticity_field, mode="conservative")
```

## Staggered variable locations

Each variable type sits at a different position within a grid cell.
The figures below show the actual staggered positions for several
domain topologies:

### Rectangular basin

![Rectangular basin](images/demo_masks/staggered_basin.png)

### Basin with island

![Basin with island](images/demo_masks/staggered_island.png)

### Zonal channel

![Zonal channel](images/demo_masks/staggered_channel.png)

### Irregular coastline

![Irregular coastline](images/demo_masks/staggered_irregular.png)

## Land / coast classification

The mask includes a 4-level classification (0 = land, 1 = coast,
2 = near-coast, 3 = open ocean):

### Rectangular basin

![Classification: basin](images/demo_masks/classification_basin.png)

### Basin with island

![Classification: island](images/demo_masks/classification_island.png)

### Zonal channel

![Classification: channel](images/demo_masks/classification_channel.png)

### Irregular coastline

![Classification: irregular](images/demo_masks/classification_irregular.png)

## Vorticity boundary classification

At xy-corner points, cells are classified based on their relationship
to adjacent velocity faces:

- **xy_corner_valid** — interior: all 4 adjacent velocity faces are wet
- **xy_corner_y_wall** — on a vertical (y-direction) boundary
- **xy_corner_x_wall** — on a horizontal (x-direction) boundary
- **xy_corner_convex** — at convex corners (both boundary types)

## Stencil capability and adaptive WENO

Each cell stores how many contiguous wet neighbours it has in each
direction via `StencilCapability2D`. This drives adaptive stencil
selection for WENO reconstruction near coastlines:

```python
# Mutually-exclusive masks: largest usable stencil at each point
adaptive = masks.get_adaptive_masks(direction="x", source="h")
# adaptive[2]  → 1st-order upwind only
# adaptive[4]  → WENO3
# adaptive[6]  → WENO5
# adaptive[8]  → WENO7
# adaptive[10] → WENO9
```

## Jupyter notebook

A complete interactive demo is available as a jupytext notebook:
[`notebooks/demo_masks.py`](https://github.com/jejjohnson/finitevolX/blob/main/notebooks/demo_masks.py)
