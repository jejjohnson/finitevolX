# Arakawa C-Grid Masks

`ArakawaCGridMask` builds all staggered masks from a single cell-centre
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

| Point     | Location                  | Variable                         |
|-----------|---------------------------|----------------------------------|
| **h**     | cell centre               | tracers, height, pressure        |
| **u**     | y-face (east/west)        | zonal velocity                   |
| **v**     | x-face (north/south)      | meridional velocity              |
| **w/psi** | SW corner                 | vorticity, streamfunction        |

## Creating masks

All you need is a binary h-grid mask (True = ocean, False = land).
The factory method derives everything else:

```python
import numpy as np
from finitevolx import ArakawaCGridMask

# Rectangular basin with land boundaries
n = 10
h_mask = np.ones((n, n), dtype=bool)
h_mask[0, :] = h_mask[-1, :] = False
h_mask[:, 0] = h_mask[:, -1] = False

# With an island
h_mask[4:7, 4:7] = False

masks = ArakawaCGridMask.from_mask(h_mask)
```

For an all-ocean domain, use the shortcut:

```python
masks = ArakawaCGridMask.from_dimensions(ny=12, nx=12)
```

Or construct from an SSH field where NaN marks land:

```python
masks = ArakawaCGridMask.from_ssh(ssh_field)
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

At vorticity (w) points, cells are classified based on their relationship
to adjacent velocity faces:

- **w_valid** — interior: all 4 adjacent velocity faces are wet
- **w_vertical_bound** — on a vertical (y-direction) boundary
- **w_horizontal_bound** — on a horizontal (x-direction) boundary
- **w_cornerout_bound** — at convex corners (both boundary types)

## Stencil capability and adaptive WENO

Each cell stores how many contiguous wet neighbours it has in each
direction via `StencilCapability`. This drives adaptive stencil
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

## Operator API: how operators consume masks

Every class operator in finitevolX accepts an optional
``mask: ArakawaCGridMask | None = None`` keyword argument. The contract
is uniform across the whole library:

```python
out = op(input, ..., mask=mask)
```

When a mask is supplied, the output is **zero in every dry cell of the
operator's output stagger**. Reading the rest of the section is the
fastest way to understand exactly what that means.

### Architecture: where masks live

finitevolX is organised in three layers; only the top one knows about
masks at all:

| Layer | Examples | Mask-aware? |
|---|---|---|
| **Stencils** (pure ndarray ops) | `avg_x_fwd`, `diff_x_bwd`, … in `stencils.py` | ❌ |
| **Functional operators** (top-level functions) | `divergence_2d`, `_curl_2d`, `diffusion_2d`, `kinetic_energy`, … | ❌ |
| **Class operators** (`eqx.Module` subclasses) | `Difference2D`, `Diffusion2D`, `Vorticity2D`, `Coriolis2D`, all `Spherical*`, `BiharmonicDiffusion2D/3D`, … | ✅ |

The reason for keeping masks out of the lower layers is that the
functional / stencil layer is for advanced users composing primitives
inside a JAX `jit`. Those users either don't care about masks or
manage them explicitly at the call site:

```python
# Mask a functional output yourself — three lines, explicit, no hidden
# behaviour:
ke = kinetic_energy(u, v) * mask.h
div = divergence_2d(u * mask.u, v * mask.v, dx, dy) * mask.h
```

The class layer is the convenience layer: it knows the operator's
output stagger and picks the right `mask.<stagger>` field for you.

### Convention: post-compute zero by output stagger

For the vast majority of operators the mask convention is dead simple:

> Compute the operator as if the domain were all-ocean, then multiply
> the output by the mask field for the operator's *output* stagger.

The stagger map is mechanical:

| Output stagger | Mask field used | Examples |
|---|---|---|
| **T** (cell centres) | `mask.h` | `Divergence2D`, `Difference2D.laplacian`, `Difference2D.divergence`, `Interpolation2D.U_to_T`, `SphericalLaplacian2D` |
| **U** (east faces) | `mask.u` | `Difference2D.diff_x_T_to_U`, `Interpolation2D.T_to_U`, `Coriolis2D` (`du_cor`) |
| **V** (north faces) | `mask.v` | `Difference2D.diff_y_T_to_V`, `Interpolation2D.T_to_V`, `Coriolis2D` (`dv_cor`) |
| **X** (NE corners) | `mask.psi` | `Difference2D.curl`, `Vorticity2D.relative_vorticity`, `Interpolation2D.T_to_X`, `Difference2D.diff_y_U_to_X` |

`mask.psi` (the strict 4-of-4 corner mask) is used for X-point outputs
because vorticity at a corner is only meaningful where all four
surrounding velocity faces are wet.

In code, the implementation is two lines per method:

```python
def laplacian(self, h, mask=None):
    # Compute as if all-ocean.
    out = interior((diff_x_fwd(h) - diff_x_bwd(h)) / self.grid.dx**2 +
                   (diff_y_fwd(h) - diff_y_bwd(h)) / self.grid.dy**2, h)
    # Zero the dry T-cells.
    if mask is not None:
        out = out * mask.h
    return out
```

This is why we call it the *post-compute zero* convention. It's cheap,
mechanical, never silently changes interior physics, and it composes:
if you chain a sequence of mask-aware class operators, every
intermediate face/corner field is already zero in the dry cells, so
the next operator in the chain sees a clean input.

### Two exceptions: intermediate masking

There are exactly two operator families where the post-compute pattern
is **insufficient** and the class wrapper has to apply the mask
*inside* the computation:

#### 1. `Diffusion2D` / `Diffusion3D` / `BiharmonicDiffusion2D/3D`

Flux-form diffusion computes east/north face fluxes from `(h[i+1] -
h[i]) / dx`, then takes the divergence of those fluxes to get the
T-point tendency. If a wet T-cell is adjacent to a dry T-cell, the
unmasked flux step still reads `h[dry]` and produces a polluted
flux on that face. Multiplying the wet-cell *output* by 1 doesn't fix
that — the wet cell's tendency is already wrong.

So `Diffusion2D` masks at three steps:

```python
flux_x *= mask.u   # zero east-face fluxes through dry boundaries
flux_y *= mask.v   # zero north-face fluxes through dry boundaries
out    *= mask.h   # zero tendency in dry T-cells
```

The public functional form `diffusion_2d` is mask-free; the class
wrapper inlines this three-step masking via a private
`_diffusion_2d_impl` helper. Use `Diffusion2D` (not `diffusion_2d`)
whenever you need masked diffusion.

For `BiharmonicDiffusion2D` / `BiharmonicDiffusion3D` the mask is
applied to the **final** output only (not the intermediate Laplacian).
Zeroing the intermediate would corrupt the second-pass stencil; the
final post-compute multiply still guarantees the dry-cells-zero
invariant.

#### 2. `Advection2D` / `Advection3D`

Advection uses *adaptive stencil dispatch*: at each cell it picks the
widest WENO/TVD stencil that fits between the cell and the nearest
land. There is no post-compute equivalent — the choice of stencil at
each cell *is* the masking. This was wired up in finitevolX `0.0.39`
(see `ArakawaCGridMask.get_adaptive_masks` and
`upwind_flux`).

### Worked example: a custom masked compound operator

To compose your own mask-aware operator from finitevolX pieces, you
have two options.

**Option A — use class operators end-to-end**, and the mask propagates
automatically:

```python
from finitevolx import Difference2D, Interpolation2D

diff = Difference2D(grid=grid)
interp = Interpolation2D(grid=grid)

# Compute (u_T, v_T) at T-points by averaging from face centres,
# then take the divergence — both calls receive the mask, so the
# final tendency is zero in dry T-cells.
u_T = interp.U_to_T(u, mask=mask)
v_T = interp.V_to_T(v, mask=mask)
div = diff.divergence(u_T, v_T, mask=mask)
```

**Option B — drop down to the functional layer** for a tight inner
loop, and mask explicitly at the call site:

```python
from finitevolx._src.operators.divergence import divergence_2d
from finitevolx._src.operators.stencils import avg_x_bwd, avg_y_bwd

# All masking is in the user's hands here.
u_T = avg_x_bwd(u) * mask.h
v_T = avg_y_bwd(v) * mask.h
div = divergence_2d(u_T, v_T, dx, dy) * mask.h
```

Option A is recommended unless you have a specific reason to bypass
the class layer (e.g. fusing the operations under `jax.jit` for
performance, or building a differentiable kernel where the class
PyTree shape would be inconvenient).

### Constructing an `ArakawaCGridMask`

Use one of the factory class-methods:

```python
# All-ocean of given dimensions
mask = ArakawaCGridMask.from_dimensions(ny=NY, nx=NX)

# From a binary cell-centre wet/dry array
mask = ArakawaCGridMask.from_mask(h_mask)

# From an SSH field where NaN marks land
mask = ArakawaCGridMask.from_ssh(ssh_field)
```

Pass that single `mask` object as the `mask=` keyword to whichever
class operators need it. The operator picks the right field
(`mask.h`, `mask.u`, `mask.v`, `mask.psi`) automatically.

## Jupyter notebook

A complete interactive demo is available as a jupytext notebook:
[`notebooks/demo_masks.py`](https://github.com/jejjohnson/finitevolX/blob/main/notebooks/demo_masks.py)
