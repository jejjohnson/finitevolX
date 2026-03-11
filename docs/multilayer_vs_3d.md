# Multilayer vs. 3D Discretization

This page explains the crucial distinction between *multilayer* operators
(via `jax.vmap` over independent 2D layers) and *true 3D* operators in
`finitevolX`, and shows how to use each correctly.

---

## Two Different Meanings of a Leading Axis

When you have an array of shape `[N, Ny, Nx]` in ocean or atmosphere
modelling, the leading axis `N` can mean one of two fundamentally different
things:

| Use case | Leading axis meaning | Example models |
|---|---|---|
| **Multilayer / baroclinic / modal** | Layer or mode index | QG *n*-layer, baroclinic SW |
| **True 3D discretization** | Vertical spatial dimension *z* | Primitive equations |

`finitevolX` provides explicit support for both, but **they must not be
confused** because their ghost-cell conventions are opposite.

---

## Multilayer: `multilayer(fn)`

### Concept

In a multilayer model (e.g., *n*-layer quasi-geostrophic or baroclinic
shallow water), the fluid column is divided into `nl` discrete layers.
Each layer is a complete, physically real 2D horizontal field with its own
dynamics.  There are **no "ghost" layers** — every slice `k = 0 … nl-1`
holds a real physical field.

The JAX-idiomatic way to apply any 2D horizontal operator to all layers at
once is `jax.vmap`, which vectorises the call over the leading axis without
any overhead.  `finitevolX` exposes this pattern through the
`multilayer` helper:

```python
import jax.numpy as jnp
import finitevolx as fvx

grid = fvx.ArakawaCGrid2D.from_interior(64, 64, 1e4, 1e4)
diff2d = fvx.Difference2D(grid=grid)

nl = 3                                     # number of layers
h = jnp.ones((nl, grid.Ny, grid.Nx))      # [nl, Ny, Nx]

# Apply diff_x_T_to_U to every layer in one vectorised call
dh_dx = fvx.multilayer(diff2d.diff_x_T_to_U)(h)   # shape [nl, Ny, Nx]
```

For operators that take multiple arguments (e.g., `divergence`), pass the
bound method directly — :func:`jax.vmap` batches each positional argument
over the leading axis independently:

```python
u = jnp.ones((nl, grid.Ny, grid.Nx))
v = jnp.ones((nl, grid.Ny, grid.Nx))

div = fvx.multilayer(diff2d.divergence)(u, v)
```

### What `multilayer` does

- Wraps `fn` with `jax.vmap`, batching over the **first** axis.
- The 2D stencil (ghost-cell ring, interior writes) is completely unchanged.
- **All** `nl` slices are processed; there is no "outer ghost layer" concept.

---

## True 3D: `Difference3D`

### Concept

In a true 3D primitive-equation model, the leading axis is a *spatial*
vertical coordinate.  The discretization uses the same ghost-cell convention
as the horizontal dimensions: the **top** (`k = 0`) and **bottom**
(`k = Nz-1`) layers are ghost shells that must be filled by vertical boundary
conditions before chaining operators.  Only the interior `k = 1 … Nz-2`
layers are written by the stencil.

```python
grid3d = fvx.ArakawaCGrid3D.from_interior(64, 64, 10, 1e4, 1e4, 50.0)
diff3d = fvx.Difference3D(grid=grid3d)

# h has shape [Nz, Ny, Nx]; Nz = 12 total (10 interior + 2 ghost)
h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))

dh_dx = diff3d.diff_x_T_to_U(h)
# dh_dx[0]  = 0   ← ghost layer (k=0), not written
# dh_dx[1:-1] ≠ 0 ← interior layers, written
# dh_dx[-1] = 0   ← ghost layer (k=Nz-1), not written
```

---

## Key Differences at a Glance

| Property | `multilayer(fn)` | `Difference3D` |
|---|---|---|
| Leading axis meaning | Layer / mode index | Vertical *z* dimension |
| Ghost layers at k=0, k=N-1 | ❌ None — all slices are real | ✅ Required — top/bottom are ghost shells |
| Which slices are written | All `k = 0 … nl-1` | Only interior `k = 1 … Nz-2` |
| Underlying mechanism | `jax.vmap(fn)` over axis 0 | Explicit `[1:-1, …]` slice writes |
| Correct for | Baroclinic / QG layered models | Primitive-equation 3D models |

---

## Equivalence in the Interior

Although the two approaches differ at the boundary layers, they produce
**identical results in the horizontal interior** for the shared interior
z-levels:

```python
import jax.numpy as jnp
import numpy as np
import finitevolx as fvx

grid2d = fvx.ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
grid3d = fvx.ArakawaCGrid3D.from_interior(8, 8, 4, 1.0, 1.0, 1.0)

diff2d = fvx.Difference2D(grid=grid2d)
diff3d = fvx.Difference3D(grid=grid3d)

Nz = grid3d.Nz   # 6 total (4 interior + 2 ghost)
h = jnp.arange(Nz * grid3d.Ny * grid3d.Nx, dtype=float).reshape(
    Nz, grid3d.Ny, grid3d.Nx
)

ml_result = fvx.multilayer(diff2d.diff_x_T_to_U)(h)
d3_result = diff3d.diff_x_T_to_U(h)

# Interior z-levels AND horizontal interior agree exactly
np.testing.assert_allclose(
    ml_result[1:-1, 1:-1, 1:-1],
    d3_result[1:-1, 1:-1, 1:-1],
)

# But the boundary layers differ:
# Difference3D: k=0, k=5 are zero (ghost shells)
print(d3_result[0].max())    # 0.0
print(d3_result[-1].max())   # 0.0

# multilayer: k=0, k=5 are real (non-zero) layers
print(ml_result[0].max())    # non-zero
print(ml_result[-1].max())   # non-zero
```

---

## Summary

- Use **`multilayer(fn)`** when your leading axis indexes independent
  physical layers or modes.  Every layer is real; `jax.vmap` is the
  correct, efficient way to batch over them.
- Use **`Difference3D`** when your leading axis is a spatial vertical
  coordinate that requires top/bottom ghost shells for boundary conditions.
- The two are **not interchangeable**.  Applying `Difference3D` to a true
  multilayer field silently zeros out the top and bottom layers, which are
  physically real in that context.
