# Interpolation Operators

This page covers the theory behind the interpolation (averaging) operators in
finitevolX, explains why staggered-grid interpolation is necessary, and gives
practical guidance for composing operators.

---

## The Problem

On an Arakawa C-grid, scalars live at **T-points** (cell centres) while
velocities live at **U/V-points** (face centres) and vorticity at **X-points**
(corners).  Many operations require a quantity at a location other than where
it is naturally defined:

- Computing kinetic energy requires **u²** and **v²** both at T-points, but
  they are stored at U- and V-points respectively.
- The Coriolis term $(f\,v)$ at U-points requires **v** to be interpolated
  from V-points to U-points (a cross-face average).
- The potential vorticity $((\zeta + f)/h)$ requires **h** to be interpolated
  to X-points where $\zeta$ is defined.

finitevolX provides a complete set of second-order **linear (arithmetic)**
averaging operators that move quantities between all four location types.

---

## Staggered Grid Locations

All arrays share the same shape `[Ny, Nx]` with a one-cell ghost ring.

| Symbol | Position | Stored value |
|--------|----------|-------------|
| **T** | `(j, i)` — cell centre | Scalars: $h$, $T$, $p$ |
| **U** | `(j, i+½)` — east face | x-velocity $u$ |
| **V** | `(j+½, i)` — north face | y-velocity $v$ |
| **X** | `(j+½, i+½)` — NE corner | Vorticity $\zeta$, PV $q$ |

---

## Averaging Stencils

### 1-D (2-point) averages

For 1-D arrays or for a single direction in 2-D:

$$
\text{T} \to \text{U}: \quad h_{i+\frac{1}{2}} = \tfrac{1}{2}(h_i + h_{i+1})
$$

$$
\text{U} \to \text{T}: \quad u_i = \tfrac{1}{2}(u_{i+\frac{1}{2}} + u_{i-\frac{1}{2}})
$$

These are the building blocks for all 2-D averages.

### Face-to-face averages (bilinear 4-point)

Moving between the two face types (U ↔ V) requires a **4-point bilinear**
average because the target point is displaced in both directions:

$$
u_{j+\frac{1}{2},i} = \tfrac{1}{4}
  \bigl(u_{j,i+\frac{1}{2}} + u_{j+1,i+\frac{1}{2}}
      + u_{j,i-\frac{1}{2}} + u_{j+1,i-\frac{1}{2}}\bigr)
$$

$$
v_{j, i+\frac{1}{2}} = \tfrac{1}{4}
  \bigl(v_{j+\frac{1}{2},i} + v_{j-\frac{1}{2},i}
      + v_{j+\frac{1}{2},i+1} + v_{j-\frac{1}{2},i+1}\bigr)
$$

These cross-face averages are used in the Coriolis operator (interpolating
the cross-component velocity to the target face) and in the energy-conserving
momentum advection scheme.

### T ↔ X (corner) averages (bilinear 4-point)

$$
h_{j+\frac{1}{2}, i+\frac{1}{2}} = \tfrac{1}{4}
  \bigl(h_{j,i} + h_{j,i+1} + h_{j+1,i} + h_{j+1,i+1}\bigr)
\quad (\text{T} \to \text{X})
$$

$$
q_{j,i} = \tfrac{1}{4}
  \bigl(q_{j+\frac{1}{2},i+\frac{1}{2}} + q_{j-\frac{1}{2},i+\frac{1}{2}}
      + q_{j+\frac{1}{2},i-\frac{1}{2}} + q_{j-\frac{1}{2},i-\frac{1}{2}}\bigr)
\quad (\text{X} \to \text{T})
$$

---

## Complete Operator Map

The table below lists every averaging operator in `Interpolation2D` and its
physical interpretation.

| Method | From → To | Average | Typical use |
|--------|-----------|---------|-------------|
| `T_to_U` | T → U | ½(T_i + T_{i+1}) | Height at east face (pressure gradient) |
| `T_to_V` | T → V | ½(T_j + T_{j+1}) | Height at north face (pressure gradient) |
| `T_to_X` | T → X | ¼(T_{j,i} + T_{j,i+1} + T_{j+1,i} + T_{j+1,i+1}) | h at corners for PV = (ζ+f)/h |
| `U_to_T` | U → T | ½(U_i + U_{i-1}) | u² at T-points for kinetic energy |
| `V_to_T` | V → T | ½(V_j + V_{j-1}) | v² at T-points for kinetic energy |
| `X_to_T` | X → T | ¼(X_{j+½,i+½} + X_{j-½,i+½} + X_{j+½,i-½} + X_{j-½,i-½}) | Vorticity from corner to centre |
| `U_to_X` | U → X | ½(U_j + U_{j+1}) | u at corners for vorticity-flux |
| `V_to_X` | V → X | ½(V_i + V_{i+1}) | v at corners for vorticity-flux |
| `X_to_U` | X → U | ½(X_j + X_{j-1}) | Vorticity at face for Coriolis |
| `X_to_V` | X → V | ½(X_i + X_{i-1}) | Vorticity at face for Coriolis |
| `U_to_V` | U → V | ¼(4-point bilinear) | Cross-velocity for Coriolis term |
| `V_to_U` | V → U | ¼(4-point bilinear) | Cross-velocity for Coriolis term |

---

## Ghost-Cell Conventions

All operators write only to `[1:-1, 1:-1]`.  The ghost ring provides the
stencil data for the last interior cell:

| Direction | Ghost consumed | Slice |
|-----------|---------------|-------|
| T → U (x-forward) | East T ghost `T[:, Nx−1]` | `h[1:-1, 2:]` |
| T → V (y-forward) | North T ghost `T[Ny−1, :]` | `h[2:, 1:-1]` |
| U → T (x-backward) | West U ghost `U[:, 0]` | `u[1:-1, :-2]` |
| V → T (y-backward) | South V ghost `V[0, :]` | `v[:-2, 1:-1]` |

!!! note "Bilinear averages"
    `T_to_X`, `U_to_V`, `V_to_U`, and `X_to_T` each read ghost cells in
    *both* directions.  Make sure the appropriate BCs are applied before
    calling these operators in a boundary region.

---

## 3-D Extension

`Interpolation3D` applies the same 2-D horizontal stencils to each z-level
independently.  Array shapes are `[Nz, Ny, Nx]` and only the interior
`[1:-1, 1:-1, 1:-1]` is written.  Currently `T_to_U`, `T_to_V`, `U_to_T`,
and `V_to_T` are provided.

---

## Quick Usage

```python
import jax.numpy as jnp
from finitevolx import ArakawaCGrid2D, Interpolation2D

grid = ArakawaCGrid2D.from_interior(64, 64, 1e6, 1e6)
interp = Interpolation2D(grid=grid)

h = jnp.ones((grid.Ny, grid.Nx))   # T-point thickness
u = jnp.zeros((grid.Ny, grid.Nx))  # U-point velocity
v = jnp.zeros((grid.Ny, grid.Nx))  # V-point velocity
f = jnp.ones((grid.Ny, grid.Nx))   # T-point Coriolis parameter

# Height at east faces (for shallow-water pressure gradient)
h_u = interp.T_to_U(h)   # shape [Ny, Nx], values at U-points

# Height at NE corners (for potential vorticity)
h_q = interp.T_to_X(h)   # shape [Ny, Nx], values at X-points

# Cross-face velocity average (for Coriolis term)
v_on_u = interp.V_to_U(v)   # v interpolated to U-points
u_on_v = interp.U_to_V(u)   # u interpolated to V-points

# Coriolis parameter at velocity points
f_on_u = interp.T_to_U(f)   # f at U-points (x-average)
f_on_v = interp.T_to_V(f)   # f at V-points (y-average)
```

---

## Decision Guide

```
What quantity do you need and where?
│
├── Scalar from T-points to...
│   ├── U-points (east face)   → T_to_U
│   ├── V-points (north face)  → T_to_V
│   └── X-points (NE corner)   → T_to_X
│
├── Velocity from face to...
│   ├── U → T  (x-mean)        → U_to_T
│   ├── V → T  (y-mean)        → V_to_T
│   ├── U → X  (y-mean)        → U_to_X
│   ├── V → X  (x-mean)        → V_to_X
│   ├── X → T  (bilinear)      → X_to_T
│   ├── X → U  (y-mean)        → X_to_U
│   └── X → V  (x-mean)        → X_to_V
│
└── Cross-face velocity for Coriolis...
    ├── u at V-points           → U_to_V  (bilinear 4-pt)
    └── v at U-points           → V_to_U  (bilinear 4-pt)
```

---

## References

- Arakawa & Lamb (1977) — Staggered C-grid layout
- Sadourny (1975) — Cross-face averaging in the shallow-water system
- Veros ocean model — C-grid interpolation conventions
