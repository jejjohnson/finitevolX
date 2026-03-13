# Finite-Difference Operators

This page covers the theory behind the finite-difference operators in finitevolX,
explains the staggered-grid indexing conventions, and provides practical guidance
for choosing and composing operators.

---

## The Problem

Ocean and atmosphere models compute spatial derivatives of scalar and vector fields
at each time step.  On an Arakawa C-grid the variables live at different staggered
locations, so the derivative of a quantity at one location must be evaluated at a
*different* location — and the rule for where the output lives is determined by
the direction and sign of the shift.

The core question is:

> Given a field at location **A**, how do we compute its derivative and what
> location does the result live at?

finitevolX answers this through a consistent **forward / backward** convention
that is the backbone of all higher-level operators (divergence, vorticity,
Laplacian, diffusion, advection, etc.).

---

## Staggered Grid Locations

All arrays share the same shape `[Ny, Nx]` with a one-cell ghost ring on each side.
The physical interior is `[1:-1, 1:-1]`.  Four location types are defined:

| Symbol | Location | Array indices | Description |
|--------|----------|---------------|-------------|
| **T** | Cell centre | `T[j, i]` at `(j, i)` | Scalars: thickness, pressure, temperature |
| **U** | East face | `U[j, i]` at `(j, i+½)` | x-velocity, x-flux |
| **V** | North face | `V[j, i]` at `(j+½, i)` | y-velocity, y-flux |
| **X** | NE corner | `X[j, i]` at `(j+½, i+½)` | Vorticity, PV, corner quantities |

!!! note "Same-index convention"
    `U[j, i]` stores the value at the **east** face of cell `(j, i)`, i.e. at
    position `(j, i+½)`.  Similarly `V[j, i]` is at the **north** face `(j+½, i)`.
    This means the forward difference `T → U` uses `h[i+1] − h[i]`, whereas
    the backward difference `U → T` uses `u[i] − u[i−1]`.

---

## Forward vs. Backward Differences

### Forward differences

A **forward** difference moves the output one half-step in the **positive**
direction relative to the input:

$$
T \xrightarrow{\partial/\partial x} U, \qquad
T \xrightarrow{\partial/\partial y} V, \qquad
U \xrightarrow{\partial/\partial y} X, \qquad
V \xrightarrow{\partial/\partial x} X
$$

| Operator | Stencil | Use case |
|----------|---------|----------|
| `diff_x_T_to_U` | $(h_{i+1} - h_i)/\Delta x$ | Pressure gradient force, diffusion flux |
| `diff_y_T_to_V` | $(h_{j+1} - h_j)/\Delta y$ | Pressure gradient force, diffusion flux |
| `diff_y_U_to_X` | $(u_{j+1} - u_j)/\Delta y$ | Part of vorticity $\partial u / \partial y$ |
| `diff_x_V_to_X` | $(v_{i+1} - v_i)/\Delta x$ | Part of vorticity $\partial v / \partial x$ |

### Backward differences

A **backward** difference moves the output one half-step in the **negative**
direction:

$$
U \xrightarrow{\partial/\partial x} T, \qquad
V \xrightarrow{\partial/\partial y} T, \qquad
X \xrightarrow{\partial/\partial y} U, \qquad
X \xrightarrow{\partial/\partial x} V
$$

| Operator | Stencil | Use case |
|----------|---------|----------|
| `diff_x_U_to_T` | $(u_i - u_{i-1})/\Delta x$ | Divergence, tracer tendency from flux |
| `diff_y_V_to_T` | $(v_j - v_{j-1})/\Delta y$ | Divergence, tracer tendency from flux |
| `diff_y_X_to_U` | $(q_j - q_{j-1})/\Delta y$ | Coriolis & vorticity-flux term |
| `diff_x_X_to_V` | $(q_i - q_{i-1})/\Delta x$ | Coriolis & vorticity-flux term |

!!! tip "Forward → Backward pairs"
    Every forward operator has a corresponding backward operator that reverses
    the shift.  Composing `diff_x_T_to_U` followed by `diff_x_U_to_T`
    gives a second-order centred difference at T-points (i.e. the Laplacian
    in x), which is the basis for the diffusion operator.

---

## Compound Operators

### Divergence

The **discrete divergence** of a 2-D velocity field $(u, v)$ at T-points:

$$
\delta_{j,i} = \frac{u_{j,i+\frac{1}{2}} - u_{j,i-\frac{1}{2}}}{\Delta x}
             + \frac{v_{j+\frac{1}{2},i} - v_{j-\frac{1}{2},i}}{\Delta y}
$$

This is the composition of two backward differences: `diff_x_U_to_T` + `diff_y_V_to_T`.
It is available as `Difference2D.divergence(u, v)` or the standalone
`divergence_2d(u, v, dx, dy)`.

### Curl (Relative Vorticity)

The **discrete curl** of $(u, v)$ at X-points (corners):

$$
\zeta_{j+\frac{1}{2}, i+\frac{1}{2}}
  = \frac{v_{j+\frac{1}{2}, i+1} - v_{j+\frac{1}{2}, i}}{\Delta x}
  - \frac{u_{j+1, i+\frac{1}{2}} - u_{j, i+\frac{1}{2}}}{\Delta y}
$$

This is `diff_x_V_to_X` − `diff_y_U_to_X`, available as `Difference2D.curl(u, v)`.

### Laplacian

The **5-point discrete Laplacian** at T-points:

$$
\nabla^2 h_{j,i}
  = \frac{h_{j,i+1} - 2h_{j,i} + h_{j,i-1}}{\Delta x^2}
  + \frac{h_{j+1,i} - 2h_{j,i} + h_{j-1,i}}{\Delta y^2}
$$

Available as `Difference2D.laplacian(h)`.  Used directly in the diffusion and
multigrid solver operators.

### Perpendicular Gradient (Geostrophic Velocity)

The operator `grad_perp` maps a T-point streamfunction $\psi$ to C-grid face
velocities via the geostrophic relation $(u, v) = (-\partial\psi/\partial y,\,
+\partial\psi/\partial x)$, using a compact 4-point stencil that reads T-point
ghost cells directly:

$$
u_{j, i+\frac{1}{2}}
  = -\frac{\psi_{j+1,i} + \psi_{j+1,i+1} - \psi_{j-1,i} - \psi_{j-1,i+1}}{4\,\Delta y}
$$

$$
v_{j+\frac{1}{2}, i}
  = +\frac{\psi_{j,i+1} + \psi_{j+1,i+1} - \psi_{j,i-1} - \psi_{j+1,i-1}}{4\,\Delta x}
$$

The resulting velocity field is **discretely non-divergent**: $\delta(u, v) = 0$
at all interior T-points.  This is the standard geostrophic velocity reconstruction
used in QG models.

---

## Ghost-Cell Conventions

All operators write only to `[1:-1, 1:-1]` interior cells.  The one-cell ghost
ring is used as follows:

| Difference direction | Ghost consumed | Slice |
|----------------------|---------------|-------|
| Forward x (T → U) | East T ghost `T[:,  Nx−1]` | `h[1:-1, 2:]` at last col |
| Forward y (T → V) | North T ghost `T[Ny−1, :]` | `h[2:, 1:-1]` at last row |
| Backward x (U → T) | West U ghost `U[:, 0]` | `u[1:-1, :-2]` at first col |
| Backward y (V → T) | South V ghost `V[0, :]` | `v[:-2, 1:-1]` at first row |

!!! warning "Caller responsibility"
    Ghost cells for **backward** operators (west U-face, south V-face) are **not**
    filled by any forward operator — the caller must set them via boundary
    conditions before composing operators.  Leaving them at zero is equivalent to
    imposing a homogeneous Dirichlet condition on the flux, which is correct for
    closed-wall BCs but wrong for periodic or open boundaries.

---

## 3-D Extension

`Difference3D` applies the same 2-D horizontal stencils independently to each
z-level.  Array shapes are `[Nz, Ny, Nx]` and only the interior
`[1:-1, 1:-1, 1:-1]` is written.

---

## Quick Usage

```python
import jax.numpy as jnp
from finitevolx import ArakawaCGrid2D, Difference2D

grid = ArakawaCGrid2D.from_interior(ny=64, nx=64, Ly=1e6, Lx=1e6)
diff = Difference2D(grid=grid)

h  = jnp.ones((grid.Ny, grid.Nx))  # T-point field

# Forward difference: h at T-points → pressure gradient at U-points
dh_dx = diff.diff_x_T_to_U(h)   # shape [Ny, Nx], values at U-points

# Backward difference: u at U-points → divergence contribution at T-points
u = jnp.zeros((grid.Ny, grid.Nx))
v = jnp.zeros((grid.Ny, grid.Nx))
div = diff.divergence(u, v)      # shape [Ny, Nx], values at T-points

# Curl: relative vorticity at X-points
zeta = diff.curl(u, v)           # shape [Ny, Nx], values at X-points

# Laplacian: ∇²h at T-points
lap_h = diff.laplacian(h)        # shape [Ny, Nx], values at T-points

# Geostrophic velocity from streamfunction
psi = jnp.zeros((grid.Ny, grid.Nx))
u_geo, v_geo = diff.grad_perp(psi)  # u at U-points, v at V-points
```

---

## Decision Guide

```
What derivative do you need?
│
├── ∂(·)/∂x from T-points → diff_x_T_to_U  (result at U-points)
├── ∂(·)/∂y from T-points → diff_y_T_to_V  (result at V-points)
├── ∂(·)/∂y from U-points → diff_y_U_to_X  (result at X-points)
├── ∂(·)/∂x from V-points → diff_x_V_to_X  (result at X-points)
│
├── ∂(·)/∂x from U-points → diff_x_U_to_T  (result at T-points)
├── ∂(·)/∂y from V-points → diff_y_V_to_T  (result at T-points)
├── ∂(·)/∂y from X-points → diff_y_X_to_U  (result at U-points)
├── ∂(·)/∂x from X-points → diff_x_X_to_V  (result at V-points)
│
├── ∇·(u, v) at T-points  → diff.divergence(u, v)
├── ζ = ∇×(u, v) at X-pts → diff.curl(u, v)
├── ∇²h at T-points       → diff.laplacian(h)
└── geostrophic u, v      → diff.grad_perp(psi)
```

---

## References

- Arakawa & Lamb (1977) — Staggered C-grid layout and half-index notation
- Sadourny (1975) — Discretisation of the shallow-water equations on a C-grid
- Veros ocean model — C-grid operator conventions
