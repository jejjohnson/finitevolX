# Arakawa C-Grid Discretization

This document describes the numerical discretization scheme used throughout
`finitevolX`, explaining the same-size array convention, ghost-cell layout,
variable co-location, and the slicing patterns that implement each stencil.

---

## Overview

`finitevolX` uses an **Arakawa C-grid**, a staggered finite-volume mesh where
different physical variables live at different grid locations.  The key
departure from many textbook implementations is that **every array has the
same total shape `[Ny, Nx]`**, including one ring of ghost cells on each
side.  There are no separate `[Ny, Nx+1]` or `[Nx-1, Ny-1]` arrays.

---

## Why same-shape storage?

The textbook way to discretise a staggered grid is to give each variable the
exact array shape it needs.  For an `Nx × Ny` grid of cell centres `T`, this
typically means:

```
T : [Ny,     Nx    ]    cell centres        — Nx*Ny    values
U : [Ny,     Nx + 1]    east faces          — Nx+1 columns
V : [Ny + 1, Nx    ]    north faces         — Ny+1 rows
X : [Ny + 1, Nx + 1]    NE corners          — both sides extended
```

Each array is exactly the right size, and `U[j, i]` lives at the east face
of `T[j, i]` for `i ∈ [0, Nx-1]`, with `U[j, Nx]` being the rightmost
boundary face.  This is a perfectly defensible convention — and it's what
many production ocean models use.

`finitevolX` makes a different choice: **every staggered field has the same
shape `[Ny, Nx]` as `T`, with a one-cell ghost ring on each side.**  The
physical interior occupies `[1:-1, 1:-1]`; the outer ring carries
boundary-condition data set by the caller.  Under this convention,
`U[j, i]` still lives at the east face of `T[j, i]`, but now the index
`i = Nx-1` falls in the ghost ring instead of being an extra "boundary
face" slot.

The advantages of same-shape storage compound across the rest of the
library:

1. **Static shapes for JAX JIT.**  Every array has the same shape, derived
   from a single `(Ny, Nx)` pair on the `Grid` object.  JIT-compiled
   kernels have one statically known shape for every field, no special
   cases for U/V/X.

2. **Free broadcasting in field arithmetic.**  Expressions like
   `h * u**2 + v**2 - p` just work — no reshaping, no padding, no
   `jnp.pad` shims.  This is especially useful for diagnostics
   (kinetic energy, Bernoulli potential, vorticity) that mix all four
   point types.

3. **Uniform `vmap` semantics.**  When a 2D operator is vectorised over a
   batch axis (multilayer ensembles, ensemble Kalman filters,
   parameter sweeps), every input has the same leading shape.  No
   "this one is `Nx+1`-wide and the others are `Nx`" gotchas.

4. **One ghost-cell convention to learn.**  All four point types share
   the same `[1:-1, 1:-1]` interior slice.  Boundary conditions are
   applied uniformly: write into the ghost ring, then call the operator.
   You never have to remember "U has an extra east column but V has an
   extra north row."

5. **Clean composition of stencils.**  A finite-difference operator
   reads `arr[1:-1, 2:]` and `arr[1:-1, 1:-1]`, writes the result to
   `out[1:-1, 1:-1]`, and leaves the ghost ring at zero.  Every shifted
   slice has the same `(Ny-2, Nx-2)` shape because the ghost ring
   absorbs the edge — a single-step shift never reads outside the array
   bounds.

The cost is that you have to internalise one fixed convention: **index
`0` and index `N-1` along each axis are ghost; real data lives at
`[1:-1]`**.  Once that's in muscle memory, the rest of the library is
remarkably uniform — the same slicing patterns appear in every operator
in this document.

---

## Variable Co-location

Four distinct locations are used, identified by letter:

| Symbol | Name        | Physical location          | Same-index meaning                       |
|--------|-------------|----------------------------|------------------------------------------|
| `T`    | cell centre | `(j·dy,      i·dx     )`  | `T[j, i]`  lives at `(j,     i    )`    |
| `U`    | east face   | `(j·dy,     (i+½)·dx  )`  | `U[j, i]`  lives at `(j,     i+½  )`    |
| `V`    | north face  | `((j+½)·dy,  i·dx     )`  | `V[j, i]`  lives at `(j+½,   i    )`    |
| `X`    | NE corner   | `((j+½)·dy, (i+½)·dx  )`  | `X[j, i]`  lives at `(j+½,   i+½  )`    |

The "same-index" rule means that array index `[j, i]` encodes the
**south-west** corner of the stencil neighbourhood:

```
   X[j,i] ---V[j,i]--- X[j,i+1]
     |           |           |
   U[j,i]  T[j,i]  U[j,i+1]
     |           |           |
  X[j-1,i]--V[j-1,i]--X[j-1,i+1]
```

---

## Ghost Cells

For a grid with `Nx × Ny` total cells, the **physical interior** occupies
indices `[1:-1, 1:-1]` (shape `(Ny-2) × (Nx-2)`).  The outer ring —
rows `0` and `Ny-1`, columns `0` and `Nx-1` — consists of **ghost cells**
reserved for boundary conditions.

### T-point ghost cells

```
 col:   0    1    2  ...  Nx-2  Nx-1
row 0: [g]  [g]  [g] ...  [g]  [g]   ← ghost (south)
row 1: [g]  [ ]  [ ] ...  [ ]  [g]   ← first interior row
  ...        interior               ...
row Ny-2:[g][ ]  [ ] ...  [ ]  [g]   ← last interior row
row Ny-1:[g] [g] [g] ...  [g]  [g]   ← ghost (north)
        ^                        ^
     ghost                    ghost
     (west)                   (east)
```

### U-point ghost cells

`U[j, i]` sits at east face `(j, i+1/2)`.

| Index | Physical meaning | Who sets it? |
|-------|-----------------|--------------|
| `U[j, 0]` | West **boundary face** `(j, ½)` | BC layer (e.g. periodic copy) |
| `U[j, 1..Nx-2]` | Interior east faces | Forward operators (`diff_x_T_to_U`, `T_to_U`) |
| `U[j, Nx-1]` | Outside domain | Unused / left zero |
| `U[0, i]`, `U[Ny-1, i]` | South/north ghost rows | BC layer |

`U[j, Nx-2]` is the **east boundary face** — it sits inside `[1:-1, 1:-1]`
and is computed by forward operators using the east ghost T-cell `T[j, Nx-1]`.

### V-point ghost cells

`V[j, i]` sits at north face `(j+½, i)`.

| Index | Physical meaning | Who sets it? |
|-------|-----------------|--------------|
| `V[0, i]` | South **boundary face** `(½, i)` | BC layer |
| `V[1..Ny-2, i]` | Interior north faces | Forward operators (`diff_y_T_to_V`, `T_to_V`) |
| `V[Ny-1, i]` | Outside domain | Unused / left zero |
| `V[j, 0]`, `V[j, Nx-1]` | West/east ghost cols | BC layer |

`V[Ny-2, i]` is the **north boundary face** — computed using north ghost T.

### X-point ghost cells

`X[j, i]` sits at NE corner `(j+½, i+½)`.

| Index | Physical meaning | Who sets it? |
|-------|-----------------|--------------|
| `X[0, i]` | South ghost X-row | BC layer |
| `X[j, 0]` | West ghost X-col | BC layer |
| `X[1..Ny-2, 1..Nx-2]` | Interior corners | Forward operators (`T_to_X`, `diff_y_U_to_X`, etc.) |
| `X[Ny-1, i]`, `X[j, Nx-1]` | Outside domain | Unused / left zero |

### Ghost asymmetry: forward vs backward

The same `[1:-1, 1:-1]` write range has **different ghost-cell implications**
depending on direction:

| Operator direction | Last interior output uses... | First interior output reads... |
|--------------------|------------------------------|-------------------------------|
| T→U (forward x)    | east ghost T `T[j, Nx-1]`   | —                             |
| T→V (forward y)    | north ghost T `T[Ny-1, i]`  | —                             |
| V→X (forward x)    | east ghost V `V[j, Nx-1]`   | —                             |
| U→X (forward y)    | north ghost U `U[Ny-1, i]`  | —                             |
| U→T (backward x)   | —                            | west ghost U `U[j, 0]`        |
| V→T (backward y)   | —                            | south ghost V `V[0, i]`       |
| X→U (backward y)   | —                            | south ghost X `X[0, i]`       |
| X→V (backward x)   | —                            | west ghost X `X[j, 0]`        |
| U→V (cross)        | north ghost U `U[Ny-1, i]`  | —                             |
| V→U (cross)        | east ghost V `V[j, Nx-1]`   | —                             |

**Operators write only to `[1:-1, 1:-1]`.**  Ghost cells remain at their
initialised value (typically zero).  Callers are responsible for filling
ghosts via boundary-condition helpers (`pad_interior`, `enforce_periodic`,
`BoundaryConditionSet`, etc.) before the next operator call.

---

## Vertical staggering in 3D

In 3D the horizontal staggering repeats **at each z-level**.  Every
`[k, :, :]` slab looks exactly like the 2D picture above:

```
T[k, j, i]  cell centre  at ( k,    j,    i    )
U[k, j, i]  east face    at ( k,    j,    i+½  )
V[k, j, i]  north face   at ( k,    j+½,  i    )
X[k, j, i]  NE corner    at ( k,    j+½,  i+½  )
```

All of these are `[Nz, Ny, Nx]`, still same-shape, and the horizontal
operators in `Difference3D` / `Interpolation3D` / `Vorticity3D` simply
apply the corresponding 2D stencil at each fixed `k`.  The vertical
ghost convention mirrors the horizontal: `k = 0` and `k = Nz - 1` are
top/bottom ghost shells, and the physical interior is
`k = 1 … Nz - 2` (see [Multilayer vs. 3D](multilayer_vs_3d.md) for the
distinction with multilayer/baroclinic models).

The interesting question is what to do about the **vertical interfaces**
— the faces between `T[k, j, i]` and `T[k+1, j, i]`, where vertical
velocity `w` lives.  There are two viable conventions, and `finitevolX`
currently uses *both* depending on the subsystem.

### Convention A — same-shape `[Nz, Ny, Nx]` (preferred)

Apply the same trick as the horizontal: store w-faces at `[Nz, Ny, Nx]`
under a positive-half-step rule:

```
w[k, j, i]  at ( k+½,  j,  i )    ← top face of T[k, j, i]
```

`w[k]` is then the top face of cell `k` (equivalently the bottom face of
cell `k+1`).  Both physical vertical boundaries — the sea floor at the
bottom and the sea surface at the top — are absorbed into the ghost
shells, just like the horizontal walls.  An interior cell `T[k, j, i]`
has two adjacent w-faces: `w[k, j, i]` above and `w[k-1, j, i]` below.

This is what **`Mask3D.w`** uses, and it composes cleanly with the rest
of the 3D mask machinery: `mask.h`, `mask.u`, `mask.v`, `mask.w`, and
`mask.xy_corner` all share the same `[Nz, Ny, Nx]` shape, so any field
arithmetic broadcasts without reshaping.

### Convention B — separate `[Nz+1, Ny, Nx]` array

Store w-faces in a *longer* array, one extra index in z:

```
w[0,    :, :]    sea floor             ← bottommost physical interface
w[1,    :, :]    interface between T[0] and T[1]
...
w[Nz,   :, :]    sea surface           ← topmost physical interface
```

Under this convention every physical interface is explicitly indexable.
Every cell `T[k]` has two adjacent faces: `w[k]` below and `w[k+1]`
above.  Boundary values can be set directly — `w[0] = 0` for a rigid
bottom, `w[Nz] = ∂η/∂t` for a free surface.

This is what **`vertical_velocity`** in
[`finitevolx._src.operators.diagnostics`](api/diagnostics.md) uses,
because it integrates the horizontal divergence from bottom to top
(`w[k+1] = w[k] − div_h[k] · dz`) and benefits from having `w[0] = 0` as
an explicit initial condition.

### Tradeoffs

| Property | Convention A — `[Nz, Ny, Nx]` | Convention B — `[Nz+1, Ny, Nx]` |
|---|---|---|
| Broadcasts with `h` / `u` / `v` | ✅ | ❌ |
| Consistent with horizontal staggering | ✅ | ❌ |
| `vmap` over z-levels across fields | ✅ | ❌ (shape mismatch) |
| Number of data-layout categories to learn | 1 | 2 |
| Explicit bottom boundary index | ❌ (in ghost shell) | ✅ (`w[0]`) |
| Explicit top boundary index | ❌ (in ghost shell) | ✅ (`w[Nz]`) |
| Clean free-surface / rigid-lid BC application | Awkward | Natural |

In practice convention **A** wins on code hygiene (uniform shapes,
broadcasting, vmap, fewer special cases), but convention **B** wins on
"naturalness" for the one operation that genuinely needs both vertical
boundaries as first-class indices: `vertical_velocity`.

The intent of `finitevolX` going forward is to standardise on **convention
A** so that every staggered field — horizontal *and* vertical — has the
same `[Nz, Ny, Nx]` shape and the same ghost-cell semantics.  See
[issue #210](https://github.com/jejjohnson/finitevolX/issues/210) for the
tracking issue and the planned refactor of `vertical_velocity`.

---

## Creating a Grid

```python
from finitevolx import ArakawaCGrid2D

# 64 physical cells in each direction; 66×66 total array shape
grid = ArakawaCGrid2D.from_interior(nx_interior=64, ny_interior=64,
                                     Lx=1.0, Ly=1.0)
# grid.Nx == 66, grid.Ny == 66
# grid.dx == 1/64, grid.dy == 1/64
```

All field arrays are then allocated as `jnp.zeros((grid.Ny, grid.Nx))`.

---

## Difference Operators

Every finite-difference stencil is a **one-cell shift divided by the grid
spacing**.  The direction of the shift (forward or backward) is determined
by which point type the output lives at.

### Forward differences (T → face / corner)

| Method              | Stencil formula                                         | Writes to         |
|---------------------|---------------------------------------------------------|-------------------|
| `diff_x_T_to_U`     | `dh/dx[j, i+½] = (h[j, i+1] - h[j, i]) / dx`          | U-points          |
| `diff_y_T_to_V`     | `dh/dy[j+½, i] = (h[j+1, i] - h[j, i]) / dy`          | V-points          |
| `diff_y_U_to_X`     | `du/dy[j+½, i+½] = (u[j+1, i] - u[j, i]) / dy`        | X-points          |
| `diff_x_V_to_X`     | `dv/dx[j+½, i+½] = (v[j, i+1] - v[j, i]) / dx`        | X-points          |

**Example slice** for `diff_x_T_to_U` (writes to `[1:-1, 1:-1]`):

```python
# dh_dx[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx
out = out.at[1:-1, 1:-1].set((h[1:-1, 2:] - h[1:-1, 1:-1]) / dx)
```

The slice `h[1:-1, 2:]` gives `h[j, i+1]` for `i` ranging over interior
columns; `h[1:-1, 1:-1]` gives `h[j, i]`.

### Backward differences (face / corner → T)

| Method              | Stencil formula                                         | Writes to         |
|---------------------|---------------------------------------------------------|-------------------|
| `diff_x_U_to_T`     | `du/dx[j, i] = (u[j, i] - u[j, i-1]) / dx`             | T-points          |
| `diff_y_V_to_T`     | `dv/dy[j, i] = (v[j, i] - v[j-1, i]) / dy`             | T-points          |
| `diff_y_X_to_U`     | `dq/dy[j, i+½] = (q[j, i] - q[j-1, i]) / dy`           | U-points          |
| `diff_x_X_to_V`     | `dq/dx[j+½, i] = (q[j, i] - q[j, i-1]) / dx`           | V-points          |

**Example slice** for `diff_x_U_to_T`:

```python
# du_dx[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx
#             = (u[j, i]     - u[j, i-1]   ) / dx
out = out.at[1:-1, 1:-1].set((u[1:-1, 1:-1] - u[1:-1, :-2]) / dx)
```

### Divergence, Curl, and Laplacian

```python
diff = Difference2D(grid=grid)

divergence = diff.divergence(u, v)   # du/dx + dv/dy at T-points
curl       = diff.curl(u, v)         # dv/dx - du/dy at X-points
laplacian  = diff.laplacian(h)       # d²h/dx² + d²h/dy² at T-points
```

**Discrete-non-divergence property**: A velocity derived from a
corner-streamfunction `ψ` via
`u = -diff_y_X_to_U(ψ)`, `v = diff_x_X_to_V(ψ)`
is **exactly** non-divergent at interior T-points.

---

## Interpolation Operators

Each interpolation moves a field between co-location points by **averaging
the two nearest neighbours** along the relevant axis.

### Averaging table

| Method      | Formula                                                  | Source → Target |
|-------------|----------------------------------------------------------|-----------------|
| `T_to_U`    | `h[j, i+½] = ½(h[j,i] + h[j,i+1])`                     | T → U           |
| `T_to_V`    | `h[j+½, i] = ½(h[j,i] + h[j+1,i])`                     | T → V           |
| `T_to_X`    | `h[j+½, i+½] = ¼(h[j,i]+h[j,i+1]+h[j+1,i]+h[j+1,i+1])`| T → X           |
| `U_to_T`    | `u[j, i] = ½(u[j,i+½] + u[j,i-½]) = ½(u[j,i]+u[j,i-1])`| U → T           |
| `V_to_T`    | `v[j, i] = ½(v[j+½,i] + v[j-½,i]) = ½(v[j,i]+v[j-1,i])`| V → T           |
| `X_to_T`    | 4-point average over `q[j,i], q[j-1,i], q[j,i-1], q[j-1,i-1]` | X → T   |
| `U_to_X`    | `u[j+½,i+½] = ½(u[j,i] + u[j+1,i])`                    | U → X           |
| `V_to_X`    | `v[j+½,i+½] = ½(v[j,i] + v[j,i+1])`                    | V → X           |
| `X_to_U`    | `q[j,i+½] = ½(q[j,i] + q[j-1,i])`                      | X → U           |
| `X_to_V`    | `q[j+½,i] = ½(q[j,i] + q[j,i-1])`                      | X → V           |
| `U_to_V`    | 4-point average over `u[j,i], u[j+1,i], u[j,i-1], u[j+1,i-1]` | U → V   |
| `V_to_U`    | 4-point average over `v[j,i], v[j-1,i], v[j,i+1], v[j-1,i+1]` | V → U   |

**Example slice** for `T_to_U`:

```python
# h_on_u[j, i+1/2] = 0.5 * (h[j, i] + h[j, i+1])
out = out.at[1:-1, 1:-1].set(0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:]))
```

The key: `h[1:-1, 1:-1]` is `h[j, i]` and `h[1:-1, 2:]` is `h[j, i+1]`
for interior column indices `i = 1 … Nx-2`.

---

## Kinetic Energy and Bernoulli Potential

These operators live in `finitevolx._src.operators.diagnostics` and
follow the same conventions.

### Kinetic energy at T-points

```
ke[j, i] = ½ (u²_on_T[j, i] + v²_on_T[j, i])
```

where the face-squared fields are averaged to cell centres:

```
u²_on_T[j, i] = ½ (u[j, i+½]² + u[j, i-½]²)
              = ½ (u[j, i]²   + u[j, i-1]²)

v²_on_T[j, i] = ½ (v[j+½, i]² + v[j-½, i]²)
              = ½ (v[j, i]²   + v[j-1, i]²)
```

```python
# ke[j, i] at T-points
u2 = u**2;  v2 = v**2
u2_on_T = 0.5 * (u2[1:-1, 1:-1] + u2[1:-1, :-2])   # U → T in x
v2_on_T = 0.5 * (v2[1:-1, 1:-1] + v2[:-2, 1:-1])   # V → T in y
out = out.at[1:-1, 1:-1].set(0.5 * (u2_on_T + v2_on_T))
```

### Bernoulli potential at T-points

```
p[j, i] = ke[j, i] + g · h[j, i]
```

Both `ke` and `h` are at T-points, so this is a simple elementwise sum
restricted to the interior:

```python
out = out.at[1:-1, 1:-1].set(ke[1:-1, 1:-1] + gravity * h[1:-1, 1:-1])
```

---

## Slicing Reference

The table below collects all the slice patterns used in stencil
implementations.  `Ny, Nx` denote the total array sizes.

| Pattern                   | Meaning (rows = j, cols = i)          | Shape         |
|---------------------------|---------------------------------------|---------------|
| `arr[1:-1, 1:-1]`         | interior at `(j, i)`, both directions | `(Ny-2, Nx-2)` |
| `arr[1:-1, 2:]`           | interior rows, one step **east**      | `(Ny-2, Nx-2)` |
| `arr[1:-1, :-2]`          | interior rows, one step **west**      | `(Ny-2, Nx-2)` |
| `arr[2:, 1:-1]`           | interior cols, one step **north**     | `(Ny-2, Nx-2)` |
| `arr[:-2, 1:-1]`          | interior cols, one step **south**     | `(Ny-2, Nx-2)` |
| `arr[2:, 2:]`             | one step north-east                   | `(Ny-2, Nx-2)` |
| `arr[:-2, :-2]`           | one step south-west                   | `(Ny-2, Nx-2)` |

Notice that every shifted slice has the **same shape** as the unshifted
interior slice.  This is possible because arrays have one ghost-cell ring
on each side, so a single-step shift never reads outside the array bounds.

---

## Ghost-Cell Interaction at Stencil Boundaries

The first and last interior points see one ghost neighbour.  For example,
`diff_x_T_to_U` at the first interior U-column (`i=1`, output `out[j, 1]`)
uses `h[j, 1]` (interior T-point) and `h[j, 2]` (second interior
T-point) — no ghosts involved.

However, `diff_x_U_to_T` at `i=1` uses `u[j, 1]` and `u[j, 0]`.  If
`u[j, 0]` is a ghost (zero by default), the result at `i=1` reflects that
zero ghost rather than a physical extrapolation.  This is intentional: the
**caller** sets ghost values via the boundary-condition layer before calling
operators.

A consequence: the composition
`diff_x_U_to_T(diff_x_T_to_U(h))` equals `d²h/dx²` only at columns
`i = 2 … Nx-3` (deep interior), because the intermediate `dh_u` field has
a zero ghost at `i = 0` that pollutes column `i = 1`.

### Concrete example

For `h[j, i] = c·i·dx` and no boundary conditions applied:

```
dh_u = diff_x_T_to_U(h)
# dh_u[j, 0] = 0      ← ghost U-face, NOT written by the operator
# dh_u[j, 1] = c      ← first interior U-face
# dh_u[j, 2..Nx-2] = c ← rest of interior

result = diff_x_U_to_T(dh_u)
# result[j, 1] = (c - 0) / dx = c/dx   ← non-zero, ghost pollutes i=1
# result[j, 2..Nx-3] = (c - c) / dx = 0 ← correct 2nd derivative
```

This is **correct** operator behaviour. The ghost U-face at `i=0` is the
*west boundary face* `U[j, ½]`; its value must be supplied by the BC layer
(periodic, no-slip, etc.), not by the forward-difference stencil.  Always
apply boundary conditions to intermediate fields before chaining operators.

---

## Example: Advection Step

A typical shallow-water advection step uses all four co-location points:

```python
from finitevolx import (
    ArakawaCGrid2D, Difference2D, Interpolation2D, Vorticity2D,
    enforce_periodic,
)
import jax.numpy as jnp

grid  = ArakawaCGrid2D.from_interior(64, 64, 1.0, 1.0)
diff  = Difference2D(grid=grid)
interp = Interpolation2D(grid=grid)
vort  = Vorticity2D(grid=grid)

# ---- assume u, v, h are shape (66, 66) with updated ghosts ----

# 1. Potential vorticity at X-points
f     = jnp.zeros((grid.Ny, grid.Nx))           # Coriolis (T-points)
q     = vort.potential_vorticity(u, v, h, f)   # X-points

# 2. PV flux (Arakawa-Lamb)
qu, qv = vort.pv_flux_arakawa_lamb(q, u, v)   # U- and V-points

# 3. Kinetic energy and Bernoulli potential
from finitevolx._src.operators.diagnostics import bernoulli_potential
p = bernoulli_potential(h=h, u=u, v=v)         # T-points

# 4. Tendencies
du_dt = -diff.diff_x_T_to_U(p) + qv           # U-points
dv_dt = -diff.diff_y_T_to_V(p) - qu           # V-points
dh_dt = -diff.divergence(h_u, h_v)            # T-points (h*u fluxes)
```

Every intermediate array has the same shape `(66, 66)`.  The ghost ring
carries boundary-condition data; the interior `[1:-1, 1:-1]` carries the
physical solution.
