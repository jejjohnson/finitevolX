# Boundary Conditions

This page covers the theory behind the boundary condition (BC) system in
finitevolX, explains the ghost-cell approach used on the Arakawa C-grid, and
provides guidance for composing and applying BCs.

---

## The Problem

All spatial operators in finitevolX write only to interior cells `[1:-1, 1:-1]`
and read the one-cell ghost ring to compute stencils at the boundary.  The
physical boundary condition is therefore **encoded in the ghost cells**: the
BC layer sets ghost values such that the interior stencil evaluates to the
correct physical quantity at the domain wall.

This design cleanly separates the operator code (which never checks for
boundaries) from the boundary logic (which only writes to ghost cells).  It
also allows arbitrary BCs to be applied by simply overwriting ghost cells
before the next operator call.

---

## The Ghost-Cell Method

For a field with interior values $\phi_1, \phi_2, \ldots$ and ghost cell
$\phi_0$, the physical boundary is modelled as lying exactly **between** the
first interior cell and the ghost cell.  This gives the standard ghost-cell
relationships:

**Dirichlet** ($\phi_{\text{wall}} = g$):

$$
\phi_{\text{wall}} = \tfrac{1}{2}(\phi_1 + \phi_0) = g
\implies \phi_0 = 2g - \phi_1
$$

**Neumann** ($\partial\phi/\partial n\big|_{\text{wall}} = g$, outward normal):

$$
\frac{\phi_0 - \phi_1}{\Delta n} = g
\implies \phi_0 = \phi_1 + g\,\Delta n
$$

where $\phi_0$ is the ghost cell and $\phi_1$ the adjacent interior cell. The
ghost cell lies outward of the interior on every face, so the same formula
holds on all four faces: $g$ is the true gradient along the outward normal
(e.g. $g = -\partial\phi/\partial y$ on the south wall).

**Periodic** ($\phi_0 = \phi_{N-1}$, $\phi_N = \phi_1$):

The ghost cell simply copies the opposite interior edge.

---

## Boundary Condition Types

### `Dirichlet1D`

Fixes the **value** of the field at the domain wall.  The ghost cell is set to
the reflection that makes the average equal to the target value:

$$
\phi_{\text{ghost}} = 2\,v_{\text{target}} - \phi_{\text{interior}}
$$

**Physical use cases:**
- No-slip walls for the tangential velocity: $v = 0$ on a west / east wall.
- No normal flow through a wall: $u = 0$ on a west / east wall, with
  `point="U"` (see below).
- Fixed surface elevation: $\eta = 0$ on rigid lid.
- Streamfunction: $\psi = 0$ on closed coastline (used by DST Poisson solver).

### Staggered fields: `point=`

The formula above places the wall half a cell outside the first interior
node, which is right for every **cell-centred** direction: `T` on all four
faces, `V` on the west / east faces, `U` on the south / north faces.  With
same-index staggering the other pairs have a node **on** the wall:
`U[:, 0]` and `U[:, Nx-2]` are the west and east wall faces, `V[0, :]` and
`V[Ny-2, :]` the south and north ones, and an `X` field has a node on all
four walls (`U[:, Nx-1]`, `V[Ny-1, :]` are outside the domain).

Every atom except `Periodic1D` takes a static `point="T" | "U" | "V" | "X"`
(default `"T"`).  For a node on the wall the condition is imposed on that
node: `Dirichlet1D` writes `value` to it, `Neumann1D` / `Robin1D` use the
one-sided gradient between it and the first interior node, `Slip1D` writes
`a * interior`, and the outside slot on the east / north side gets the odd
mirror $2\phi_{\text{wall}} - \phi_{\text{adjacent}}$.  `Periodic1D` needs no
`point`: wrapping by one period is right for every stagger (`U[:, 0]` is the
same face as `U[:, Nx-2]`).

!!! warning "Normal velocity needs `point=`"
    `Dirichlet1D("east", 0.0)` on a U-field sets the ghost column
    `U[:, Nx-1] = -U[:, Nx-2]` and leaves the east wall face `U[:, Nx-2]`
    untouched, so mass flows through a "closed" wall.  Use
    `Dirichlet1D("east", 0.0, point="U")`, or
    `BoundaryConditionSet.closed(point="U")` for the whole basin.

```python
import finitevolx as fvx

bc_u = fvx.BoundaryConditionSet.closed(point="U")  # u = 0 on W/E faces, no-slip on S/N
bc_v = fvx.BoundaryConditionSet.closed(point="V")  # v = 0 on S/N faces, no-slip on W/E
u, v = bc_u(u, dx, dy), bc_v(v, dx, dy)
# Difference2D(grid).divergence(u, v) now integrates to zero over the basin.
```

### `Neumann1D`

Fixes the **outward-normal gradient** $\partial\phi/\partial n = g$:

$$
\phi_{\text{ghost}} = \phi_{\text{interior}} + g\,\Delta n
$$

**Physical use cases:**
- Free-slip walls: $\partial u_{\parallel}/\partial n = 0$.
- No-flux tracer BC: $\partial T/\partial n = 0$ on insulating walls.
- Pressure BC: $\partial p/\partial n = 0$ on solid walls.

### `Periodic1D`

Wraps the field: the ghost cell on one face copies the last interior cell on
the opposite face.

**Physical use cases:**
- Doubly-periodic channels and basins.
- Zonal re-entrant channels (periodic in x, closed in y).

!!! note "enforce_periodic helper"
    For fully periodic domains the helper `enforce_periodic(field, ...)` is
    available as a thin wrapper that applies `Periodic1D` on all four faces
    simultaneously.  Prefer `BoundaryConditionSet` for per-face control.

### `Reflective1D`

Reflects the field about the boundary:

$$
\phi_{\text{ghost}} = \phi_{\text{interior}} \quad (\text{even reflection})
$$

This is Neumann with $g = 0$.  Used for **free-slip** walls where the
normal velocity is zero and the tangential velocity has zero normal gradient.

### `Slip1D`

General slip boundary condition controlled by a scalar `coefficient = a`:

$$
\phi_{\text{ghost}} = (2a - 1)\,\phi_{\text{interior}}
$$

where:

- `a = 1.0` (default) gives **free-slip** (even reflection),
- `a = 0.0` gives **no-slip** (odd reflection, equivalent to Dirichlet with $g=0$),
- `0 < a < 1` gives **partial-slip**, interpolating smoothly between free-slip and no-slip.

This BC is typically used for velocity components at solid walls, with the
choice of `coefficient` controlling how strongly the tangential velocity is
damped at the boundary.

### `Outflow1D`

Zero-gradient outflow condition (equivalent to Neumann with $g=0$):

$$
\phi_{\text{ghost}} = \phi_{\text{interior}}
$$

Used at **open boundaries** where the flow leaves the domain and the outward
gradient is assumed zero.

### `Extrapolation1D`

Linear extrapolation from the two innermost interior cells:

$$
\phi_{\text{ghost}} = 2\,\phi_1 - \phi_2
$$

More accurate than zero-gradient for smoothly varying fields at open
boundaries.  Avoids spurious reflections from the domain edge.

### `Robin1D`

Mixed (Robin / third-kind) condition: a linear combination of value and
gradient:

$$
a\,\phi_{\text{wall}} + b\,\frac{\partial\phi}{\partial n}\bigg|_{\text{wall}} = g
$$

With $\phi_{\text{wall}} = (\phi_{\text{ghost}} + \phi_{\text{interior}})/2$ and
$\partial\phi/\partial n = (\phi_{\text{ghost}} - \phi_{\text{interior}})/\Delta n$
(outward normal, same on every face), the ghost value is

$$
\phi_{\text{ghost}} = \frac{g - (a/2 - s)\,\phi_{\text{interior}}}{a/2 + s},
\qquad s = \frac{b}{\Delta n}
$$

Used for **sponge-layer** or **radiation** boundary conditions in more
advanced setups.

### `Sponge1D`

Relaxation towards a target profile $\phi^*$ within a sponge layer of
specified half-width:

$$
\phi_{\text{ghost}} = \phi_{\text{interior}} + \lambda(\phi^* - \phi_{\text{interior}})
$$

Used at **open boundaries** to absorb outgoing waves and prevent reflections.

---

## Composing Boundary Conditions

### `BoundaryConditionSet`

Applies one BC atom on each of the four faces of a 2-D field.  Faces are
processed in **south → north → west → east** order, so west/east BCs
overwrite corner ghost cells filled by south/north BCs:

```python
from finitevolx import BoundaryConditionSet, Dirichlet1D, Periodic1D

# Closed basin: Dirichlet ψ=0 on all walls
bc_set = BoundaryConditionSet(
    south=Dirichlet1D(face="south", value=0.0),
    north=Dirichlet1D(face="north", value=0.0),
    west=Dirichlet1D(face="west",  value=0.0),
    east=Dirichlet1D(face="east",  value=0.0),
)

psi = bc_set(psi, dx=grid.dx, dy=grid.dy)
```

The same set can describe the boundary of an **elliptic solve**: pass it as
`bc=` to `streamfunction_from_vorticity`, `pressure_from_divergence` or
`pv_inversion`.  All-`Dirichlet1D` faces select the DST, all zero
`Neumann1D` faces the DCT and all `Periodic1D` faces the FFT; a set's
`mask` is used by the mask-based methods (`method="spectral"` raises on a
`bc.mask`); and non-zero `Dirichlet1D` values are imposed on each face's
wall nodes:

```python
bc_set = BoundaryConditionSet(
    mask=mask,
    south=Dirichlet1D(face="south", value=0.0),
    north=Dirichlet1D(face="north", value=0.1),
    west=Dirichlet1D(face="west",  value=0.0),
    east=Dirichlet1D(face="east",  value=0.0),
)
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc=bc_set, method="cg")
```

When time-stepping, a `Dirichlet1D` value sits on the wall face, half a
cell outside the first wet T cell.  ψ lives at X points, and the X nodes
`0` and `N-2` lie **on** the walls, so a solve pins those wall nodes (the
coast ring of the ψ mask) to the face value: the same wall.  For the
T-point `pressure_from_divergence` the face is half a cell from any cell
centre, so non-zero `Dirichlet1D` faces raise there.  See
[Known Boundary Values](elliptic_solvers_usage.md#known-boundary-values-inhomogeneous-dirichlet)
for arbitrary per-cell values and interior observations.

### `FieldBCSet`

Maps a BC set to a named field in a state dictionary or pytree.  Useful
for model state vectors with multiple fields, each needing different BCs:

```python
from finitevolx import BoundaryConditionSet, Dirichlet1D, FieldBCSet, Neumann1D

# Free-slip velocity BCs: zero normal flow on the wall faces (point="U"),
# zero tangential gradient on the south / north walls
bc_u = BoundaryConditionSet(
    south=Neumann1D(face="south", value=0.0),
    north=Neumann1D(face="north", value=0.0),
    west=Dirichlet1D(face="west", value=0.0, point="U"),
    east=Dirichlet1D(face="east", value=0.0, point="U"),
)
bcs = FieldBCSet(bc_map={"u": bc_u, "h": BoundaryConditionSet.open()})
```

Each atom carries its own `point`, so `FieldBCSet` needs no stagger
information of its own.

---

## Helpers

### `pad_interior`

Pads a field array *from* an interior-only view *to* the full `[Ny, Nx]`
ghost-ring layout.  Useful when you have a field defined only on interior
points and need to apply ghost-cell BCs:

```python
from finitevolx import pad_interior

q_interior = jnp.ones((Ny - 2, Nx - 2))   # interior only
q_full     = pad_interior(q_interior, mode="constant", constant_values=0.0)
```

Supported modes match `jnp.pad`: `"constant"`, `"edge"`, `"reflect"`, `"wrap"`.

### `enforce_periodic`

Convenience wrapper: applies periodic BCs in x, y, or both directions:

```python
from finitevolx import enforce_periodic

q = enforce_periodic(q, x=True, y=True)   # both directions
```

---

## Decision Guide

```
What physical condition applies?
│
├── Fixed value at wall (ψ=0, T=T_wall, tangential velocity)
│   └── Dirichlet1D(face=..., value=...)
│
├── No normal flow through a wall (u on W/E, v on S/N)
│   └── Dirichlet1D(face=..., value=0, point="U" / "V")
│
├── Zero normal gradient (free-slip, insulating)
│   └── Reflective1D  or  Neumann1D(face=..., value=0)
│
├── Specified normal gradient (heat flux, wind stress)
│   └── Neumann1D(face=..., value=g)
│
├── No-slip (u=0, v=0 at wall)
│   └── Slip1D  or  Dirichlet1D(value=0)
│
├── Periodic domain
│   └── Periodic1D  or  enforce_periodic(field)
│
├── Open boundary, outflow
│   ├── Zero gradient   → Outflow1D  or  Reflective1D
│   └── Linear extrap   → Extrapolation1D
│
└── Open boundary with wave absorption
    └── Sponge1D  or  Robin1D
```

---

## References

- Haidvogel & Beckmann (1999) — *Numerical Ocean Circulation Modelling*, Ch. 4
- Griffies (2004) — *Fundamentals of Ocean Climate Models*, Ch. 3
- Durran (2010) — *Numerical Methods for Fluid Dynamics*, Ch. 8 (open BCs)
