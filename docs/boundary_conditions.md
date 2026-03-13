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
\frac{\phi_1 - \phi_0}{\Delta n} = \pm g
\implies \phi_0 = \phi_1 \mp g\,\Delta n
$$

where the sign depends on the face: $-$ for south/west (outward normal points
inward in index space), $+$ for north/east.

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
- No-slip walls: $u = 0$ on solid boundary.
- Fixed surface elevation: $\eta = 0$ on rigid lid.
- Streamfunction: $\psi = 0$ on closed coastline (used by DST Poisson solver).

### `Neumann1D`

Fixes the **outward-normal gradient** $\partial\phi/\partial n = g$:

$$
\phi_{\text{ghost}} = \phi_{\text{interior}} \mp g\,\Delta n
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

### `FieldBCSet`

Maps a BC set to a named field in a state dictionary or pytree.  Useful
for model state vectors with multiple fields, each needing different BCs:

```python
from finitevolx import FieldBCSet, BoundaryConditionSet, Neumann1D

# Free-slip velocity BCs
bc_u = BoundaryConditionSet(
    south=Neumann1D(face="south", value=0.0),
    north=Neumann1D(face="north", value=0.0),
    west=Dirichlet1D(face="west",  value=0.0),
    east=Dirichlet1D(face="east",  value=0.0),
)
```

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
├── Fixed value at wall (ψ=0, u=0, T=T_wall)
│   └── Dirichlet1D(face=..., value=...)
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
