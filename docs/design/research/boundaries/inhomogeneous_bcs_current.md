# Inhomogeneous Boundary Conditions in finitevolX: Current State

This document describes how finitevolX handles boundary conditions today,
highlights the gap between the time-stepping BC vocabulary and the elliptic
solver layer, and walks through the manual lifting-trick workaround for
non-zero Dirichlet data.

**Companion document:** [inhomogeneous_bcs_design.md](./inhomogeneous_bcs_design.md)
proposes a future API that closes this gap.

**GitHub issue:** [jejjohnson/finitevolX#186](https://github.com/jejjohnson/finitevolX/issues/186)

---

## 1. When Non-Zero Boundary Data Arises

| Use case | Boundary data source |
|----------|---------------------|
| **Regional nesting** | Parent model provides SSH/streamfunction at open boundaries |
| **Data assimilation** | Tide gauges or satellite altimetry fix boundary values |
| **Reanalysis-forced runs** | ERA5 or ORAS5 SSH prescribed at lateral boundaries |
| **Two-way coupling** | Outer model provides BCs for an inner high-resolution domain |
| **Manufactured solutions** | Testing with non-trivial exact solutions that don't vanish at boundaries |

All of these require solving $(A - \lambda)\psi = f$ with $\psi = g$ on the
boundary, where $g \neq 0$.

---

## 2. What finitevolX Already Has: The Ghost-Cell BC System

The boundary module (`finitevolx/_src/boundary/`) provides a rich, composable
set of BC objects built on the ghost-cell paradigm.  All arrays have shape
`(Ny, Nx)` with an outermost one-cell ghost ring; the physical boundary lies
between the first interior cell and the ghost cell.

### 2.1 Single-Face BC Atoms (`bc_1d.py`)

Nine BC types, all `equinox.Module` subclasses with a common signature:

```python
bc = SomeBCType(face="south", ...)
field_updated = bc(field, dx=dx, dy=dy)  # returns field with one ghost face set
```

| BC Type | Class | Ghost-cell formula | Typical use |
|---------|-------|--------------------|-------------|
| Dirichlet | `Dirichlet1D` | `ghost = 2*value - interior` | Fixed wall value ($\psi = 0$, $u = 0$) |
| Neumann | `Neumann1D` | `ghost = interior + sign*value*spacing` | Zero normal gradient (free-slip, no-flux) |
| Periodic | `Periodic1D` | `ghost = opposite_interior` | Doubly-periodic channels |
| Outflow | `Outflow1D` | `ghost = interior` | Open boundary (zero gradient) |
| Reflective | `Reflective1D` | `ghost = interior` | Even symmetry (Neumann with $g = 0$) |
| Sponge | `Sponge1D` | `ghost = (1-w)*interior + w*background` | Wave absorption |
| Robin | `Robin1D` | $\alpha u + \beta \partial u/\partial n = \gamma$ | Mixed BC |
| Extrapolation | `Extrapolation1D` | Lagrange polynomial (orders 1--5) | Smooth open boundaries |
| Slip | `Slip1D` | `ghost = (2a-1)*interior` | Partial-slip walls ($a \in [0,1]$) |

**Key point:** `Dirichlet1D` already accepts a non-zero `value=g`:

```python
from finitevolx import Dirichlet1D

# Non-zero Dirichlet: ψ = 0.5 on the south wall
bc_south = Dirichlet1D(face="south", value=0.5)
psi = bc_south(psi, dx=dx, dy=dy)
# ghost[0, :] = 2*0.5 - psi[1, :] = 1.0 - psi[1, :]
```

### 2.2 Per-Face Containers (`bc_set.py`)

`BoundaryConditionSet` composes one atom per face (south, north, west, east):

```python
from finitevolx import BoundaryConditionSet, Dirichlet1D, Periodic1D

bc_set = BoundaryConditionSet(
    south=Dirichlet1D(face="south", value=0.0),
    north=Dirichlet1D(face="north", value=0.0),
    west=Periodic1D(face="west"),
    east=Periodic1D(face="east"),
)
psi = bc_set(psi, dx=dx, dy=dy)  # all four faces updated
```

Application order: south -> north -> west -> east (west/east overwrite corners).

Factory methods for common patterns:

```python
BoundaryConditionSet.periodic()  # all four faces periodic
BoundaryConditionSet.open()      # all four faces zero-gradient outflow
```

### 2.3 Multi-Field Dispatch (`bc_field.py`)

`FieldBCSet` applies different BC sets to different state-dictionary entries:

```python
from finitevolx import FieldBCSet

field_bcs = FieldBCSet(
    bc_map={"psi": bc_closed_basin, "eta": bc_open_boundary},
    default=BoundaryConditionSet.periodic(),
)
state_out = field_bcs(state_dict, dx=dx, dy=dy)
```

### 2.4 Functional Helpers (`boundary.py`)

Quick one-liners for common homogeneous patterns:

| Function | Effect |
|----------|--------|
| `zero_boundaries(field)` | Homogeneous Dirichlet ($\psi = 0$) on all faces |
| `zero_gradient_boundaries(field)` | Homogeneous Neumann ($\partial\psi/\partial n = 0$) |
| `no_flux_boundaries(field)` | Alias for `zero_boundaries` (normal flux = 0) |
| `no_slip_boundaries(field)` | Tangential velocity = 0 (ghost = $-$interior) |
| `free_slip_boundaries(field)` | Zero normal gradient (ghost = interior) |
| `enforce_periodic(field)` | Periodic wrapping in x and/or y |
| `wall_boundaries(field, stag)` | C-grid composite: picks correct BC per staggering (`h`/`u`/`v`/`q`) |

---

## 3. What the Elliptic Solvers Accept

The three convenience wrappers live in `finitevolx/_src/solvers/elliptic.py`:

```python
# Streamfunction from vorticity: ∇²ψ − λψ = ζ
psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy,
    bc="dst",              # ← a STRING, not a BC object
    lambda_=0.0,
    method="spectral",     # or "cg" or "capacitance"
    mask=None,
    capacitance_solver=None,
    preconditioner=None,
)

# Pressure from divergence: ∇²p = ∇·u
p = fvx.pressure_from_divergence(div_u, dx, dy, bc="dct")

# PV inversion: (∇² − λ)ψ = q  (multi-layer / batched)
psi = fvx.pv_inversion(pv, dx, dy, lambda_=lam, bc="dst")
```

### The `bc=` parameter is a transform selector, not a BC object

| `bc` value | Transform | Implicit boundary condition |
|------------|-----------|----------------------------|
| `"dst"` | Discrete Sine Transform | Homogeneous Dirichlet ($\psi = 0$) |
| `"dct"` | Discrete Cosine Transform | Homogeneous Neumann ($\partial\psi/\partial n = 0$) |
| `"fft"` | Fast Fourier Transform | Periodic |

All three call the same internal dispatcher (`_solve_dispatch`, line 253),
which routes to one of:

- **Spectral** (`_solve_spectral`): calls `spectraldiffx`'s `solve_helmholtz_dst/dct/fft`
- **CG** (`_solve_cg_method`): uses `masked_laplacian` as the `matvec` operator
- **Capacitance** (`_solve_capacitance_method`): calls a pre-built `CapacitanceSolver`

Plus the **multigrid** path, accessible through `build_multigrid_solver` ->
`make_multigrid_preconditioner` -> CG.

### All four solver methods enforce homogeneous BCs

**Masked Laplacian** (`iterative.py:144-191`):

```python
def masked_laplacian(psi, mask, dx, dy, lambda_=0.0):
    psi_m = psi * mask_arr          # ← zero outside domain
    lap = (
        jnp.roll(psi_m, 1, axis=1) + jnp.roll(psi_m, -1, axis=1) - 2.0 * psi_m
    ) / dx**2 + (
        jnp.roll(psi_m, 1, axis=0) + jnp.roll(psi_m, -1, axis=0) - 2.0 * psi_m
    ) / dy**2
    return (lap - lambda_ * psi_m) * mask_arr  # ← zero outside domain
```

The two `* mask_arr` operations hard-wire $\psi = 0$ at the mask boundary.
There is no parameter to inject non-zero boundary values.

**Capacitance solver:** Enforces $\psi = 0$ at all inner-boundary points
(ocean cells adjacent to land) through the Sherman-Morrison correction.

**Multigrid operator** (`multigrid.py:327-410`): Zeroes the face coefficients
`cx`/`cy` at the mask edge, baking homogeneous Dirichlet into the precomputed
level hierarchy.

### Unsurfaced prior art: `modify_rhs_*` from spectraldiffx

`spectraldiffx` already ships `modify_rhs_1d`, `modify_rhs_2d`, and
`modify_rhs_3d` for encoding inhomogeneous BC values into the RHS of a
spectral solve.  finitevolX re-exports them (`spectral.py:48-50`) but **none
of the convenience wrappers surface them** -- they sit unused in the public API.

---

## 4. The Gap

The BC vocabulary is **rich on one side** (9 atom types, per-face composition,
multi-field dispatch, non-zero `value=` support) and **a single string on the
other** (the `bc="dst"` parameter on the elliptic wrappers).

```
Time-stepping BCs              Elliptic solver BCs
──────────────────              ──────────────────
Dirichlet1D(value=g)           bc="dst" → ψ=0
Neumann1D(value=g)             bc="dct" → ∂ψ/∂n=0
Robin1D(α, β, γ)               bc="fft" → periodic
Sponge1D(bg, weight)           (nothing)
Slip1D(coefficient)            (nothing)
BoundaryConditionSet           (nothing)
FieldBCSet                     (nothing)
```

Non-zero boundary data has nowhere to go.  Users who need $\psi = g \neq 0$
must hand-roll the **lifting trick**.

---

## 5. Worked Example: The Lifting Trick Today

This is the workflow from `docs/notebooks/demo_solvers.py`, Section 8
(lines 944--1087).  It solves the Helmholtz equation on a basin domain with
prescribed non-zero Dirichlet data: $\psi = g$ on the boundary, where
$g(y) = 0.1 \sin(2\pi y / L_y)$.

### The Decomposition

$$
\psi = \psi_{\text{lift}} + \psi_{\text{hom}}
$$

where $\psi_{\text{lift}}$ matches the prescribed boundary data $g$, and
$\psi_{\text{hom}}$ solves the **corrected** equation with **zero** BCs:

$$
(A - \lambda)\,\psi_{\text{hom}} = f - (A - \lambda)\,\psi_{\text{lift}},
\qquad \psi_{\text{hom}} = 0 \;\text{on boundary}
$$

### Step 1: Identify the boundary cells

The "boundary cells" for the lift are the **dry cells adjacent to at least one
wet cell** -- the 1-cell-wide land ring around the ocean.

```python
from scipy.ndimage import binary_dilation

wet = mask_basin.astype(bool)
wet_dilated = binary_dilation(wet)
dry_boundary = wet_dilated & ~wet
```

### Step 2: Build the lifting function

Place the prescribed values at dry boundary cells, zero elsewhere:

```python
import numpy as np

g_field = 0.1 * np.sin(2 * np.pi * Y / Ny)

psi_lift = np.zeros((Ny, Nx))
psi_lift[dry_boundary] = g_field[dry_boundary]
psi_lift_jnp = jnp.array(psi_lift)
```

### Step 3: Correct the RHS

Subtract the operator applied to the lift at wet cells:

```python
A_psi_lift = fvx.masked_laplacian(
    psi_lift_jnp, mask_basin_jnp, dx, dy, lambda_=lambda_
)
rhs_corrected = rhs_basin - A_psi_lift
```

### Step 4: Solve the homogeneous problem

Any solver works here -- the lift only changed the RHS:

```python
sol_cg_inhom, info_inhom = fvx.solve_cg(
    A_basin, rhs_corrected,
    preconditioner=pc_basin,
    rtol=1e-10, atol=1e-10,
)
sol_cg_inhom = sol_cg_inhom * mask_basin_jnp  # zero outside
```

### Step 5: Reconstruct the full solution

```python
psi_full_inhom = psi_lift_jnp + sol_cg_inhom
```

At wet cells: $\psi = 0 + \psi_{\text{hom}}$ (the solver result).
At dry boundary cells: $\psi = g + 0$ (the prescribed data).

### Summary table

| Step | Operation | Code |
|------|-----------|------|
| 1 | Find boundary ring | `dry_boundary = binary_dilation(wet) & ~wet` |
| 2 | Build $\psi_{\text{lift}}$ | `psi_lift[dry_boundary] = g[dry_boundary]` |
| 3 | Correct RHS | `rhs' = rhs - masked_laplacian(psi_lift, ...)` |
| 4 | Solve homogeneous | `psi_hom = solver(rhs')` |
| 5 | Reconstruct | `psi = psi_lift + psi_hom` |

---

## 6. Pain Points With the Manual Approach

1. **Users must know which cells are "boundary".**  The lift lives at *dry
   cells adjacent to wet*, not the coast ring itself.  Getting this wrong
   silently produces garbage.

2. **The dilation requires SciPy.**  `scipy.ndimage.binary_dilation` is a
   NumPy operation that cannot be JIT-compiled.  In a time-stepping loop
   where $g$ changes every step, the mask work must be hoisted outside the
   JIT boundary -- which is non-obvious.

3. **The masked Laplacian must be reapplied to the lift.**  Step 3 is easy
   to forget or get wrong (e.g. applying the bare Laplacian instead of the
   Helmholtz operator, or forgetting the mask).

4. **Reconstruction is a separate step.**  Step 5 is trivial but easy to
   forget, especially in a multi-layer `pv_inversion` loop where each layer
   needs its own lift.

5. **No spectral-path equivalent.**  The lifting trick as written above
   works only with CG/capacitance/multigrid (all mask-based).  For the
   spectral path on a rectangular domain, the correct approach is to
   encode the inhomogeneous data via `modify_rhs_2d` from `spectraldiffx`
   -- but the convenience wrappers don't expose this, so users have to
   drop down to the raw spectral API.

6. **No connection to the BC vocabulary.**  A user who writes
   `Dirichlet1D(face="west", value=0.5)` for time-stepping has no way to
   feed that same information to `streamfunction_from_vorticity`.  The two
   systems are completely disconnected.

7. **Multi-layer PV inversion is particularly painful.**  `pv_inversion`
   accepts a `lambda_` array for per-layer Helmholtz parameters and handles
   the `vmap` internally -- but the user must replicate that batching logic
   externally to apply the lift per layer, defeating the purpose of the
   convenience wrapper.

---

## 7. Existing Infrastructure That a Future API Can Reuse

| Component | Location | Reuse potential |
|-----------|----------|----------------|
| `Dirichlet1D(value=g)` | `bc_1d.py:68-93` | Already carries non-zero values; can be expanded from per-face to per-cell |
| `BoundaryConditionSet` | `bc_set.py:11-66` | Natural container to pass 4-face Dirichlet data to a solver |
| `masked_laplacian` | `iterative.py:144-191` | Step 3 of the lift already uses this; no new operator needed |
| `_solve_dispatch` | `elliptic.py:253-273` | Central routing point; `known_values`/`known_mask` kwargs can be added here |
| `modify_rhs_2d` | `spectral.py:49` (re-export) | Spectral-path inhomogeneous BCs; already available but unused |
| `_inner_boundary_indices` | `tests/test_solver_wrappers.py:47-54` | SciPy-based helper for finding boundary ring; candidate for JAX-native port |
| `pv_inversion` vmap logic | `elliptic.py:437-542` | Multi-layer batching that the lift must integrate with |

---

*See [inhomogeneous_bcs_design.md](./inhomogeneous_bcs_design.md) for the
proposed API that addresses these pain points.*
