# Unified Boundary Conditions in finitevolX: Design Document

This document proposes a **unified boundary condition system** for finitevolX
that works across time-stepping, direct state application, and elliptic
solvers — with first-class support for **known values** at any cell (outer
boundaries, island coasts, and sparse interior observations).

Responds to [jejjohnson/finitevolX#186](https://github.com/jejjohnson/finitevolX/issues/186).

**Companion document:** [inhomogeneous_bcs_current.md](./inhomogeneous_bcs_current.md)
describes the current state and the manual lifting-trick workaround.

---

## 1. Goal

A unified boundary condition system for finitevolX that:

- **Works everywhere** — time-stepping (ghost-cell updates), direct state
  application, and elliptic solvers, all through the same BC object.
- **Carries domain geometry** — the `BoundaryConditionSet` owns a `Mask2D`,
  so solvers don't need a separate mask argument.
- **Supports known values at any cell** — outer boundaries, island coasts
  (auto-derived from mask topology), and sparse interior observations
  (user-supplied `known_mask`).
- **Is JIT-compilable and vmap-compatible**, including time-varying known
  values.
- **Causes zero breakage** for existing call sites — all new fields default
  to `None`.

---

## 2. Unified BC and BCSet Design

### 2.1 BC Atoms (unchanged)

The existing 9 atom types in `bc_1d.py` remain as-is.  They are face-level
building blocks with no mask awareness:

```python
# These stay exactly the same
Dirichlet1D(face="south", value=0.0)
Neumann1D(face="west", value=0.0)
Robin1D(face="north", alpha=1.0, beta=0.5, gamma=0.0)
Periodic1D(face="east")
# ... plus Outflow1D, Reflective1D, Sponge1D, Extrapolation1D, Slip1D
```

No mask on individual atoms — they describe *what kind* of condition applies
on a face, not *where* the domain is.

### 2.2 BoundaryConditionSet (evolved)

The existing `BoundaryConditionSet` (`bc_set.py`) gains three new optional
fields:

```python
class BoundaryConditionSet(eqx.Module):
    # --- Existing fields (unchanged) ---
    south: BoundaryCondition1D | None = None
    north: BoundaryCondition1D | None = None
    west: BoundaryCondition1D | None = None
    east: BoundaryCondition1D | None = None

    # --- NEW fields ---
    mask: Mask2D | Float[Array, "Ny Nx"] | None = None
    known_values: Float[Array, "Ny Nx"] | None = None
    known_mask: Bool[Array, "Ny Nx"] | None = None
```

All new fields default to `None`, so every existing call site continues to
work unchanged.

#### Mask ownership

- The BCSet owns a `Mask2D` (or raw float array).
- When a `Mask2D` is provided, each atom can extract the correct staggering
  internally (h/u/v/q) — the atom knows its face, the `Mask2D` knows the
  geometry.
- Solver operators read the mask from the BCSet — the user doesn't pass it
  separately.
- `mask=None` for simple rectangular domains (backward compatible).

#### Known values

- `known_values`: `(Ny, Nx)` array of prescribed values.  Only cells
  identified as "known" are read; values at other cells are ignored.
- `known_mask`: `(Ny, Nx)` boolean array marking **explicit** known-value
  locations (e.g., sparse observations at interior wet cells).
- Both are optional.  When absent, behavior is identical to today.

#### Auto-derive + merge

The set of "known cells" is computed by combining two sources:

```
# 1. Auto-derive boundary cells from mask topology
#    (outer boundary ring + island coast rings)
boundary_cells = _dilate(wet_mask) & ~wet_mask

# 2. Merge with explicit known_mask (sparse obs, etc.)
if known_mask is not None:
    all_known = boundary_cells | known_mask
else:
    all_known = boundary_cells
```

This handles three scenarios transparently:

```
Scenario 1: Simple basin         Scenario 2: Basin + island      Scenario 3: + sparse observations

0 0 0 0 0 0 0 0                  0 0 0 0 0 0 0 0                 0 0 0 0 0 0 0 0
0 . . . . . . 0                  0 . . . . . . 0                 0 . . . . . . 0
0 . . . . . . 0                  0 . . 0 0 . . 0                 0 . . . . . . 0
0 . . . . . . 0                  0 . . 0 0 . . 0                 0 . X . . X . 0
0 . . . . . . 0                  0 . . . . . . 0                 0 . . . . . . 0
0 . . . . . . 0                  0 . . . . . . 0                 0 . . . X . . 0
0 0 0 0 0 0 0 0                  0 0 0 0 0 0 0 0                 0 0 0 0 0 0 0 0

0 = land (dry)                   0 = land (dry + island)          X = observation (known interior)
. = ocean (wet, solve here)      . = ocean (wet, solve here)      . = ocean (wet, solve here)

known_mask: not needed           known_mask: not needed           known_mask: marks X cells
Auto-derive finds outer ring     Auto-derive finds outer +        Auto-derive + merge with
                                 island rings                     user-supplied known_mask
```

In all three cases, the solver:
1. Identifies all known cells (boundary ring + explicit `known_mask`)
2. Places `known_values` at those cells (the "lift")
3. Shrinks the solve domain: `effective_solve_mask = wet & ~all_known`
4. Corrects the RHS and solves the homogeneous problem on the reduced domain

#### Unified interface

```python
bc = BoundaryConditionSet(
    mask=cgrid_mask,
    south=Dirichlet1D("south", value=0.0),
    north=Dirichlet1D("north", value=0.1),
    west=Neumann1D("west", value=0.0),
    east=Dirichlet1D("east", value=0.05),
    known_values=obs_field,       # optional: (Ny, Nx) prescribed values
    known_mask=obs_locations,     # optional: (Ny, Nx) where obs apply
)

# Time-stepping: apply ghost cells as before
state = bc(state, dx, dy)

# Elliptic solver: mask + known values extracted from bc
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc=bc)
```

#### Factory methods (extended)

```python
# Existing factories still work
BoundaryConditionSet.periodic()
BoundaryConditionSet.open()

# New: closed basin with mask
BoundaryConditionSet.closed(mask=cgrid_mask)
# → Dirichlet(value=0) on all four faces, mask attached
```

### 2.3 FieldBCSet (minimal change)

No structural change.  It dispatches per-field `BoundaryConditionSet` as
before.  Each BCSet in the map can now carry its own mask + known values:

```python
field_bcs = FieldBCSet(
    bc_map={
        "psi": BoundaryConditionSet(mask=mask, south=..., known_values=obs_psi),
        "eta": BoundaryConditionSet(mask=mask, south=..., known_values=obs_eta),
    },
)
```

---

## 3. Design Principles

1. **Unified vocabulary** — one BC object for time-stepping and solvers.
   Users write `BoundaryConditionSet(...)` once and use it everywhere.

2. **BCSet owns the mask** — solvers extract it from the BC object; users
   don't pass the mask separately.

3. **known_values generalizes boundary data** — outer boundary ring, island
   coasts, and sparse interior observations all flow through the same
   mechanism.

4. **Auto-derive + merge** — boundary cells are computed from mask topology
   (JAX-native dilation); merged with any explicit `known_mask`.

5. **The lifting trick is the implementation, not the API** — users never
   write `binary_dilation` or manually compute corrected RHS.

6. **JAX-native throughout** — no SciPy in the hot path.  The dilation uses
   `jnp.roll` + boolean OR (~4 rolls).

7. **Additive, not breaking** — all new fields default to `None`.  Every
   existing call site continues to work unchanged.

---

## 4. Solver Integration: Three Layers

### Layer 1 — Pure Function: `apply_known_values`

The foundation.  A composable pure function that performs the generalized
lifting trick.  Accepts all parameters explicitly (maximum composability for
power users who want to bring their own solver).

```python
from finitevolx import apply_known_values

rhs_corrected, value_lift = apply_known_values(
    rhs=rhs,
    known_values=obs_field,     # (Ny, Nx) prescribed values
    mask=mask,                  # (Ny, Nx) wet/dry mask
    dx=dx, dy=dy,
    lambda_=0.0,
    known_mask=obs_locations,   # optional: explicit obs cells
)

# Solve with ANY solver
psi_hom = my_solver(rhs_corrected)

# Reconstruct
psi = value_lift + psi_hom
```

**What it does internally:**

```
1. Auto-derive boundary ring from mask (JAX-native dilation):
   wet = mask > 0.5
   dilated = wet | roll(wet, +1, 0) | roll(wet, -1, 0)
                  | roll(wet, +1, 1) | roll(wet, -1, 1)
   boundary_cells = dilated & ~wet

2. Merge with explicit known_mask:
   all_known = boundary_cells | known_mask   (if known_mask given)
   all_known = boundary_cells                (otherwise)

3. Build lift:
   value_lift = jnp.where(all_known, known_values, 0.0)

4. Compute effective solve mask:
   effective_solve_mask = wet & ~all_known

5. Correct RHS:
   A_lift = masked_laplacian(value_lift, effective_solve_mask, dx, dy, lambda_)
   rhs_corrected = rhs - A_lift

6. Return (rhs_corrected, value_lift)
```

**Signature:**

```python
def apply_known_values(
    rhs: Float[Array, "Ny Nx"],
    known_values: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"] | Mask2D,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    known_mask: Bool[Array, "Ny Nx"] | None = None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Correct RHS for known values via the generalized lifting trick.

    Parameters
    ----------
    rhs : (Ny, Nx) array
        Original right-hand side.
    known_values : (Ny, Nx) array
        Prescribed values.  Only cells identified as "known" (boundary
        ring from mask topology + explicit known_mask) are used.
    mask : (Ny, Nx) array or Mask2D
        Binary wet/dry mask (1=wet, 0=dry).
    dx, dy : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter.
    known_mask : (Ny, Nx) bool array or None
        Explicit known-value locations (e.g., sparse observations at
        interior wet cells).  Merged with auto-derived boundary ring.

    Returns
    -------
    rhs_corrected : (Ny, Nx) array
        Corrected RHS for the reduced homogeneous problem.
    value_lift : (Ny, Nx) array
        Lifting function — add to homogeneous solution to get full solution.
    """
```

**Proposed location:** `finitevolx/_src/solvers/inhomogeneous.py` (new file),
re-exported from `finitevolx/__init__.py`.

---

### Layer 2 — Convenience Wrappers: `known_values` / `known_mask` kwargs

Adds optional parameters to the existing `streamfunction_from_vorticity`,
`pressure_from_divergence`, and `pv_inversion` wrappers.

```python
# Current API (unchanged, homogeneous only)
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")

# New: pass known values directly
psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy,
    bc="dst",
    known_values=g,           # NEW
    known_mask=obs_mask,      # NEW, optional
    mask=mask,
)

# Pressure
p = fvx.pressure_from_divergence(
    div_u, dx, dy,
    bc="dct",
    known_values=p_boundary,
    mask=mask,
)

# PV inversion (per-layer broadcast)
psi = fvx.pv_inversion(
    pv, dx, dy,
    lambda_=lambdas,
    known_values=psi_boundary,
    mask=mask,
)
```

**Internal flow when `known_values` is not None:**

```
1. apply_known_values(rhs, known_values, mask, dx, dy, lambda_, known_mask)
      → (rhs_corrected, value_lift)
2. _solve_dispatch(rhs_corrected, ...) on effective_solve_mask
      → psi_hom
3. Return value_lift + psi_hom
```

**When `known_values is None`:** existing fast path.  Zero behavior change.

**Multi-layer PV inversion:** the lift depends on `lambda_` (via
`masked_laplacian`), so each layer gets its own corrected RHS.  The
implementation vmaps `apply_known_values` alongside the solve, matching
the existing vmap pattern in `elliptic.py:437-542`.

- `known_values` shape `(Ny, Nx)` → broadcast to all layers.
- `known_values` shape `(nl, Ny, Nx)` → per-layer known values.

**Spectral path on rectangular domains:** when the user passes
`known_values` with `method="spectral"` and no mask, synthesise a trivial
all-wet mask whose dry-boundary ring is the outer ghost frame.

---

### Layer 3 — BCSet Passthrough (unified)

The primary user-facing API.  Pass a `BoundaryConditionSet` directly to
any solver — the mask, known values, and face BCs are all read from it.

```python
from finitevolx import BoundaryConditionSet, Dirichlet1D

bc = BoundaryConditionSet(
    mask=cgrid_mask,
    south=Dirichlet1D("south", value=0.0),
    north=Dirichlet1D("north", value=0.1),
    west=Dirichlet1D("west", value=0.05),
    east=Dirichlet1D("east", value=0.05),
    known_values=obs_field,       # optional
    known_mask=obs_locations,     # optional
)

# One object drives everything:
state = bc(state, dx, dy)                                     # time-stepping
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc=bc)  # solver
p = fvx.pressure_from_divergence(div_u, dx, dy, bc=bc)        # solver
```

**Internal conversion when `bc` is a `BoundaryConditionSet`:**

1. Extract `mask` from `bc.mask`.
2. If the BCSet has per-face `Dirichlet1D` atoms with non-zero values,
   expand them into a `(Ny, Nx)` array (values at the ghost ring, zero in
   the interior).
3. Merge with `bc.known_values` / `bc.known_mask`.
4. Determine the BC type for the spectral dispatch:
   - All faces `Dirichlet1D` → `"dst"`
   - All faces `Neumann1D` → `"dct"`
   - All faces `Periodic1D` → `"fft"`
5. Call Layer 1 (`apply_known_values`) with the merged data.

**Fast-path detection:** if all Dirichlet values are zero, no `known_values`,
and no `known_mask` → take the existing spectral fast path (no mask needed).

**Why this matters:** users maintain one mental model.  The same
`BoundaryConditionSet` they write for time-stepping drives the elliptic
solve.  No more `Dirichlet1D(value=g)` for one system and `bc="dst"` for the
other.

---

## 5. Considered Alternatives

### Option B from issue #186: `InhomogeneousBCSolver` wrapper class

```python
solver = InhomogeneousBCSolver(base_solver=..., mask=mask, dx=dx, dy=dy)
psi = solver(rhs, known_values=g)
```

**Why not:** a class adds state without adding power.  The layered function +
BCSet passthrough covers every code path with less surface area and better
composability.

### Option D from issue #186: new `DirichletBC` / `NeumannBC` types

```python
bc = DirichletBC(values=g, mask=mask)
```

**Why not:** invents a new BC type parallel to the existing `Dirichlet1D`.
The revised design evolves the existing `BoundaryConditionSet` instead of
creating a parallel hierarchy.

---

## 6. Edge Cases and Decisions

### Spectral solvers without a mask

When `method="spectral"` and no mask is provided, but `known_values` is
given: synthesise a rectangular mask (all-wet interior, dry outer ring).
Apply the lifting trick normally.

### Capacitance solver

Operator-agnostic.  The capacitance solver sees only the corrected RHS and
solves the homogeneous problem as usual.  No changes needed.

### Multigrid

Same: only the RHS changes.  The V-cycle hierarchy is unaware of the lift.

### Time-varying known values

Because Layer 1 is a pure function, the correction is recomputed at every
call inside a JIT-compiled timestep.  No caching or stale-state issues.

### Sparse observations as known values

When `known_mask` marks interior wet cells, those cells are removed from the
solve domain (`effective_solve_mask = wet & ~all_known`).  Their prescribed
values feed into the stencil of neighboring "unknown" cells via the lift.
Mathematically, this treats observations as additional Dirichlet constraints
in the interior of the domain.

### Multi-layer PV inversion

The lift depends on `lambda_` (via `masked_laplacian`), so each layer gets
its own `rhs_corrected` and `value_lift`.  `known_values` can be `(Ny, Nx)`
(broadcast) or `(nl, Ny, Nx)` (per-layer).

### Sign of lambda

Preserved by the existing `masked_laplacian` definition: it computes
$(∇^2 - \lambda)\psi$, so both the RHS and the correction have consistent
sign.

### Homogeneous special case (all known values = 0)

The lift is zero, the RHS correction is zero, and the solver takes the fast
path.  No separate code branch needed.

---

## 7. Phasing

### PR 1: Evolve `BoundaryConditionSet` + `apply_known_values`

- Add `mask`, `known_values`, `known_mask` fields to `BoundaryConditionSet`
  (all default `None` — backward compatible).
- New file `finitevolx/_src/solvers/inhomogeneous.py` with
  `apply_known_values` pure function + JAX-native dilation helper.
- Re-export from `finitevolx/__init__.py`.
- Tests:
  - Manufactured solution: $\psi_{\text{exact}} = \sin(\pi x)\sin(\pi y) +
    0.1\sin(2\pi y)$ with CG, capacitance, spectral.
  - Consistency: `known_values=0` matches current homogeneous API.
  - Island mask: verify auto-derive finds inner + outer boundary rings.
  - Sparse obs: verify `known_mask` pins interior cells correctly.

### PR 2: Thread through convenience wrappers + BCSet passthrough

- Add `known_values` / `known_mask` kwargs to
  `streamfunction_from_vorticity`, `pressure_from_divergence`, `pv_inversion`.
- Accept `BoundaryConditionSet` as `bc=` parameter (Layer 3).
- Multi-layer vmap integration for `pv_inversion`.
- Auto-synthesise rectangular mask for spectral path.
- Tests for all three wrappers with known values and BCSet passthrough.

### PR 3: Documentation + notebook migration

- Update `docs/notebooks/demo_solvers.py` Section 8 to use new API.
- Documentation in elliptic solver guide.
- Update `docs/boundary_conditions.md` with unified BCSet description.

---

## 8. Acceptance Criteria

From issue #186, revised for the broader scope:

- [ ] `BoundaryConditionSet` gains `mask`, `known_values`, `known_mask` fields
- [ ] `apply_known_values(rhs, known_values, mask, dx, dy, lambda_, known_mask)` implemented
- [ ] Convenience wrappers accept `known_values` / `known_mask` parameters
- [ ] `BoundaryConditionSet` accepted as `bc=` on all convenience wrappers
- [ ] Works with all solver methods (spectral, CG, capacitance, multigrid)
- [ ] Manufactured-solution test passes to solver tolerance
- [ ] Island-mask test: auto-derive finds inner + outer boundary rings
- [ ] Sparse-obs test: `known_mask` pins interior wet cells correctly
- [ ] JIT-compatible for time-stepping applications
- [ ] Multi-layer `pv_inversion` supports per-layer `known_values`
- [ ] JAX-native dilation (no SciPy dependency in the hot path)
- [ ] Documentation in elliptic solver guide
- [ ] Example in `demo_solvers` notebook updated to use new API

---

## 9. Future Extensions (Out of Scope)

| Extension | Integration point |
|-----------|-------------------|
| **Spatially-varying per-face Dirichlet** (1-D arrays on `Dirichlet1D`) | Layer 3 conversion: expand 1-D face arrays into `(Ny, Nx)` lift |
| **Inhomogeneous Neumann** ($\partial\psi/\partial n = g$) | `modify_rhs_2d` from spectraldiffx (spectral path); extended `masked_laplacian` (CG path) |
| **Robin / mixed BCs on the solver** | `MixedBCHelmholtzSolver2D` from spectraldiffx; new Robin-aware operator for CG |
| **Open-boundary conditions** (radiation, sponge) | Sponge1D / Robin1D already exist for time-stepping; solver integration requires absorbing-layer operator |
| **3-D support** | Lifting trick generalises to 3-D with a 6-point dilation; all spectral solvers have 3-D variants |
| **Weighted observations** (soft constraints) | Penalty-method variant of the lifting trick: add `weight * (psi - obs)` to the operator instead of hard pinning |

---

*See [inhomogeneous_bcs_current.md](./inhomogeneous_bcs_current.md) for the
current state and the manual lifting trick this design replaces.*
