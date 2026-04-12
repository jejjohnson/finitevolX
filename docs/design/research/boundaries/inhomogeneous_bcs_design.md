# Inhomogeneous Boundary Conditions in finitevolX: Future API Design

This document proposes a layered API for **inhomogeneous (non-zero) Dirichlet
boundary conditions** on finitevolX's elliptic solvers, responding to
[jejjohnson/finitevolX#186](https://github.com/jejjohnson/finitevolX/issues/186).

**Companion document:** [inhomogeneous_bcs_current.md](./inhomogeneous_bcs_current.md)
describes the current state and the manual lifting trick.

---

## 1. Goal

Support solving $(A - \lambda)\psi = f$ with $\psi = g$ on the boundary
($g \neq 0$), where:

- The solution is **JIT-compilable** and **vmap-compatible** (including
  time-varying $g$).
- It **composes with all four solver methods**: spectral, CG, capacitance,
  and multigrid.
- It causes **zero behavior change** for existing call sites that don't pass
  boundary data.
- It **bridges** the existing `Dirichlet1D` / `BoundaryConditionSet`
  vocabulary from the time-stepping BC layer to the elliptic solver layer.

---

## 2. Design Principles

1. **Reuse, don't reinvent.**  The `Dirichlet1D`, `BoundaryConditionSet`, and
   `masked_laplacian` APIs already exist.  The new code should call them,
   not duplicate their logic.

2. **The lifting trick is the implementation, not the API.**  Users should
   never write `binary_dilation` or manually compute the corrected RHS.
   That logic is encapsulated behind a single entry point.

3. **JAX-native throughout.**  Replace the SciPy `binary_dilation` from the
   demo notebook with a JAX-native one-cell dilation (XOR of mask shifts),
   so the entire path is JIT-friendly without a NumPy round-trip.

4. **One implementation, three layers of API.**  The lifting logic is written
   once (Layer 1, a pure function).  Layers 2 and 3 are thin wrappers that
   call Layer 1.

5. **Additive, not breaking.**  All new parameters have default `None`, so
   existing call sites remain unchanged.

---

## 3. Recommended Design: Three Layers

### Layer 1 -- Pure Function: `apply_inhomogeneous_bc`

The foundation.  A composable pure function that performs the lifting trick
and returns the corrected RHS + lift array.

```python
from finitevolx import apply_inhomogeneous_bc

rhs_corrected, psi_lift = apply_inhomogeneous_bc(
    rhs=rhs,
    bc_values=g,          # (Ny, Nx) array; only dry-boundary values matter
    mask=mask,            # (Ny, Nx) binary mask (1=wet, 0=dry)
    dx=dx, dy=dy,
    lambda_=0.0,
)

# Solve with ANY solver (spectral, CG, capacitance, multigrid)
psi_hom = my_solver(rhs_corrected)

# Reconstruct
psi = psi_lift + psi_hom
```

**What it does internally:**

```
1. Compute dry boundary ring (JAX-native dilation):
   wet = mask > 0.5
   dilated = wet | roll(wet, +1, axis=0) | roll(wet, -1, axis=0)
                  | roll(wet, +1, axis=1) | roll(wet, -1, axis=1)
   dry_boundary = dilated & ~wet

2. Build lift:
   psi_lift = jnp.where(dry_boundary, bc_values, 0.0)

3. Correct RHS:
   A_lift = masked_laplacian(psi_lift, mask, dx, dy, lambda_)
   rhs_corrected = rhs - A_lift

4. Return (rhs_corrected, psi_lift)
```

**Key design choices:**

- Returns **both** `rhs_corrected` and `psi_lift` so the caller can
  reconstruct the full solution.  Returning only the corrected RHS would
  force users to recompute the lift to add it back.
- Accepts a full `(Ny, Nx)` array for `bc_values`, not just boundary
  slices.  Interior values are ignored (masked out by `dry_boundary`).
  This is simpler and more flexible -- users can pass a full reanalysis
  field without extracting boundary cells.
- Uses `masked_laplacian` (`iterative.py:144-191`) for the
  $A\psi_{\text{lift}}$ step.  No new operator.
- The JAX-native dilation replaces `scipy.ndimage.binary_dilation`, making
  the function JIT-compilable.  The mask is static (shape doesn't change),
  so the dilation is cheap.

**Proposed location:** `finitevolx/_src/solvers/inhomogeneous.py` (new file),
re-exported from `finitevolx/_src/solvers/elliptic.py` and
`finitevolx/__init__.py`.

**Signature:**

```python
def apply_inhomogeneous_bc(
    rhs: Float[Array, "Ny Nx"],
    bc_values: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"] | Mask2D,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Correct RHS for inhomogeneous Dirichlet BCs via the lifting trick.

    Parameters
    ----------
    rhs : (Ny, Nx) array
        Original right-hand side (should be zero at dry cells).
    bc_values : (Ny, Nx) array
        Prescribed boundary values.  Only values at dry cells adjacent
        to the wet domain are used; interior values are ignored.
    mask : (Ny, Nx) array or Mask2D
        Binary wet/dry mask (1=wet, 0=dry).
    dx, dy : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter.

    Returns
    -------
    rhs_corrected : (Ny, Nx) array
        Corrected RHS for the homogeneous problem.
    psi_lift : (Ny, Nx) array
        Lifting function -- add to homogeneous solution to get full solution.
    """
```

---

### Layer 2 -- `bc_values=` kwarg on Convenience Wrappers

The one-line API for common cases.  Adds an optional `bc_values` parameter to
the existing `streamfunction_from_vorticity`, `pressure_from_divergence`, and
`pv_inversion` wrappers.

```python
# Current API (unchanged, homogeneous only)
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")

# New: pass boundary values directly
psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy,
    bc="dst",
    bc_values=g,       # NEW kwarg
    mask=mask,         # required when bc_values is given
)

# Same for pressure
p = fvx.pressure_from_divergence(
    div_u, dx, dy,
    bc="dct",
    bc_values=p_boundary,
    mask=mask,
)

# Same for PV inversion (per-layer broadcast)
psi = fvx.pv_inversion(
    pv, dx, dy,
    lambda_=lambdas,
    bc_values=psi_boundary,
    mask=mask,
)
```

**Internal flow when `bc_values` is not None:**

```
1. Call apply_inhomogeneous_bc(rhs, bc_values, mask, dx, dy, lambda_)
      → (rhs_corrected, psi_lift)
2. Dispatch rhs_corrected through existing _solve_dispatch(...)
      → psi_hom
3. Return psi_lift + psi_hom
```

**When `bc_values is None`:** the wrapper takes the existing fast path.
Zero behavior change for all current call sites.

**Changes to `elliptic.py`:**

- Add `bc_values: Float[Array, "Ny Nx"] | None = None` to the signature of
  `streamfunction_from_vorticity`, `pressure_from_divergence`, and
  `pv_inversion`.
- Add `bc_values` to `_solve_dispatch` (or wrap the dispatch call).
- For `pv_inversion` with array `lambda_`: the lift must be vmapped over
  layers the same way the solve is vmapped today (`elliptic.py:437-542`).
  Since the lift depends on `lambda_` (the Helmholtz parameter appears in
  `masked_laplacian`), each layer gets its own corrected RHS.

**Spectral path on rectangular domains:**

When the user passes `bc_values` with `method="spectral"` and no mask:

- Synthesise a trivial all-wet mask whose dry-boundary ring is the outer
  ghost frame.
- Apply the standard lifting trick.
- Pass the corrected RHS to the spectral solver (which still uses DST/DCT
  internally and sees a homogeneous problem).

Alternatively, the spectral path could use the already-exported
`modify_rhs_2d` from `spectraldiffx` to encode the inhomogeneous data
directly in the transform coefficients.  Both approaches are mathematically
equivalent; the lifting trick via `masked_laplacian` is more uniform across
solver methods.

---

### Layer 3 -- Bridge to the Existing BC Vocabulary

Accept a `BoundaryConditionSet` of `Dirichlet1D` atoms wherever `bc_values`
is accepted, in addition to a raw array.

```python
from finitevolx import BoundaryConditionSet, Dirichlet1D

# User already writes this for time-stepping:
bc = BoundaryConditionSet(
    south=Dirichlet1D(face="south", value=0.0),
    north=Dirichlet1D(face="north", value=0.1),
    west=Dirichlet1D(face="west",  value=0.05),
    east=Dirichlet1D(face="east",  value=0.05),
)

# Now the SAME object can drive the elliptic solve:
psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy,
    bc=bc,           # replaces both the string "dst" AND bc_values
    mask=mask,
)
```

**Internal conversion:**

When `bc` is a `BoundaryConditionSet` (instead of a string):

1. Extract the per-face `Dirichlet1D.value` scalars (or 1-D arrays, for
   spatially-varying data -- see "Future Extensions" below).
2. Build a `(Ny, Nx)` array with the Dirichlet values at the ghost ring
   and zero in the interior.
3. Call Layer 1 (`apply_inhomogeneous_bc`) with this array.

If all four values are zero, the wrapper recognises this as homogeneous
Dirichlet and takes the fast spectral DST path (no mask needed).

**Why this matters:**

Users currently maintain two separate mental models -- `Dirichlet1D(value=g)`
for time-stepping and `bc="dst"` for elliptic solves.  Layer 3 unifies them:
one BC object, used everywhere.

**Scope:**

- **In scope for Layer 3:** `BoundaryConditionSet` containing `Dirichlet1D`
  atoms (constant value per face).
- **Out of scope:** `Neumann1D`, `Robin1D`, and spatially-varying per-face
  values.  These require deeper changes (e.g. integrating with
  `MixedBCHelmholtzSolver2D` from `spectraldiffx` for the spectral path,
  or extending `masked_laplacian` for Neumann conditions at the mask edge).
  Flagged as future work.

---

## 4. Considered Alternatives

### Option B from issue #186: `InhomogeneousBCSolver` wrapper class

```python
solver = InhomogeneousBCSolver(
    base_solver=base_solver,
    mask=mask, dx=dx, dy=dy, lambda_=0.0,
)
psi = solver(rhs, bc_values=g)
```

**Why not:** A class adds state (base solver, mask, grid params) without
adding power.  The layered function + kwarg design covers every code path the
class would, with less surface area and better composability (the pure
function in Layer 1 works with any solver without wrapping it).

### Option D from issue #186: `DirichletBC` / `NeumannBC` value objects

```python
bc = DirichletBC(values=g, mask=mask)
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc=bc)
```

**Why not now:** This is essentially what Layer 3 does, but it invents a
*new* BC type (`DirichletBC`) parallel to the existing `Dirichlet1D`.  Layer 3
reuses the existing vocabulary instead.  If the library eventually needs
`NeumannBC` / `RobinBC` for the solver layer, these can be introduced later
with clear integration points (the `MixedBCHelmholtzSolver2D` from
`spectraldiffx` for the spectral path, extended `masked_laplacian` for the
CG path).

---

## 5. Edge Cases and Decisions

### Spectral solvers without a mask

When `method="spectral"` and no mask is provided, but `bc_values` is given:

- Synthesise a rectangular mask: all-wet interior, dry boundary ring.
  This is just `mask[1:-1, 1:-1] = 1, mask[boundary] = 0` for an `(Ny, Nx)`
  domain.
- Apply the lifting trick normally.
- Document that this adds a small overhead vs. the pure spectral path
  (one extra Laplacian evaluation for the correction).

### Capacitance solver

The lift trick is operator-agnostic.  The capacitance solver
(`build_capacitance_solver`, `elliptic.py:146-183`) sees only the corrected
RHS and solves the homogeneous problem as usual.  No changes needed.

### Multigrid

Same: only the RHS changes.  The V-cycle hierarchy is unaware of the lift.
The multigrid operator (`multigrid.py:327-410`) already enforces zero BCs via
its face coefficients; the corrected RHS accounts for the non-zero data.

### Time-varying boundary values

Because Layer 1 is a pure function, the correction is recomputed at every
call inside a JIT-compiled timestep.  No caching or stale-state issues.  The
JAX-native dilation is cheap (~4 rolls + OR), and `masked_laplacian` is a
single 5-point stencil pass.

### Multi-layer PV inversion

`pv_inversion` with array `lambda_` currently vmaps the solve over layers
(`elliptic.py:437-542`).  The lift depends on `lambda_` (via
`masked_laplacian`), so each layer gets its own `rhs_corrected` and
`psi_lift`.  The implementation vmaps `apply_inhomogeneous_bc` over the layer
axis alongside the solve, or pre-computes the per-layer lifts in a batched
call.

If `bc_values` has shape `(nl, Ny, Nx)`, each layer gets its own boundary
data.  If `bc_values` has shape `(Ny, Nx)`, it is broadcast to all layers
(common case: same boundary data for all vertical modes).

### Sign of lambda

Preserved by the existing `masked_laplacian` definition: it computes
$(∇^2 - \lambda)\psi$, so both the RHS and the correction have consistent
sign.  No special-casing needed.

### Homogeneous BC special case (g = 0)

When `bc_values` is all zeros (or `None`), the lift is zero, the RHS
correction is zero, and the solver takes the fast path.  The design doc does
not need a separate code path for this -- the math handles it naturally, and
the overhead of the zero-lift check is negligible.

---

## 6. Phasing

### PR 1: Layer 1 -- `apply_inhomogeneous_bc`

**Scope:**

- New file `finitevolx/_src/solvers/inhomogeneous.py` with
  `apply_inhomogeneous_bc`.
- JAX-native dilation helper (private function in the same module).
- Re-export from `finitevolx/__init__.py`.
- Manufactured-solution test: $\psi_{\text{exact}} = \sin(\pi x)\sin(\pi y) +
  0.1\sin(2\pi y)$ (second term doesn't vanish at boundaries).  Compute
  $f = (A - \lambda)\psi_{\text{exact}}$, extract $g = \psi_{\text{exact}}\big|_{\text{boundary}}$,
  solve, verify $\|\psi - \psi_{\text{exact}}\| < \epsilon$.
- Consistency test: homogeneous BCs ($g = 0$) give the same result as the
  current API.
- Test with CG, capacitance, and spectral solver methods.

**Estimated size:** ~60 lines of implementation, ~120 lines of tests.

### PR 2: Layer 2 -- `bc_values=` kwarg

**Scope:**

- Add `bc_values` parameter to `streamfunction_from_vorticity`,
  `pressure_from_divergence`, and `pv_inversion`.
- Thread it through `_solve_dispatch`.
- Multi-layer vmap integration for `pv_inversion`.
- Auto-synthesise rectangular mask for spectral path when no mask given.
- Tests for all three wrappers with `bc_values`.

**Estimated size:** ~80 lines of implementation changes, ~150 lines of tests.

### PR 3: Layer 3 -- `BoundaryConditionSet` bridge

**Scope:**

- Accept `BoundaryConditionSet` in `bc=` parameter.
- Extract per-face Dirichlet values, build `(Ny, Nx)` lift array.
- Fast-path detection for all-zero homogeneous case.
- Update `docs/notebooks/demo_solvers.py` Section 8 to use the new API
  instead of the manual lifting trick.
- Documentation in elliptic solver guide.

**Estimated size:** ~60 lines of implementation, ~80 lines of tests, ~100
lines of doc updates.

---

## 7. Acceptance Criteria

From issue #186, with additions:

- [ ] `apply_inhomogeneous_bc(rhs, bc_values, mask, dx, dy, lambda_)` implemented
- [ ] Convenience wrappers accept `bc_values` parameter
- [ ] `BoundaryConditionSet` of `Dirichlet1D` accepted as `bc=` on wrappers
- [ ] Works with all solver methods (spectral, CG, capacitance, multigrid)
- [ ] Manufactured-solution test passes to solver tolerance
- [ ] JIT-compatible for time-stepping applications
- [ ] Multi-layer `pv_inversion` supports per-layer `bc_values`
- [ ] Documentation in elliptic solver guide
- [ ] Example in `demo_solvers` notebook updated to use new API
- [ ] JAX-native dilation (no SciPy dependency in the hot path)

---

## 8. Future Extensions (Out of Scope)

These are explicitly **not** part of the initial implementation but have clear
integration points:

| Extension | Integration point |
|-----------|-------------------|
| **Spatially-varying per-face Dirichlet values** (1-D arrays instead of scalars on `Dirichlet1D`) | Layer 3 conversion: expand 1-D face arrays into `(Ny, Nx)` lift |
| **Inhomogeneous Neumann** ($\partial\psi/\partial n = g$) | `modify_rhs_2d` from spectraldiffx (spectral path); extended `masked_laplacian` with ghost-cell injection (CG path) |
| **Robin / mixed BCs on the solver** | `MixedBCHelmholtzSolver2D` from spectraldiffx (`spectral.py:19`); new Robin-aware operator for CG |
| **Open-boundary conditions** (radiation, sponge) | Sponge1D / Robin1D already exist for time-stepping; solver integration requires absorbing-layer operator modifications |
| **3-D support** | All existing spectral solvers have 3-D variants (`solve_helmholtz_dst1_3d`, etc.); the lifting trick generalises to 3-D with a 6-point dilation |

---

*See [inhomogeneous_bcs_current.md](./inhomogeneous_bcs_current.md) for the
current state and the manual lifting trick this design replaces.*
