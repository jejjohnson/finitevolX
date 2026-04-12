Context
The original design doc (committed on claude/inhomogeneous-bcs-design-r3fte)
proposed a 3-layer API narrowly scoped to inhomogeneous Dirichlet BCs.
After user review, the scope has expanded based on four critiques:

Broader BC scope — design a unified BC/BCSet that works for
time-stepping, solvers, and direct state application (not just solver-side
inhomogeneous Dirichlet).
BCSet owns the mask — the BoundaryConditionSet should carry a
Mask2D, so solvers can read the mask from the BC object.
Solver operators accept the mask/BC; functionals don't — class-based
solver operators hold the mask; functional convenience wrappers
(streamfunction_from_vorticity) extract it from the BCSet.
known_values, not bc_values — generalize from "boundary values" to
"known values" at any cell: outer boundary, island coasts (auto-derived
from mask topology), AND sparse interior observations (explicit
known_mask). Auto-derive + merge approach.

Deliverables
Rewrite docs/design/research/boundaries/inhomogeneous_bcs_design.md on the
same branch. The current-state doc stays mostly unchanged (minor terminology
updates). Commit + push. Then create PR.

Revised Design Doc Outline
Section 1: Goal (expanded)
A unified boundary condition system for finitevolX that:

Works for time-stepping (ghost-cell updates), direct state application,
and elliptic solvers
Carries domain geometry (Mask2D) so solvers don't need a separate mask arg
Supports known values at any cell — outer boundaries, island coasts,
and sparse interior observations
Is JIT-compilable, vmap-compatible, and causes zero breakage for existing
call sites

Section 2: Unified BC and BCSet Design (NEW — top of doc)
This is the biggest change. Start the design doc with the proposed BC
class hierarchy before getting into the solver integration.
2.1 BC Atoms (evolve existing bc_1d.py)
Keep the existing 9 atom types (Dirichlet1D, Neumann1D, etc.) with their
current interface. No mask on individual atoms — they remain face-level
building blocks.
2.2 BoundaryConditionSet (evolve existing bc_set.py)
Add two new fields to the existing BoundaryConditionSet:
pythonclass BoundaryConditionSet(eqx.Module):
    # --- Existing fields (unchanged) ---
    south: BoundaryCondition1D | None = None
    north: BoundaryCondition1D | None = None
    west: BoundaryCondition1D | None = None
    east: BoundaryCondition1D | None = None

    # --- NEW fields ---
    mask: Mask2D | Float[Array, "Ny Nx"] | None = None
    known_values: Float[Array, "Ny Nx"] | None = None
    known_mask: Bool[Array, "Ny Nx"] | None = None  # explicit obs locations
Mask ownership:

BCSet owns a Mask2D (or raw float array).
Each atom extracts the right staggering when it runs (h/u/v/q from Mask2D).
Solvers read the mask from the BCSet — user doesn't pass it separately.
mask=None for simple rectangular domains (backward compatible).

Known values:

known_values: (Ny, Nx) array of prescribed values. Only cells marked
as "known" are used; interior values at non-known cells are ignored.
known_mask: (Ny, Nx) boolean marking explicit known-value locations
(e.g., sparse observations at interior wet cells).
Both are optional. When absent, behavior is identical to today.

Auto-derive + merge for boundary cells:
boundary_cells = auto_derive_from_mask(mask)   # outer ring + island coasts
if known_mask is not None:
    all_known = boundary_cells | known_mask     # merge in obs
else:
    all_known = boundary_cells
Three scenarios this covers:
Scenario 1: Simple basin        Scenario 2: Basin + island     Scenario 3: + observations
0 0 0 0 0 0 0 0                 0 0 0 0 0 0 0 0                0 0 0 0 0 0 0 0
0 . . . . . . 0                 0 . . . . . . 0                0 . . . . . . 0
0 . . . . . . 0                 0 . . 0 0 . . 0                0 . . . . . . 0
0 . . . . . . 0                 0 . . 0 0 . . 0                0 . X . . X . 0
0 . . . . . . 0                 0 . . . . . . 0                0 . . . . . . 0
0 0 0 0 0 0 0 0                 0 0 0 0 0 0 0 0                0 . . . X . . 0
                                                                0 0 0 0 0 0 0 0
known_mask: not needed          known_mask: not needed          known_mask: marks X cells
Auto-derive handles it          Auto-derive handles islands     Auto + merge with explicit
Unified interface:
pythonbc = BoundaryConditionSet(
    mask=cgrid_mask,
    south=Dirichlet1D("south", value=0.0),
    north=Dirichlet1D("north", value=0.1),
    west=Neumann1D("west", value=0.0),
    east=Dirichlet1D("east", value=0.05),
    known_values=obs_field,       # optional
    known_mask=obs_locations,     # optional
)

# Works everywhere:
state = bc(state, dx, dy)                                    # time-stepping
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc=bc) # solver
2.3 FieldBCSet (minimal change)
No structural change — it dispatches per-field BCSet as before. Each BCSet
in the map can now carry its own mask + known values.
Section 3: Design Principles (updated)

Unified vocabulary — one BC object for time-stepping and solvers.
BCSet owns the mask — solvers extract it; users don't pass it twice.
known_values generalizes boundary data — outer rim, islands, and
sparse obs all go through the same mechanism.
Auto-derive + merge — boundary cells from mask topology; merge with
explicit known_mask for interior constraints.
Lifting trick is the implementation — users never write
binary_dilation; the generalized lift handles all known cells.
JAX-native — no SciPy in the hot path.
Additive, not breaking — new fields default to None.

Section 4: Solver Integration (3 layers, revised)
Layer 1 — Pure function: apply_known_values
Renamed from apply_inhomogeneous_bc. Accepts mask + known_values +
known_mask explicitly (maximum composability for power users).
pythonrhs_corrected, value_lift = apply_known_values(
    rhs=rhs,
    known_values=known_field,
    mask=mask,
    dx=dx, dy=dy,
    lambda_=0.0,
    known_mask=obs_mask,  # optional; auto-derives boundary ring if None
)
psi_hom = my_solver(rhs_corrected)
psi = value_lift + psi_hom
Internally:

Auto-derive boundary ring from mask (JAX-native dilation)
Merge with explicit known_mask if provided
Build lift: value_lift = where(all_known, known_values, 0.0)
Compute effective_solve_mask = wet_mask & ~all_known
Correct RHS: rhs' = rhs - masked_laplacian(value_lift, effective_solve_mask, ...)
Return (rhs_corrected, value_lift)

Layer 2 — Convenience wrappers accept known_values / known_mask
pythonpsi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy,
    bc="dst",
    known_values=g,        # NEW
    known_mask=obs_mask,   # NEW, optional
    mask=mask,
)
When known_values is None → existing fast path. When given → calls
apply_known_values internally.
Layer 3 — BCSet passthrough (unified)
pythonbc = BoundaryConditionSet(
    mask=mask,
    south=Dirichlet1D("south", value=0.0),
    known_values=obs_field,
    known_mask=obs_locations,
)

# Solver reads mask, known_values, known_mask from bc
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc=bc)
When bc is a BoundaryConditionSet (not a string):

Extract mask from bc.mask
If bc has per-face Dirichlet atoms, expand to (Ny, Nx) boundary values
Merge with bc.known_values / bc.known_mask
Call Layer 1

If all values are zero and no known_mask → fast spectral path.
Section 5: Considered Alternatives (brief)

Option B (wrapper class): adds state without power
Option D (new DirichletBC type): reinvents existing vocabulary

Section 6: Edge Cases

Spectral without mask: synthesise rectangular mask
Capacitance / multigrid: only RHS changes, solver unaware
Time-varying known_values: recomputed each step (pure function)
Multi-layer PV: vmap over layers
Sparse obs: effective_solve_mask shrinks to exclude obs cells

Section 7: Phasing

PR 1: Evolve BoundaryConditionSet (add mask, known_values, known_mask)

apply_known_values pure function + tests


PR 2: Thread through convenience wrappers (known_values/known_mask kwargs

BCSet passthrough)


PR 3: Update demo_solvers notebook + docs

Section 8: Acceptance Criteria
(Updated from issue #186 + new requirements)
Section 9: Future Extensions

Spatially-varying per-face Dirichlet
Inhomogeneous Neumann / Robin on solver
3-D support


Files to modify
FileChangedocs/design/research/boundaries/inhomogeneous_bcs_design.mdFull rewrite with revised designdocs/design/research/boundaries/inhomogeneous_bcs_current.mdMinor: rename bc_values → known_values in Section 7 table
Verification

All code snippets reference real existing file:line locations
API names consistent across all three layers
Backward compatibility: existing call sites unchanged when new fields = None
Cross-links between the two docs maintained
