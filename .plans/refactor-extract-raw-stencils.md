# Plan: Extract Raw Stencil Primitives (3-Layer Architecture)

**Goal**: Refactor the operator codebase into a 3-layer architecture so that
raw index arithmetic is shared across coordinate systems (Cartesian, spherical,
cylindrical, curvilinear).

```
Layer 1:  Raw stencils      — pure index arithmetic, no scaling
Layer 2:  Scaled primitives  — Layer 1 + coordinate-specific metric scaling
Layer 3:  Compound operators — compose Layer 2 primitives
```

**Guiding principles**:
- Existing public API is unchanged (same classes, same method signatures)
- Existing tests must pass unmodified after each phase
- Raw stencils are **public API** — exported, fully documented with NumPy-style
  docstrings and half-index stencil formulas so users can compose custom operators
- Module name: `operators/stencils.py` (no underscore prefix)
- Each phase is one PR — reviewable and revertible independently

---

## Phase 1: Extract difference raw stencils

**Files**: `operators/difference.py`, new `operators/stencils.py`

Create `stencils.py` with fully documented raw difference functions (no `/ dx`
or `/ dy`, no `interior()` wrapping). Each function gets a NumPy-style
docstring with the half-index stencil formula, input/output shapes, and the
grid-point interpretation:

```python
"""Raw finite-difference and averaging stencils for Arakawa C-grids.

These functions compute the pure index arithmetic of C-grid stencils without
any metric scaling (no division by dx, dy, etc.) or ghost-ring padding.
They return arrays sized to the **interior** of the output grid location.

Use these as building blocks for custom operators on any coordinate system —
Cartesian, spherical, cylindrical, or curvilinear — by applying the
appropriate metric scale factors to the result.

All functions are pure, stateless, and compatible with ``jax.jit``,
``jax.vmap``, and ``jax.grad``.
"""

from jaxtyping import Array, Float


def diff_x_fwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Forward difference in x (centre → east face).

    Δx h[j, i+½] = h[j, i+1] − h[j, i]

    Maps T-points → U-points (or V-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (T or V), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Raw difference at interior destination points (U or X).
    """
    return h[1:-1, 2:] - h[1:-1, 1:-1]


def diff_y_fwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Forward difference in y (centre → north face).

    Δy h[j+½, i] = h[j+1, i] − h[j, i]

    Maps T-points → V-points (or U-points → X-points).
    ...
    """
    return h[2:, 1:-1] - h[1:-1, 1:-1]

# ... etc for all stencils, each with full docstring
```

Then refactor `Difference1D/2D/3D` to call these:

```python
# Before
def diff_x_T_to_U(self, h):
    return interior((h[1:-1, 2:] - h[1:-1, 1:-1]) / self.grid.dx, h)

# After
from finitevolx._src.operators.stencils import diff_x_fwd

def diff_x_T_to_U(self, h):
    return interior(diff_x_fwd(h) / self.grid.dx, h)
```

**Deliverables**:
- [x] `stencils.py` with 10 raw difference stencil functions, each with full
      NumPy-style docstring including half-index formula, shape annotations,
      and grid-point mapping (e.g., "T → U" or "V → X")
- [x] All stencil functions exported from `__init__.py`
- [x] `Difference1D` refactored (3 methods)
- [x] `Difference2D` refactored (8 primitive methods + laplacian, divergence, curl)
- [x] `Difference3D` refactored (4 primitive methods + divergence, laplacian)
- [x] All existing tests pass
- [x] Tests for raw stencils (shape, symmetry, edge cases)
- Note: `grad_perp` left as-is (uses non-standard centered stencils)

---

## Phase 2: Extract interpolation raw stencils

**Files**: `operators/interpolation.py`, extend `operators/stencils.py`

Add raw averaging functions, each with full docstrings following the same
conventions as the difference stencils:

```python
def avg_x_fwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Forward average in x (centre → east face).

    h̄[j, i+½] = ½ (h[j, i] + h[j, i+1])

    Maps T-points → U-points (or V-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (T or V), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior destination points (U or X).
    """
    return 0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:])

# ... etc for all averaging stencils
```

Then refactor `Interpolation1D/2D/3D` to call these.

**Deliverables**:
- [x] 14 raw averaging functions in `stencils.py`, each with full docstring
- [x] All averaging stencils exported from `__init__.py`
- [x] `Interpolation1D` refactored (2 methods)
- [x] `Interpolation2D` refactored (12 methods)
- [x] `Interpolation3D` refactored (4 methods)
- [x] All existing tests pass

---

## Phase 3: Refactor compound operators to use Layer 2 primitives

**Files**: `vorticity.py`, `coriolis.py`, `diagnostics.py`, `momentum.py`

These already compose `Difference2D` and `Interpolation2D`, so no stencil
extraction is needed. The goal here is to audit and confirm they only depend
on Layer 2 (the `Difference2D`/`Interpolation2D` methods) and not on raw
index arithmetic.

**Known inline stencils to extract**:
- `diagnostics.py`: `_interp_T_to_X` duplicates `Interpolation2D.T_to_X` stencil
- `diagnostics.py`: `shear_strain` and `tensor_strain` have inline diff stencils
- `diagnostics.py`: `_curl_2d` is shared with `difference.py` — consolidate
- `difference.py`: `_curl_2d` and `_divergence_2d` are module-level helpers
  that should call raw stencils

**Deliverables**:
- [x] `_interp_T_to_X` in diagnostics.py calls `avg_xy_fwd`
- [x] `shear_strain`, `tensor_strain` use raw stencils from `stencils.py`
- [x] `_curl_2d` uses raw stencils (consolidated in difference.py, Phase 1)
- [x] `kinetic_energy`, `qg_potential_vorticity` use raw stencils
- [x] `coriolis.py` and `diffusion/momentum.py` audited — already clean
- [x] All existing tests pass

---

## Phase 4: Refactor diffusion to use raw stencils

**Files**: `diffusion/diffusion.py`, `diffusion/momentum.py`

The `diffusion_2d` function has its own inline flux stencils that are
forward/backward differences. Refactor to use `_diff_x_fwd`, `_diff_x_bwd`, etc.

Note: diffusion uses **asymmetric slices** (`[1:-1, 1:-2]`, `[1:-2, 1:-1]`)
for flux arrays. These don't map directly to the standard raw stencils.
Options:
1. Add dedicated flux-stencil variants (preferred — keeps them explicit)
2. Leave as-is (acceptable — diffusion stencils are self-contained)

`MomentumAdvection2D` already composes `Difference2D` and `Interpolation2D`,
so it only needs the same audit as Phase 3.

**Deliverables**:
- [x] Decided: Option 2 (leave as-is) — diffusion flux stencils use asymmetric
      slices for no-flux BCs that don't map to standard raw stencils
- [x] `momentum.py` confirmed clean (no inline raw stencils)
- [x] All existing tests pass

---

## Phase 5: Refactor reconstruction/advection (optional, lower priority)

**Files**: `advection/reconstruction.py`, `advection/advection.py`

Reconstruction stencils (WENO, TVD, upwind) are complex multi-point stencils
that don't have spherical counterparts yet. Extracting them is useful for
consistency but not blocking for the spherical plan.

The advection flux-divergence step (`(fe - fe_shifted) / dx`) does use
raw differences — these can use `_diff_x_bwd` etc.

**Deliverables**:
- [x] Assessed: advection uses ghost=2 slicing that doesn't match standard
      raw stencils — left as-is (not blocking for spherical work)
- [ ] Reconstruction stencils cataloged for future coordinate generalization
- [x] All existing tests pass

---

## Summary of new/modified files

| File | Action |
|------|--------|
| `operators/stencils.py` | **NEW** — ~26 public raw stencil functions, fully documented |
| `operators/difference.py` | Refactor methods to call raw stencils |
| `operators/interpolation.py` | Refactor methods to call raw stencils |
| `operators/diagnostics.py` | Replace inline stencils with calls to raw stencils |
| `operators/vorticity.py` | Audit only (already composes Diff/Interp) |
| `operators/coriolis.py` | Audit only |
| `diffusion/diffusion.py` | Optionally refactor flux stencils |
| `diffusion/momentum.py` | Audit only |
| `advection/advection.py` | Optionally refactor flux divergence |
| `__init__.py` | Export all raw stencil functions |
| `tests/test_stencils.py` | **NEW** — raw stencil tests |

**Risk**: Low. Each phase is mechanical extraction — same index arithmetic,
just moved to named functions. Tests verify no regressions.

**Not in scope**: Adding new coordinate systems or modifying grid classes.
Those belong to Plan 2 (spherical operators).

## Documentation

The `stencils.py` module is user-facing and should be included in the API
docs. Each function's docstring should include:
1. One-line summary with grid-point mapping (e.g., "Forward difference in x (T → U)")
2. Half-index stencil formula (e.g., `Δx h[j, i+½] = h[j, i+1] − h[j, i]`)
3. Which source/destination points it maps between
4. Input shape with ghost ring, output shape (interior only)
5. A note that no metric scaling is applied — the caller divides by `dx`, `R·cosφ·dλ`, etc.

Example usage in user docs:

```python
from finitevolx import diff_x_fwd, interior

# Custom operator: gradient with user-defined metric
raw = diff_x_fwd(h)                     # pure index arithmetic
scaled = raw / my_custom_metric          # user applies their own scaling
result = interior(scaled, h)             # pad back to full grid shape
```
