# Refactor: Eliminate Code Duplication Across Modules

## Problem Statement

The codebase has a systematic duplication pattern: **3D operator classes reimplement
their 2D counterparts with an extra z-index** instead of vmapping the 2D versions.
Additionally, several diagnostic functions inline stencil computations that already
exist as shared primitives.

---

## Phase 1: 3D Operators → vmap(2D) Wrappers

### 1A. `Coriolis3D` (coriolis.py:126-235)

**Current**: Manually computes T→U, T→V, V→U, U→V interpolations and Coriolis
products with raw 3D slicing (~110 lines).

**Target**: Instantiate `Coriolis2D` internally, vmap its `__call__` over the
z-axis. The `f` parameter is 2D (depth-independent), so it broadcasts naturally.

**Estimated changes**:
- Add `Interpolation2D` import (already imported)
- Store a `Coriolis2D` instance (or a 2D grid derived from the 3D grid)
- Replace `__call__` body with vmap over z-levels
- Need a helper to extract a 2D grid from a 3D grid (or pass dx/dy directly)
- ~80 lines removed, ~15 lines added

**Risk**: Need `ArakawaCGrid2D` from `ArakawaCGrid3D`. Currently no such factory
method exists. Options:
  1. Add `ArakawaCGrid3D.to_horizontal_grid() -> ArakawaCGrid2D`
  2. Construct `ArakawaCGrid2D` manually from `(Ny, Nx, Lx, Ly, dx, dy)`
  3. Have `Coriolis3D` directly call a functional helper instead of the class

**Recommendation**: Option 2 — construct `ArakawaCGrid2D` directly. This avoids
changing the grid API and is a one-liner.

**Tests**: Existing `Coriolis3D` tests (if any) must pass. Add a numerical
equivalence test comparing old inline vs new vmap implementation.

---

### 1B. `Vorticity3D.relative_vorticity` (vorticity.py:209-249)

**Current**: Manually computes curl with 3D slicing (~40 lines).

**Target**: Vmap `_curl_2d(u_k, v_k, dx, dy)` over the z-axis, matching how
`Vorticity2D.relative_vorticity` delegates to `self.diff.curl`.

**Estimated changes**:
- Import `_curl_2d` from difference module
- Replace body with `jax.vmap(lambda u_k, v_k: _curl_2d(u_k, v_k, dx, dy))(u, v)`
- ~20 lines removed, ~5 lines added

**Risk**: Low. `_curl_2d` is already the shared primitive used by `Vorticity2D`.

**Tests**: Verify numerical equivalence on a small 3D grid.

---

### 1C. `MomentumAdvection3D` (momentum.py:209-457)

**Current**: Reimplements all interpolation, KE gradient, vorticity flux, and
blending logic with 3D slicing (~250 lines). This is the largest duplication.

**Target**: Instantiate `MomentumAdvection2D` and vmap its `__call__` over z-levels.

**Estimated changes**:
- Store a `MomentumAdvection2D` instance
- Replace `__call__` with vmap of the 2D operator
- Remove `_kinetic_energy_gradients`, `_vorticity_flux_energy`,
  `_vorticity_flux_enstrophy` methods entirely
- ~230 lines removed, ~15 lines added

**Risk**: Medium. The 2D version writes to `[2:-2, 2:-2]` while 3D writes to
`[1:-1, 2:-2, 2:-2]`. Need to verify the z-ghost-cell handling is consistent
with vmap (vmap over z should produce `[1:-1, ...]` automatically if the input
z-ghost slices are zero).

Actually, vmap applies per-slice, so the output per z-slice writes `[2:-2, 2:-2]`.
The z-ghost slices (index 0 and -1) will also be processed by vmap but should
produce zeros if inputs are zero there. This matches the current `[1:-1, 2:-2, 2:-2]`
write pattern only if we explicitly zero-out z-ghost results. May need
`jnp.zeros` for z=0 and z=-1 slices, or just vmap over `[1:-1]` and pad.

**Recommendation**: Vmap over all z-levels (including ghost). The ghost z-slices
have zero inputs, so the 2D operator produces zero outputs — matching current
behavior. Verify with a test.

**Tests**: Numerical equivalence test for all three schemes (energy, enstrophy, al).

---

### 1D. `Diffusion3D` (diffusion.py:295-445)

**Current**: Reimplements flux-form diffusion stencil with 3D slicing (~150 lines
for `__call__` + `fluxes`).

**Target**: Vmap `diffusion_2d` over the z-axis for `__call__`. For `fluxes`,
vmap `Diffusion2D.fluxes`.

**Estimated changes**:
- Replace `__call__` body with vmap of `diffusion_2d`
- Replace `fluxes` body with vmap of 2D fluxes
- ~100 lines removed, ~20 lines added

**Risk**: Medium. `kappa` can be scalar or `[Nz, Ny, Nx]`. When scalar, it
broadcasts naturally. When 3D array, need to vmap over kappa's z-axis too.
Same for mask arrays. The vmap `in_axes` specification needs care:
`in_axes=(0, 0_or_None, None, None, 0_or_None, 0_or_None, 0_or_None)`.

**Tests**: Numerical equivalence for scalar kappa, array kappa, with/without masks.

---

## Phase 2: Diagnostics Deduplication

### 2A. `qg_potential_vorticity` — inline Laplacian (diagnostics.py:456-465)

**Current**: Manually computes `d2x + d2y` (centered second differences).

**Target**: This is a pure functional diagnostic — it intentionally avoids
requiring a grid/operator object. The inline Laplacian is only 4 lines and
is semantically clear. **Recommend keeping as-is.** Adding a `Difference2D`
dependency would require a grid object, changing the function signature.

**Decision**: No change. The duplication is minimal and the functional API
intentionally avoids class dependencies.

---

### 2B. `vertical_velocity` — inline divergence (diagnostics.py:615-620)

**Current**: Manually computes `du/dx + dv/dy` per z-level.

**Target**: Could use `_divergence_2d` per z-level via vmap. However, the current
code operates on 3D arrays with a single vectorized expression, which is already
efficient. The duplication is ~4 lines.

**Decision**: No change. Minimal duplication, and the 3D vectorized form is
arguably clearer for this specific use case.

---

### 2C. `_interp_T_to_X` in diagnostics (diagnostics.py:829-843)

**Current**: Private helper duplicating `Interpolation2D.T_to_X`.

**Verification needed**: The index conventions must match exactly.
- `Interpolation2D.T_to_X`: averages `h[1:-1,1:-1] + h[1:-1,2:] + h[2:,1:-1] + h[2:,2:]`
- `_interp_T_to_X`: averages `field[:-2,:-2] + field[1:-1,:-2] + field[:-2,1:-1] + field[1:-1,1:-1]`

These are **different stencils** — they pick different neighbor sets relative to
the output index. Need to verify which is correct for the shallow-water PV context.

**Decision**: Investigate and align. If they compute the same thing (just with
different index offsets mapping to the same physical stencil), replace with a
call to `Interpolation2D.T_to_X`. If semantically different, document why.

**Estimated changes**: ~10 lines changed if replacement is valid.

---

### 2D. `shear_strain` and `tensor_strain` (diagnostics.py:207-248)

**Current**: Inline `dv/dx + du/dy` (shear) and `du/dx - dv/dy` (tensor).

**Analysis**: These are NOT duplicates of `_curl_2d` or `_divergence_2d`:
- `_curl_2d` = `dv/dx - du/dy` (vorticity)
- `shear_strain` = `dv/dx + du/dy` (different sign!)
- `_divergence_2d` = `du/dx + dv/dy` (at T-points)
- `tensor_strain` = `du/dx - dv/dy` (different sign!)

**Decision**: No change. These are semantically distinct operations.

---

## Phase 3: Structural Improvement

### 3A. Add `ArakawaCGrid2D` construction helper

Several 3D operators will need to construct a 2D grid from 3D grid parameters.
Rather than repeating `ArakawaCGrid2D(Nx=grid.Nx, Ny=grid.Ny, ...)` in each
class, add a utility.

**Options**:
1. `ArakawaCGrid3D.horizontal_grid() -> ArakawaCGrid2D` method
2. A standalone function `horizontal_grid_from_3d(grid3d) -> ArakawaCGrid2D`
3. Just inline the construction (it's a one-liner)

**Recommendation**: Option 1 — cleanest API. One method, reusable everywhere.

**Estimated changes**: ~10 lines added to `grid.py`.

---

## Execution Plan

### Step 1: Add `horizontal_grid()` to `ArakawaCGrid3D` (Phase 3A)
- Modify `finitevolx/_src/grid/grid.py`
- Add to `__init__.py` exports if needed (method, not standalone — no export needed)
- Add unit test

### Step 2: Refactor `Vorticity3D` (Phase 1B) — smallest, lowest risk
- Modify `finitevolx/_src/operators/vorticity.py`
- Add numerical equivalence test
- Run full test suite

### Step 3: Refactor `Coriolis3D` (Phase 1A)
- Modify `finitevolx/_src/operators/coriolis.py`
- Uses `horizontal_grid()` from Step 1
- Add numerical equivalence test
- Run full test suite

### Step 4: Refactor `Diffusion3D` (Phase 1D)
- Modify `finitevolx/_src/diffusion/diffusion.py`
- Handle vmap over variable-shape kappa and masks
- Add numerical equivalence tests
- Run full test suite

### Step 5: Refactor `MomentumAdvection3D` (Phase 1C) — largest, highest risk
- Modify `finitevolx/_src/diffusion/momentum.py`
- Most lines removed, most potential for subtle breakage
- Add numerical equivalence tests for all 3 schemes
- Run full test suite

### Step 6: Verify `_interp_T_to_X` (Phase 2C)
- Compare stencil indices between `Interpolation2D.T_to_X` and `_interp_T_to_X`
- If equivalent, replace; if not, document the difference
- Run full test suite

### Step 7: Final validation
- `uv run pytest tests -v`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run ty check finitevolx`

---

## Estimated Impact

| Change | Lines Removed | Lines Added | Risk |
|--------|-------------|-------------|------|
| `horizontal_grid()` | 0 | ~10 | Low |
| `Vorticity3D` | ~20 | ~5 | Low |
| `Coriolis3D` | ~80 | ~15 | Low |
| `Diffusion3D` | ~100 | ~20 | Medium |
| `MomentumAdvection3D` | ~230 | ~15 | Medium |
| `_interp_T_to_X` | ~10 | ~5 | Low |
| **Total** | **~440** | **~70** | |

Net reduction: **~370 lines** of duplicated stencil code.

---

## What We're NOT Changing (and Why)

- **`diagnostics.py` inline Laplacian/divergence** (2A, 2B): Minimal duplication
  (4 lines each), and the functional API intentionally avoids class dependencies.
- **`shear_strain` / `tensor_strain`** (2D): Not actually duplicates — different
  sign combinations from curl/divergence.
- **`geographic.py` spherical operators**: Justified — spherical metric terms
  make these fundamentally different from Cartesian stencils.
- **`diffusion_2d` standalone function**: Already the shared primitive that
  `Diffusion2D` delegates to. No duplication.
- **`BiharmonicDiffusion2D/3D`**: Already properly compose `Diffusion2D/3D`.
