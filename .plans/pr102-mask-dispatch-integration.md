# PR #102 / Issue #68 — Remaining Implementation Plan

**Branch:** `copilot/add-upwind-flux-dispatch`
**Goal:** Wire the existing `upwind_flux` function into `Advection2D/3D`, refactor
`Reconstruction2D/3D` masked methods to delegate to it, address all PR review
comments, and add missing pieces.

## What's Already Done

- `upwind_flux()` in `finitevolx/_src/flux.py` — fully implemented (2D only)
- Exported in `__init__.py`
- Comprehensive tests in `tests/test_flux.py` (rectangular, masked, conservation)
- All 7 issue #68 acceptance criteria satisfied at the `upwind_flux` level
- 6 `Reconstruction2D.*_masked` methods exist (but with inline dispatch, not delegating)
- 6 `Reconstruction3D.*_masked` methods exist (tvd_x/y, weno5_x/y, wenoz5_x/y)
- `ArakawaCGridMask.stencil_capability: StencilCapability` — already exists
- `ArakawaCGridMask.get_adaptive_masks()` — already uses `StencilCapability` internally
  to produce mutually-exclusive masks from directional wet-cell counts

## Issue #68 Comment Context

The issue received a comment identifying snippet source material from `jej_vc_snippets`
with three relevant patterns:

1. **Robust Upwind Flux** — auto-degrading: WENO5 → WENO3 → upwind1 → zero (trapped).
   Already implemented by `upwind_flux` + `get_adaptive_masks`. The trapped-against-land
   case (cell `False` in all masks) correctly produces zero flux.

2. **Robust Central Difference** — central → forward/backward fallback near walls.
   Out of scope for this PR (advection only, not central differences).

3. **`Reconstruction2D.dispatch_x()` suggestion** — a unified method that auto-selects
   reconstruction based on local stencil availability. The PR's approach achieves this
   differently: `_rec_funcs_for_method_2d` maps method names to stencil hierarchies,
   and `upwind_flux` does the actual dispatch. This is functionally equivalent without
   adding a new method to `Reconstruction2D`.

The comment also noted related issues (#13, #22, #24) — these are out of scope here.

## PR Review Comments to Address

The Copilot reviewer left 4 distinct actionable comments on the existing code:

### RC1: Validate `rec_funcs` / `mask_hierarchy` key agreement in `upwind_flux`
**File:** `finitevolx/_src/flux.py`
If `rec_funcs` and `mask_hierarchy` have mismatched keys, a `KeyError` fires deep in
the loop. Add upfront validation:
```python
missing_masks = set(stencil_sizes) - set(mask_hierarchy.keys())
if missing_masks:
    raise ValueError(
        "mask_hierarchy is missing masks for stencil sizes "
        f"{sorted(missing_masks)}; got keys {sorted(mask_hierarchy.keys())}"
    )
```

### RC2: NaN-safe blending in `upwind_flux`
**File:** `finitevolx/_src/flux.py` (line 165)
`face_mask * flux` propagates NaN/Inf from higher-order stencils near land where
`face_mask` is `False` (because `0.0 * NaN == NaN`). Replace with:
```python
safe_flux = jnp.where(face_mask, flux_interior, 0.0)
interior = interior + safe_flux
```

### RC3: Remove unused `all_ocean_mask` fixture from tests
**File:** `tests/test_flux.py`
Several tests accept `all_ocean_mask` but never use it. Remove the fixture parameter
to avoid PLW0613 lint failures.

### RC4: `upwind2` stencil size mapping concern
**Context:** `_rec_funcs_for_method_2d` (to be implemented in Task 6)
`upwind2` only needs one upstream neighbor (not a symmetric 3-cell stencil), but
mapping it to stencil size 4 via `get_adaptive_masks` is stricter than necessary.
**Decision:** Map `upwind2` directly without mask dispatch (stencil size 2 suffices),
or simply let it fall through to the unmasked path since its actual data dependency
is identical to `upwind1`. The PR description says `upwind2/upwind3` fall through
unchanged — we follow that.

## Remaining Tasks

### Task 1: Fix `upwind_flux` — review comment fixes (RC1 + RC2)

Address both review comments on the existing `upwind_flux`:
1. Add key-set validation after existing input checks (RC1)
2. Replace `face_mask * fluxes[s][1:-1, 1:-1]` with
   `jnp.where(face_mask, fluxes[s][1:-1, 1:-1], 0.0)` (RC2)

**File:** `finitevolx/_src/flux.py`

### Task 2: Fix tests — remove unused fixtures (RC3)

Remove unused `all_ocean_mask` fixture parameter from test functions that don't use it.

**File:** `tests/test_flux.py`

### Task 3: Add `PLR0911/0912/0915` to ruff ignore in `pyproject.toml`

Add to the `ignore` list (line 114-131):
```toml
"PLR0911",  # too many return statements
"PLR0912",  # too many branches
"PLR0915",  # too many statements
```

Also remove the now-redundant `# noqa: PLR0915` comments from:
- `scripts/swm_linear.py:307`
- `scripts/qg_1p5_layer.py:374`
- `scripts/shallow_water.py:289`

**Files:** `pyproject.toml`, `scripts/swm_linear.py`, `scripts/qg_1p5_layer.py`,
`scripts/shallow_water.py`

### Task 4: Refactor 6 `Reconstruction2D.*_masked` methods to delegate to `upwind_flux`

Replace the ~140 lines of inline `jnp.where` tier-selection in these 6 methods with
thin calls to `upwind_flux`:

1. `tvd_x_masked` (line 1119) — currently 45 lines → ~5 lines
2. `tvd_y_masked` (line 1165) — currently 43 lines → ~5 lines
3. `weno5_x_masked` (line 1209) — currently 47 lines → ~5 lines
4. `weno5_y_masked` (line 1257) — currently 47 lines → ~5 lines
5. `wenoz5_x_masked` (line 1305) — currently 44 lines → ~5 lines
6. `wenoz5_y_masked` (line 1350) — currently 44 lines → ~5 lines

Each becomes:
```python
def weno5_x_masked(self, h, u, mask):
    amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
    rec_funcs = {2: self.upwind1_x, 4: self.weno3_x, 6: self.weno5_x}
    return upwind_flux(h, u, dim=1, rec_funcs=rec_funcs, mask_hierarchy=amasks)
```

**File:** `finitevolx/_src/reconstruction.py` (lines 1119-1393)

### Task 5: Add `Reconstruction3D.weno3_x_masked` and `weno3_y_masked`

The 3D class already has `tvd_x/y_masked` and `weno5_x/y_masked` but is missing
`weno3` masked variants. These are needed for `Advection3D(mask=..., method='weno3')`.

Pattern: same as existing 3D masked methods — get adaptive masks with `(2, 4)`,
broadcast 2D mask over z with `[None, ...]`, select between `weno3_x/y` and
`upwind1_x/y`.

**File:** `finitevolx/_src/reconstruction.py` (insert after line ~1946, after
`tvd_y_masked`)

### Task 6: Add `_rec_funcs_for_method_2d` helper to `advection.py`

Build a helper that maps a method name string to the `rec_funcs` dict consumed by
`upwind_flux`. This avoids repeating the stencil-hierarchy mapping in every call site.

```python
def _rec_funcs_for_method_2d(
    recon: Reconstruction2D, method: str
) -> tuple[dict[int, Callable], dict[int, Callable], tuple[int, ...]]:
    """Return (rec_funcs_x, rec_funcs_y, stencil_sizes) for the given method."""
```

Mappings:
- `"weno5"` → `{2: recon.upwind1_*, 4: recon.weno3_*, 6: recon.weno5_*}`, sizes `(2,4,6)`
- `"wenoz5"` → `{2: recon.upwind1_*, 4: recon.wenoz3_*, 6: recon.wenoz5_*}`, sizes `(2,4,6)`
- `"weno3"` → `{2: recon.upwind1_*, 4: recon.weno3_*}`, sizes `(2,4)`
- TVD limiters → `{2: recon.upwind1_*, 4: recon.tvd_*(limiter=name)}`, sizes `(2,4)`

Note on RC4: `upwind1/upwind2/upwind3` and `naive` are NOT mask-dispatchable — they
fall through to the existing unmasked path. `weno7/weno9` also fall through (no
lower-order hierarchy defined for 8/10-point stencils in the current design).

This is functionally equivalent to the `Reconstruction2D.dispatch_x()` method
suggested in issue #68's comment, but keeps dispatch logic in the advection layer
rather than adding a new method to the reconstruction class.

**File:** `finitevolx/_src/advection.py`

### Task 7: Add `mask` parameter to `Advection2D.__call__`

Add `mask: ArakawaCGridMask | None = None` to the signature. When `mask` is provided
and the method is one of `weno3/weno5/wenoz5` or a TVD limiter, route through
`upwind_flux` using the helper from Task 6. When `mask` is `None` or method is
`naive/upwind1/upwind2/upwind3/weno7/weno9`, fall through to the existing unmasked
path.

The trapped-against-land case (cells where even upwind1 can't be supported) is
handled automatically: those cells are `False` in all masks from
`get_adaptive_masks()`, so `upwind_flux` produces zero flux there — correct for
no-flux land boundaries.

**File:** `finitevolx/_src/advection.py` (lines 112-187)

### Task 8: Add `mask` parameter to `Advection3D.__call__`

Add `mask: ArakawaCGridMask | None = None` to the signature. When `mask` is provided,
route to the existing `Reconstruction3D.*_masked` methods for supported methods
(`weno3`, `weno5`, TVD limiters). `weno7`/`weno9`/`naive` fall through to unmasked.

Note: `upwind_flux` is 2D-only, so Advection3D routes to `Reconstruction3D.*_masked`
methods directly (which have their own inline mask dispatch with z-broadcast).

**File:** `finitevolx/_src/advection.py` (lines 205-270)

### Task 9: Tests for new Advection mask integration

Add tests verifying:
- `Advection2D(grid)(h, u, v, method="weno5", mask=mask)` works on rectangular domain
- `Advection2D` with mask on irregular domain falls back correctly
- `Advection3D(grid)(h, u, v, method="weno3", mask=mask)` works
- `Advection3D` with mask on irregular domain falls back correctly
- NaN-safety: fields with NaN on land cells don't corrupt masked flux results
- Trapped-against-land: cells surrounded by land produce zero flux

**File:** `tests/test_flux.py` or new `tests/test_advection_masked.py`

## Implementation Order

1. **Task 1** — Fix `upwind_flux` (RC1 + RC2) — prerequisite for everything else
2. **Task 2** — Fix unused test fixtures (RC3)
3. **Task 3** — pyproject.toml lint config — unblocks CI
4. **Task 4** — Refactor Reconstruction2D masked methods (core refactor)
5. **Task 5** — Add Reconstruction3D.weno3 masked (small addition)
6. **Task 6** — Add `_rec_funcs_for_method_2d` helper
7. **Task 7** — Advection2D mask param
8. **Task 8** — Advection3D mask param
9. **Task 9** — Tests

## Architecture

```
ArakawaCGridMask
    ├─ stencil_capability: StencilCapability
    │   ├─ x_pos, x_neg, y_pos, y_neg  (contiguous wet-cell counts)
    │   └─ Computed from h-mask via StencilCapability.from_mask()
    │
    └─ get_adaptive_masks(direction, source, stencil_sizes)
        ├─ Uses stencil_capability internally
        ├─ Returns: dict[stencil_size → mutually-exclusive Bool mask]
        └─ Cells where even smallest stencil fails → False in ALL masks

upwind_flux() [2D only, in flux.py]
    ├─ Takes: rec_funcs dict + mask_hierarchy (from get_adaptive_masks)
    ├─ For each interior face: identifies upwind cell via flow sign
    ├─ Selects reconstruction tier based on upwind cell's mask
    ├─ Trapped cells (False everywhere) → zero flux (no-flux BC)
    └─ NaN-safe via jnp.where blending (RC2)

_rec_funcs_for_method_2d() [in advection.py]
    ├─ Maps method name → (rec_funcs_x, rec_funcs_y, stencil_sizes)
    └─ Equivalent to issue #68's suggested Reconstruction2D.dispatch_x()

Advection2D.__call__(h, u, v, method, mask=None)
    ├─ mask=None → existing unmasked path (unchanged)
    └─ mask provided + dispatchable method → upwind_flux per axis

Advection3D.__call__(h, u, v, method, mask=None)
    ├─ mask=None → existing unmasked path (unchanged)
    └─ mask provided → Reconstruction3D.*_masked (inline z-broadcast)
```

## Key Design Decisions

- **`upwind_flux` stays 2D-only.** The existing 3D masked methods broadcast the 2D
  mask over z themselves; no need for a 3D variant of `upwind_flux`.
- **Advection3D routes to `Reconstruction3D.*_masked` directly**, not through
  `upwind_flux`, since the 3D methods handle the z-broadcast internally.
- **The `_rec_funcs_for_method_2d` helper returns x and y rec_funcs dicts** plus
  stencil sizes, so `Advection2D.__call__` can build mask hierarchies for both axes.
- **No `Reconstruction2D.dispatch()` method.** The issue #68 comment suggested this,
  but the same functionality is achieved by `_rec_funcs_for_method_2d` + `upwind_flux`
  without adding new methods to the reconstruction class.
- **Backward compatible.** `mask=None` (default) preserves the existing unmasked
  behavior exactly.
- **`upwind2/upwind3/naive/weno7/weno9` are not mask-dispatchable** — they fall
  through to unmasked. This follows the PR description and addresses RC4.
- **NaN safety** is enforced by `jnp.where` in the blending loop (RC2), so fields
  with NaN on dry cells won't corrupt results.
- **Trapped-against-land** cells produce zero flux automatically (all masks `False`),
  implementing the `else: flux = 0.0` pattern from the `jej_vc_snippets` reference.
