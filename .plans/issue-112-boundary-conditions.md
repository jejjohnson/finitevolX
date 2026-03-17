# Plan: Issue #112 — Epic: Boundary Conditions

**Goal:** Complete the BC taxonomy by adding Robin (#106) and high-order Extrapolation (#107) boundary conditions.

**Target module:** `finitevolx/_src/bc_1d.py`

---

## Sub-issue #106: Robin Boundary Condition

**What:** `α·u + β·∂u/∂n = γ` — a linear combination of Dirichlet (β=0) and Neumann (α=0).

### Ghost cell derivation

The boundary value sits at the wall between the ghost cell and the first interior cell.
Using the existing conventions:

- Wall value: `u_wall = 0.5 * (u_ghost + u_interior)`
- Wall gradient (outward): `∂u/∂n = sign * (u_ghost - u_interior) / spacing`

where `sign = _outward_sign(face)` and `spacing = _normal_spacing(face, dx, dy)`.

Substituting into `α·u_wall + β·∂u/∂n = γ`:

```
α · 0.5·(u_ghost + u_int) + β · sign·(u_ghost - u_int) / spacing = γ

u_ghost · (α/2 + β·sign/spacing) = γ - u_int · (α/2 - β·sign/spacing)

u_ghost = (γ - u_int · (α/2 - β·sign/spacing)) / (α/2 + β·sign/spacing)
```

Simplify with `s = β · sign / spacing`:

```
u_ghost = (γ - u_int · (α/2 - s)) / (α/2 + s)
```

### Steps

1. **Add `Robin1D` class** to `bc_1d.py`
   - Fields: `face` (static), `alpha`, `beta`, `gamma`
   - `__call__(self, field, dx, dy)` → computes ghost using formula above
   - Validate: `alpha` and `beta` must not both be zero
   → verify: instantiate with sample values, check ghost cell formula by hand

2. **Add `Robin1D` to the union type** `BoundaryCondition1D` in `bc_1d.py`
   → verify: type alias includes `Robin1D`

3. **Export from `__init__.py`**
   - Add import and `__all__` entry for `Robin1D`
   → verify: `from finitevolx import Robin1D` works

4. **Unit tests** in `tests/test_boundary.py`
   - Test ghost cell value for a known (α, β, γ) triple on each face
   - Test Dirichlet recovery: β=0, verify matches `Dirichlet1D` output
   - Test Neumann recovery: α=0, verify matches `Neumann1D` output
   - Test validation: α=0, β=0 raises `ValueError`
   - Test JAX jit compatibility
   → verify: `uv run pytest tests/test_boundary.py -v` passes

---

## Sub-issue #107: High-Order Extrapolation Boundary Condition

**What:** Polynomial extrapolation of orders 1–5 using interior stencil points to fill ghost cells. Generalizes `Outflow1D` (order 0 = constant extrapolation).

### Extrapolation coefficients

For evenly-spaced points, extrapolation to the ghost cell using Lagrange interpolation from `n = order + 1` interior points. The ghost cell is one step beyond the boundary. Coefficients for each order:

| Order | Points | Coefficients (nearest → farthest) |
|-------|--------|-----------------------------------|
| 1     | 2      | [2, -1]                           |
| 2     | 3      | [3, -3, 1]                        |
| 3     | 4      | [4, -6, 4, -1]                    |
| 4     | 5      | [5, -10, 10, -5, 1]              |
| 5     | 6      | [6, -15, 20, -15, 6, -1]         |

These are `(-1)^(k+1) * C(n, k+1)` for k = 0..n-1 (binomial pattern with alternating signs).

### Steps

5. **Add `Extrapolation1D` class** to `bc_1d.py`
   - Fields: `face` (static), `order` (static, int 1–5)
   - Precompute coefficients as a tuple (static field) in `__init__`
   - `__call__`: index the correct interior points for the face and dot with coefficients
   - Validate: order must be in [1, 5]
   - Note: needs a helper `_interior_stencil(field, face, depth)` to grab multiple interior rows/cols
   → verify: check extrapolation of known linear/quadratic fields gives exact ghost values

6. **Add `Extrapolation1D` to the union type** `BoundaryCondition1D`
   → verify: type alias includes `Extrapolation1D`

7. **Export from `__init__.py`**
   - Add import and `__all__` entry for `Extrapolation1D`
   → verify: `from finitevolx import Extrapolation1D` works

8. **Unit tests** in `tests/test_boundary.py`
   - Test order 1: linear field → exact ghost value
   - Test order 2: quadratic field → exact ghost value
   - Test order 3: cubic field → exact ghost value
   - Test all four faces with order 1
   - Test invalid order raises `ValueError`
   - Test JAX jit compatibility
   → verify: `uv run pytest tests/test_boundary.py -v` passes

---

## Final Verification

9. **Full test suite**: `uv run pytest tests -v` — 0 failures
10. **Lint**: `uv run ruff check finitevolx/`
11. **Format**: `uv run ruff format --check finitevolx/`
12. **Type check**: `uv run ty check finitevolx` — no errors in changed files

---

## Files to modify

| File | Change |
|------|--------|
| `finitevolx/_src/bc_1d.py` | Add `Robin1D`, `Extrapolation1D` classes; update `BoundaryCondition1D` union |
| `finitevolx/__init__.py` | Import + export new classes |
| `tests/test_boundary.py` | Add test classes for both new BCs |

## Estimated effort

~0.5–1 day as noted in the issues. Both classes follow the established pattern closely.
