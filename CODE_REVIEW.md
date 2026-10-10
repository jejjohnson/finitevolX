# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. finitevolX is finite-volume operators on Arakawa C-grids in JAX:
most defects worth finding are about **the grid and the numerics** (a slice
off by half a cell, a ghost cell read before any BC fills it, an advection
write region widened onto ghost fluxes, a mask that leaks `NaN` or changes
the unmasked result, an SI constant baked into an operator, a solver that
returns unconverged) or **reuse** (a hand-rolled stencil, spectral solve,
tridiagonal solve or CG loop that already exists), not style. Read
"Reuse before you write" and "The contracts" in [`AGENTS.md`](AGENTS.md)
first; this file is the checklist and the report format.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify master >/dev/null 2>&1 && echo master || echo main)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import
order); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against
  [`docs/capabilities.md`](docs/capabilities.md) (finitevolX, spectraldiffx,
  gaussx). A re-implemented stencil, interpolation, reconstruction, limiter,
  BC, spectral solve, eigenvalue formula, DCT / DST, tridiagonal solve, CG
  loop or preconditioner is a **High** finding, with the existing name to
  use.
- Stencils use pure `jnp` slicing or the raw stencils (`diff_x_fwd`,
  `avg_xy_fwd`, …) and `interior`; no `finitediffx` / `kernex`.
- spectraldiffx and gaussx through their public APIs; NumPy / SciPy only at
  construction time, on concrete values.

### 2. The C-grid

- Every slice matches the half-index formula in its comment and the
  staggering in its name (T→U forward reads `[1:-1, 2:]`, U→T backward reads
  `[1:-1, :-2]`); the comment is there.
- Output written only on `[1:-1, 1:-1]` (advection: `[2:-2, 2:-2]` with
  closed walls, `[1:-1, 2:-2, 2:-2]` in 3-D); the ghost ring is zero.
- Ghost cells read only where the contract says the caller fills them; a
  chained operator that reads a west / south ghost no forward operator sets
  is documented, or the BC is applied.
- 3-D operators leave the `k = 0` / `k = Nz-1` slabs alone; multilayer code
  uses `multilayer`, not a `3D` operator.
- Spherical operators use the `SphericalGrid*` metric and guard `cos(lat)`
  near the poles.

### 3. Masks

- A class operator's mask zeroes the output on the dry points of the
  **output** stagger (`h` / `u` / `v` / `xy_corner_strict`); `mask=None`
  leaves the unmasked result bit for bit unchanged.
- Inputs that may hold `NaN` on land are zeroed with `jnp.where` before a
  stencil reads them, and outputs are masked with `jnp.where`, not a float
  multiply.
- `if mask is not None:`, never `if mask:`; mask construction stays eager
  (numpy / scipy), and no irregular-boundary index array is used under
  `jit`.
- The operator has golden entries (unmasked, masked) in
  `tests/fixtures/_gen_golden.py` and a `tests/test_*_masks.py` test.

### 4. Boundary conditions and solvers

- A BC atom has a static `face`, sets one ghost row / column so the
  condition holds at the face, and joins the `BoundaryCondition1D` union.
- Elliptic solves: the Helmholtz sign convention (negative semidefinite);
  convergence reported (`CGInfo.converged`) and checked in tests;
  geometry-only precomputation built once; randomised construction takes a
  `key`.
- Spectral solves and transforms come from spectraldiffx; nothing
  re-derives their eigenvalues.

### 5. Time stepping, forcing and units

- Step functions are pure on pytrees; diffrax solvers declare their tableau
  / order; forcings subclass `AbstractForcing`, stay time-agnostic and
  compose inside the RHS.
- No physical constant (`GRAVITY`, `R_EARTH`, `OMEGA`, `RHO`, `DEG2M`)
  inside an expression in `operators/`, `advection/`, `diffusion/` or
  `solvers/`; no absolute epsilon guarding a ratio of dimensional
  quantities. A new spatial operator gets a `Case` in
  `tests/test_scale_covariance.py`.

### 6. JAX numerics

- No Python control flow on traced values; options that select a code path
  are static.
- Dtypes follow the input (no `jnp.zeros(shape)` returned without the
  input's dtype; `interior` already handles it).
- Divisions and square roots that can reach zero use the double-`where`
  pattern so gradients stay finite; non-smooth functions on a
  differentiated path use `smooth_abs` / `smooth_max` / `smooth_clamp`.
- Conservation: a flux-form operator keeps the flux single-valued at every
  face, so the tendency sums to the boundary flux (zero with closed walls).

### 7. Public API and documentation

- New names: imported and listed in `finitevolx.__all__` in their logical
  group, a `::: finitevolx.<Name>` entry on the `docs/api/` page,
  `docs/capabilities.md` regenerated.
- NumPy-style docstrings with the formula, shapes and an `Examples` section
  that actually runs (doctests are not collected, so run it).
- A new theory section or notebook is in the `mkdocs.yml` nav, and
  `uv run mkdocs build` adds no `WARNING` lines (`--strict` does not pass
  yet: the existing docstrings raise griffe warnings).

### 8. Tests

- Analytic truth (constant → 0, linear → constant, a known solution), a
  convergence order, or a conservation sum; not just "runs".
- Small grids; random inputs from a pinned key; a non-obvious tolerance
  says where it comes from.
- A module that needs float64 enables x64 itself; goldens regenerated only
  for an intended change, with the diff explained.

### 9. Modern Python and dependencies

- Type hints on every public function; `X | None`; specific exceptions with
  `raise ... from ...`; guard clauses over deep nesting.
- `eqx.Module` (not a dataclass) for anything holding arrays.
- No new runtime dependency without discussion; `uv.lock` updated.

---

## finitevolX-Specific Checks

### The slice must match the staggering

```python
# ❌ Labelled T → U (east face), but this is the backward (west-face) difference
dh_dx = jnp.zeros_like(h).at[1:-1, 1:-1].set((h[1:-1, 1:-1] - h[1:-1, :-2]) / dx)

# ✅ dh_dx[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx — equal to Difference2D.diff_x_T_to_U
dh_dx = fvx.interior(fvx.diff_x_fwd(h) / dx, h)
```

### NaN on land: sanitise the input, not only the output

```python
# ❌ The value is right, but the gradient is NaN at dry points
def ke(u):
    return jnp.sum(jnp.where(wet, 0.5 * u**2, 0.0))


# ✅ Zero dry inputs first (as LinearDrag2D does), then mask the output
def ke(u):
    u = jnp.where(wet, u, 0.0)
    return jnp.sum(jnp.where(wet, 0.5 * u**2, 0.0))
```

### Optional masks

```python
# ❌ Raises ValueError for an array mask (its truth value is ambiguous)
def apply_mask(out, mask=None):
    if mask:
        out = out * mask
    return out


# ✅
def apply_mask(out, mask=None):
    if mask is not None:
        out = out * mask
    return out
```

### Constants are defaults, not literals

```python
# ❌ Bakes SI units into the operator (fails TestConstantsAudit)
def geostrophic_u(dh_dy, f):
    return -GRAVITY / f * dh_dy


# ✅ Overridable, so a nondimensional model can pass gravity=1
def geostrophic_u(dh_dy, f, gravity: float = GRAVITY):
    return -gravity / f * dh_dy
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
