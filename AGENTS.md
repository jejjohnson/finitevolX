# Agent Guidelines

This file contains standing instructions for **all** coding agents working on this repository (Copilot, Claude, Gemini, etc.).

---

## Karpathy Coding Principles

Four behavioral principles to reduce the most common LLM coding mistakes. These bias toward caution over speed — for trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State assumptions explicitly. If uncertain, ask before writing code.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Test: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into declarative goals with verification:

- "Add validation" → Write tests for invalid inputs, then make them pass
- "Fix the bug" → Write a test that reproduces it, then make it pass
- "Optimize X" → Write the naive correct version first, then optimize while preserving correctness

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Before Every Commit

**All agents must** verify that every one of the following passes before creating a commit or reporting progress. No exceptions.

1. **Tests** – `uv run pytest tests -v` (or `make test`) must have 0 failures.
2. **Lint** – `uv run ruff check .` (or `make uv-lint`) must report no issues.
3. **Format** – `uv run ruff format --check .` must report no files to reformat.
4. **Type checks** – `uv run ty check finitevolx` must report no errors in changed files.

## Development Environment

**IMPORTANT**: Always use `uv run` when running Python tools or scripts (e.g., `pytest`, `python`, `ruff`, `ty`, `mkdocs`, `pre-commit`) so they run in the project environment. You do **not** need `uv run` for non-Python shell commands (e.g., `git`, `ls`, `cat`). Do NOT use the system Python directly.

## Pull Request Descriptions

**Never replace or remove an existing PR title or description.** When reporting progress on a PR that already has a title and description, only append new checklist items or update the status of existing ones. The original content must be preserved in full.

This is a common failure mode: an agent called to make a small follow-up change will supply a fresh description scoped only to its own work, silently discarding all prior context. Always read the existing description first and treat it as the base.

## GIT Safety Rules

- **NEVER** push to `main`/`master` or merge into `main`/`master` unless the user explicitly says "push to main" or "merge to main".
- **NEVER** push to any remote branch or run `git push` unless the user explicitly asks you to push. Only commit locally.
- Always work on feature branches.
- When the user says "merge the changes" or "merge the branch", they mean push the local branch to the remote — NOT merge into main.
- Always confirm before any action that affects shared branches (main, master, production, etc.).

## Documentation

This repo uses **MkDocs + Material + mkdocstrings + mkdocs-jupyter** for documentation.

- **Build locally**: `make docs-serve` (or `uv run mkdocs serve`)
- **Build static site**: `make docs` (or `uv run mkdocs build`)
- **Deploy to GitHub Pages**: `make docs-deploy` (or `uv run mkdocs gh-deploy --force`)
- **Auto-deploy**: the `pages.yml` workflow deploys automatically on every push to `main`/`master`

When writing docstrings, use **NumPy style** (finitevolX convention).

### Documentation Style for finitevolX

* Use numpy-style docstrings for all functions and classes.
* Be pedantic and pedagogical — every stencil must have a comment showing the half-index formula.
* Use ascii math notation (e.g. `dh_dx[j, i+1/2] = ...`).
* Track array shapes in comments (e.g. `[Ny, Nx]`).

Notebooks in `notebooks/` may be stored as `.ipynb` files.

## Packages

* `equinox` for the dataclasses / `eqx.Module` base class.
* `jaxtyping` for the type annotations.
* `jax` / `jax.numpy` for the computations.
* **Do NOT depend on `finitediffx` or `kernex`** — use pure `jnp` slicing.

## Grid Conventions (finitevolX-specific)

* All arrays have total shape `[Ny, Nx]`.  The physical interior is `(Ny-2) x (Nx-2)`;
  one ghost-cell ring on each side is reserved for boundary conditions.
* Same-index colocation:
  - `T[j, i]`  → cell centre  `(j,     i    )`
  - `U[j, i]`  → east face    `(j,     i+1/2)`
  - `V[j, i]`  → north face   `(j+1/2, i    )`
  - `X[j, i]`  → NE corner    `(j+1/2, i+1/2)`

### Ghost cells per staggered type

All four types share the same `[Ny, Nx]` shape, but the **meaning** of the
ghost ring differs.  The table below uses "BC-owned" for cells that must be
filled by the caller before running a chained operator, and "outside domain"
for cells that have no physical meaning.

| Type | West ghost `[:,0]` | East ghost `[:,Nx-1]` | South ghost `[0,:]` | North ghost `[Ny-1,:]` |
|------|--------------------|-----------------------|---------------------|------------------------|
| T    | west BC T-cell     | east BC T-cell        | south BC T-cell     | north BC T-cell        |
| U    | west boundary face (BC-owned) | outside domain | south ghost U-row (BC-owned) | north ghost U-row (BC-owned) |
| V    | west ghost V-col (BC-owned)  | east ghost V-col (BC-owned) | south boundary face (BC-owned) | outside domain |
| X    | west ghost X-col (BC-owned)  | outside domain        | south ghost X-row (BC-owned) | outside domain |

**Key asymmetry** (same-index convention):

* **Forward operators** (T→U, T→V, T→X) write to `[1:-1, 1:-1]`.  The last
  interior column/row **does** use the `+direction` ghost of the *source*
  array (east ghost T for T→U, north ghost T for T→V, NE corner ghost for
  T→X) because that face lies inside the `[1:-1, 1:-1]` output range.
* **Backward operators** (U→T, V→T, X→T) write to `[1:-1, 1:-1]`.  The
  first interior column/row reads the `-direction` ghost of the *source*
  array (west ghost U-face for U→T, south ghost V-face for V→T).  This ghost
  is **not** set by any forward operator — the caller must supply it via BC.
* **Cross operators** (U→V, V→U, U→X, V→X, X→U, X→V) follow the same
  pattern: the last interior output reads the "far side" ghost of the source.

### Stencil slice reference

| Slice pattern          | Meaning (j = row, i = col)          |
|------------------------|-------------------------------------|
| `arr[1:-1, 1:-1]`      | interior at `(j, i)`                |
| `arr[1:-1, 2:]`        | interior rows, one step east `i+1`  |
| `arr[1:-1, :-2]`       | interior rows, one step west `i-1`  |
| `arr[2:, 1:-1]`        | interior cols, one step north `j+1` |
| `arr[:-2, 1:-1]`       | interior cols, one step south `j-1` |

Every shifted slice has the **same shape** as `[1:-1, 1:-1]` because one
ghost-cell ring provides exactly one step in any direction.

### Forward difference stencils (T → face / corner)

```
# diff_x_T_to_U:  dh[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx
out.at[1:-1, 1:-1].set((h[1:-1, 2:] - h[1:-1, 1:-1]) / dx)

# diff_y_T_to_V:  dh[j+1/2, i] = (h[j+1, i] - h[j, i]) / dy
out.at[1:-1, 1:-1].set((h[2:, 1:-1] - h[1:-1, 1:-1]) / dy)

# diff_y_U_to_X:  du[j+1/2, i+1/2] = (u[j+1, i] - u[j, i]) / dy
out.at[1:-1, 1:-1].set((u[2:, 1:-1] - u[1:-1, 1:-1]) / dy)

# diff_x_V_to_X:  dv[j+1/2, i+1/2] = (v[j, i+1] - v[j, i]) / dx
out.at[1:-1, 1:-1].set((v[1:-1, 2:] - v[1:-1, 1:-1]) / dx)
```

### Backward difference stencils (face / corner → T)

```
# diff_x_U_to_T:  du[j, i] = (u[j, i] - u[j, i-1]) / dx
out.at[1:-1, 1:-1].set((u[1:-1, 1:-1] - u[1:-1, :-2]) / dx)

# diff_y_V_to_T:  dv[j, i] = (v[j, i] - v[j-1, i]) / dy
out.at[1:-1, 1:-1].set((v[1:-1, 1:-1] - v[:-2, 1:-1]) / dy)

# diff_y_X_to_U:  dq[j, i+1/2] = (q[j, i] - q[j-1, i]) / dy
out.at[1:-1, 1:-1].set((q[1:-1, 1:-1] - q[:-2, 1:-1]) / dy)

# diff_x_X_to_V:  dq[j+1/2, i] = (q[j, i] - q[j, i-1]) / dx
out.at[1:-1, 1:-1].set((q[1:-1, 1:-1] - q[1:-1, :-2]) / dx)
```

### Ghost-cell interaction at stencil boundaries

Ghost cells on the **`+` side** (east, north) of staggered arrays are
computed by forward operators using BC-owned ghost T-cells.  Ghost cells on
the **`-` side** (west, south) must be set by the caller (via BC layer)
before chained backward-diff or interpolation calls.

Without BCs, `diff_x_U_to_T` at `i=1` sees a zero west ghost U-face and
produces `(u[j,1] - 0) / dx` instead of the correct value.  This is the
**expected** behaviour — not a bug.  Apply BCs to intermediate fields before
chaining operators.

### Advection divergence

Advection operators (`Advection1D`, `Advection2D`) write tendencies to
`[2:-2]` / `[2:-2, 2:-2]` (not `[1:-1]`) to avoid reading ghost flux cells.
`Advection3D` uses `[1:-1, 2:-2, 2:-2]` — all z-levels are independent, so
the full z-interior is valid, but the horizontal plane follows the same
`[2:-2]` rule.

## Operator Idiom (finitevolX-specific)

* Every operator writes **only** into `[1:-1, 1:-1]` of the output array
  (or `[2:-2]`/`[2:-2, 2:-2]` for advection — see above).
* Initialise output with `jnp.zeros_like(input)` then use `.at[1:-1, 1:-1].set(...)`.
* The caller is responsible for boundary conditions (pad, enforce_periodic, etc.).

## Masks (finitevolX-specific)

* Use `Mask2D.from_mask(h)` to derive all staggered masks from a binary h-grid mask.
* Construction uses **numpy / scipy** (masks are built once, not traced through JAX JIT).
* When branching on an optional mask argument, always use `if mask is not None:` — never bare `if mask:`, which raises `ValueError` for JAX arrays.

## Testing

* One test per operator per dimension.
* Use fixtures for different grid sizes.
* Tests live in `tests/` at the repo root.

## Commit Messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Examples:
- `feat: add support for 3D vorticity calculations`
- `fix: correct boundary handling in upwind reconstruction`
- `docs: update installation instructions`
- `chore: update dependencies`
