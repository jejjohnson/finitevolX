# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the module map,
the boundaries with spectraldiffx and gaussx, "reuse before you write", the
contracts, the tests that enforce them, commands, the pre-commit checklist,
the docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `finitevolx/` (flat layout; implementation in
  `finitevolx/_src/`, public API in `finitevolx.__all__`). Search
  [`docs/capabilities.md`](../docs/capabilities.md) before writing a helper:
  spectral solves and transforms come from spectraldiffx, tridiagonal solves
  and Nyström preconditioning from gaussx, CG from lineax, time stepping
  from diffrax.
- Keep the contracts in `AGENTS.md`:
  - **the C-grid**: every array is `[Ny, Nx]` with one ghost ring;
    `T[j, i]` centre, `U[j, i]` east face, `V[j, i]` north face, `X[j, i]`
    NE corner; backward operators read the BC-owned west / south ghost;
  - **operators** write only `[1:-1, 1:-1]` via `interior(...)`
    (advection `[2:-2, 2:-2]` with closed walls); Layer 1 raw stencils,
    Layer 2 functional helpers (mask-free), Layer 3 `eqx.Module` classes
    holding `grid` and `mask`;
  - **masks** are built once with `Mask2D.from_mask` (numpy / scipy);
    outputs are zeroed on the dry points of their stagger; `mask=None` is
    bit-identical to before; `if mask is not None:`, never `if mask:`;
  - **units**: physical constants only as parameter defaults, so operators
    stay scale-covariant (`tests/test_scale_covariance.py`);
  - pure `jnp` slicing (no `finitediffx` / `kernex`); `eqx.Module`, not
    dataclasses; dtype follows the input.
- NumPy-style docstrings; every stencil commented with its half-index
  formula (`dh_dx[j, i+1/2] = ...`) and array shapes (`[Ny, Nx]`).
- Before committing, from the repo root: `make test`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run ty check finitevolx`;
  `make capabilities` after a public API change; regenerate the golden
  fixtures (`uv run python tests/fixtures/_gen_golden.py`) only when an
  operator's output changes on purpose.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
