---
name: add-elliptic-solver
description: Add or extend an elliptic (Poisson / Helmholtz) solver path in finitevolX — a spectral or multigrid preconditioner for gaussx's CG, a multigrid option, or known-value / boundary-condition handling — while delegating spectral solves to spectraldiffx and structured linear algebra and CG to gaussx. Use when asked to add, port or fix a Poisson, Helmholtz, PV-inversion, pressure or streamfunction solve, a CG preconditioner or a multigrid feature in gridax/_src/solvers.
---

# Add an elliptic solver path

Read contract 5 (solvers) in `AGENTS.md`, then `docs/elliptic_solvers.md`,
`docs/elliptic_solvers_usage.md` and `docs/solver_comparison.md`.

## 1. Decide where it belongs

| You need… | It lives in |
|---|---|
| A rectangular spectral solve, a new transform type, new eigenvalues, a capacitance-method change | **spectraldiffx** (finitevolX re-exports it from `solvers/spectral.py` / `spectral_transforms.py`): propose it upstream, then bump the pin (`bump-upstream-pins` skill) |
| A tridiagonal / Kronecker / low-rank solve, a Nyström-style preconditioner | **gaussx** (finitevolX re-exports `solvers/tridiagonal.py`; CG is `gaussx.linear_solve_with_info` with `gaussx.CGSolver`, and the Nyström preconditioner is `gaussx.NystromPreconditioner`) |
| A masked-domain solve on the C-grid, a preconditioner adapted to the finitevolX sign convention, multigrid | here, in `gridax/_src/solvers/` (public: `gridax.solvers`) |
| A one-call inversion (`streamfunction_from_vorticity`, `pressure_from_divergence`, `pv_inversion`), a new `method=` option on them, known-value or `BoundaryConditionSet` handling for them | **sesmax-core** (`sesmax_core.gfd`, #475; gridax keeps no alias): propose it there, built from the public pieces here |

Search `docs/capabilities.md` first: `masked_laplacian` (and the
T-point `masked_neumann_laplacian`, `masked_face_dirichlet_laplacian`,
`wet_components`, `project_zero_mean`), `make_spectral_preconditioner`,
`build_capacitance_solver`, `build_multigrid_solver` / `MultigridSolver`,
the lifting tools `boundary_ring` / `SolveDomain` / `KnownValueLifting`.

## 2. Write it (`gridax/_src/solvers/`)

- **Sign convention.** The operator is `(∇² − λ)` with `λ ≥ 0`: negative
  (semi)definite. A matvec for `gaussx.CGSolver` implements that operator; a
  preconditioner approximates its negative inverse, and gaussx solves
  `(−A) x = −b`, so the caller passes `lambda r: −M(r)`.
- **A preconditioner** goes in `preconditioners.py` and returns
  `M_inv(r: [Ny, Nx]) -> [Ny, Nx]`. Randomised ones take a `key`; a generic
  one belongs in gaussx.
- **The one-call wrappers** live in sesmax-core now (#475); gridax keeps no copy of them.
- **Masks**: accept a `Mask2D` or an array; a `Mask2D` contributes the
  mask of the field's staggering — `xy_corner_strict` for ψ (`"X"`),
  `h` for pressure (`"T"`) — cast to the rhs dtype (`_resolve_mask_arr`).
- **Wall placement**: ψ = 0 (or a known value) sits on the coast ring, the
  dry cells 4-adjacent to the wet mask — what `masked_laplacian` imposes;
  for ψ on a rectangle, the X wall nodes `0` and `N-2`.  A spectral path
  slices the unknowns of its staggering out of the full array
  (`_SPECTRAL_STAGGERED`, `_solve_spectral`) and must converge at second
  order against a manufactured solution (as the sesmax-core wrapper tests
  do).
- **Precompute** geometry once, with NumPy on concrete values
  (`build_*` functions); the returned object is an `eqx.Module` that runs
  under `jit` / `vmap` / `grad`.
- **Convergence**: iterative paths return or check `info.converged` (`gaussx.linear_solve_with_info`);
  never a Python loop on a traced residual (gaussx's CG iterates in a
  `lax.while_loop`; `MultigridSolver` runs a fixed number of V-cycles in a
  `lax.fori_loop`).
- No physical constants inside expressions (`tests/test_scale_covariance.py`
  audits `solvers/`): a Poisson solve scales as `L²`.

## 3. Export and document

`packages/gridax/src/gridax/solvers/__init__.py` (import + the name in the sorted `__all__`), the
`docs/api/` page (`elliptic.md`, `iterative.md`, `capacitance.md`,
`multigrid.md`, `spectral_solvers.md`), the usage guide and, for a new
solver type, a row in `docs/solver_comparison.md`; `make capabilities`.

## 4. Tests

- Against an analytic solution (a sine mode of the Dirichlet Laplacian, a
  cosine mode for Neumann) or a dense reference on a small grid; residual
  of the discrete operator.
- `tests/test_elliptic.py` (the spectral preconditioner,
  masked Laplacian, capacitance, `vmap`),
  `tests/test_inhomogeneous.py`, `tests/test_multigrid.py` (V-cycle
  convergence, preconditioner, JAX compatibility, implicit
  differentiation).
- A scale-covariance `Case` (order −2) in `tests/test_scale_covariance.py`
  for a new solve entry point.

## 5. Verify

```bash
uv run pytest -n 2 tests/test_elliptic.py \
  tests/test_inhomogeneous.py tests/test_multigrid.py tests/test_tridiagonal.py \
  tests/test_scale_covariance.py tests/test_capabilities.py
```

Then the `pre-pr-check` skill.
