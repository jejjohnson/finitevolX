---
name: add-elliptic-solver
description: Add or extend an elliptic (Poisson / Helmholtz) solver path in finitevolX — a convenience wrapper like streamfunction_from_vorticity / pv_inversion, a new method= option on the wrappers, a preconditioner for solve_cg, a multigrid option, or known-value / boundary-condition handling — while delegating spectral solves to spectraldiffx and structured linear algebra to gaussx and lineax. Use when asked to add, port or fix a Poisson, Helmholtz, PV-inversion, pressure or streamfunction solve, a CG preconditioner or a multigrid feature in finitevolx/_src/solvers.
---

# Add an elliptic solver path

Read contract 5 (solvers) in `AGENTS.md`, then `docs/elliptic_solvers.md`,
`docs/elliptic_solvers_usage.md` and `docs/solver_comparison.md`.

## 1. Decide where it belongs

| You need… | It lives in |
|---|---|
| A rectangular spectral solve, a new transform type, new eigenvalues, a capacitance-method change | **spectraldiffx** (finitevolX re-exports it from `solvers/spectral.py` / `spectral_transforms.py`): propose it upstream, then bump the pin (`bump-upstream-pins` skill) |
| A tridiagonal / Kronecker / low-rank solve, a Nyström-style preconditioner | **gaussx** (finitevolX re-exports or wraps it: `solvers/tridiagonal.py`, `make_nystrom_preconditioner`) |
| A masked-domain solve on the C-grid, a preconditioner adapted to the finitevolX sign convention, a one-call wrapper, multigrid | here, in `finitevolx/_src/solvers/` |

Search `docs/capabilities.md` first: the wrappers
(`streamfunction_from_vorticity`, `pressure_from_divergence`,
`pv_inversion`), `solve_cg` + `masked_laplacian`, the
`make_*_preconditioner` family and `make_preconditioner`,
`build_capacitance_solver`, `build_multigrid_solver` / `MultigridSolver`,
the lifting tools `boundary_ring` / `SolveDomain` / `KnownValueLifting`.

## 2. Write it (`finitevolx/_src/solvers/`)

- **Sign convention.** The operator is `(∇² − λ)` with `λ ≥ 0`: negative
  (semi)definite. A matvec for `solve_cg` implements that operator; a
  preconditioner approximates its negative inverse (`solve_cg` negates it
  for lineax; `make_nystrom_preconditioner` probes `−A` for gaussx).
- **A preconditioner** goes in `preconditioners.py`, returns
  `M_inv(r: [Ny, Nx]) -> [Ny, Nx]`, and gets a `kind=` branch (and its
  required-argument check) in `make_preconditioner`. Randomised ones take a
  `key`.
- **A wrapper method** goes through `_solve_dispatch` in `elliptic.py`
  (`"spectral"`, `"cg"`, `"capacitance"`; `_METHODS` lists them). Keep the
  wrappers' shared arguments (`bc`, `lambda_`, `mask`, `known_values`,
  `known_mask`), and route a `BoundaryConditionSet` through `_resolve_bc`.
- **Masks**: accept a `Mask2D` or an array; a `Mask2D` contributes the
  mask of the wrapper's staggering — `xy_corner_strict` for ψ (`"X"`),
  `h` for pressure (`"T"`) — cast to the rhs dtype (`_resolve_mask_arr`).
- **Wall placement**: ψ = 0 (or a known value) sits on the coast ring, the
  dry cells 4-adjacent to the wet mask — what `masked_laplacian` imposes;
  for ψ on a rectangle, the X wall nodes `0` and `N-2`.  A spectral path
  slices the unknowns of its staggering out of the full array
  (`_SPECTRAL_STAGGERED`, `_solve_spectral`) and must converge at second
  order against a manufactured solution
  (`tests/test_solver_wrappers.py::TestWallPlacement`).
- **Precompute** geometry once, with NumPy on concrete values
  (`build_*` functions); the returned object is an `eqx.Module` that runs
  under `jit` / `vmap` / `grad`.
- **Convergence**: iterative paths return or check `CGInfo.converged`;
  never a Python loop on a traced residual (lineax's CG iterates in a
  `lax.while_loop`; `MultigridSolver` runs a fixed number of V-cycles in a
  `lax.fori_loop`).
- No physical constants inside expressions (`tests/test_scale_covariance.py`
  audits `solvers/`): a Poisson solve scales as `L²`.

## 3. Export and document

`finitevolx/__init__.py` (import + `__all__` in the solver groups), the
`docs/api/` page (`elliptic.md`, `iterative.md`, `capacitance.md`,
`multigrid.md`, `spectral_solvers.md`), the usage guide and, for a new
solver type, a row in `docs/solver_comparison.md`; `make capabilities`.

## 4. Tests

- Against an analytic solution (a sine mode of the Dirichlet Laplacian, a
  cosine mode for Neumann) or a dense reference on a small grid; residual
  of the discrete operator.
- `tests/test_elliptic.py` (CG, preconditioners — fewer iterations than no
  preconditioner, `TestNystromPreconditioner` for keyed construction —
  masked Laplacian, capacitance, `vmap`), `tests/test_solver_wrappers.py`
  (each `method=`, its required-argument errors),
  `tests/test_elliptic_bcset.py`, `tests/test_elliptic_known_values.py`,
  `tests/test_inhomogeneous.py`, `tests/test_multigrid.py` (V-cycle
  convergence, preconditioner, JAX compatibility, implicit
  differentiation).
- A scale-covariance `Case` (order −2) in `tests/test_scale_covariance.py`
  for a new solve entry point.

## 5. Verify

```bash
uv run pytest -n 2 tests/test_elliptic.py tests/test_elliptic_bcset.py \
  tests/test_elliptic_known_values.py tests/test_solver_wrappers.py \
  tests/test_inhomogeneous.py tests/test_multigrid.py tests/test_tridiagonal.py \
  tests/test_scale_covariance.py tests/test_capabilities.py
```

Then the `pre-pr-check` skill.
