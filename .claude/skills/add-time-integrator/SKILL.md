---
name: add-time-integrator
description: Add a time-integration scheme to finitevolX — a pure functional step (explicit RK, multistep, IMEX, split-explicit, semi-Lagrangian) and its diffrax solver form (a Butcher-tableau AbstractERK, an AbstractSolver, or an eqx.Module with init / step for schemes that do not fit diffeqsolve) — with convergence-order tests and docs. Use when asked to add, port or implement a time stepper, Runge–Kutta scheme, multistep / leapfrog scheme, IMEX or split-explicit integrator in finitevolx/_src/timestepping.
---

# Add a time integrator

Read contract 6 (time integration) in `AGENTS.md`, then
`docs/time_integration.md` and `docs/time_integration_usage.md`.

## 1. Make sure it does not exist yet

Search `docs/capabilities.md`: the steps `euler_step`, `heun_step`,
`rk3_ssp_step`, `rk4_step`, `ab2_step`, `ab3_step`, `leapfrog_raf_step`,
`imex_ssp2_step`, `split_explicit_step`, `semi_lagrangian_step`; the
diffrax solvers `ForwardEulerDfx`, `RK2Heun`, `RK3SSP`, `RK4Classic`,
`SSP_RK2`, `SSP_RK104`, `IMEX_SSP2`, `SemiLagrangianSolver`; the
`eqx.Module` solvers `AB2Solver`, `LeapfrogRAFSolver`,
`SplitExplicitRKSolver`; and `solve_ocean_pde`. diffrax itself ships many
more (`dfx.Tsit5`, `dfx.Dopri5`, `dfx.Kvaerno5`, …): a generic ODE scheme
belongs there, not here. finitevolX adds the schemes ocean models need
(SSP, multistep with filters, IMEX, split-explicit).

## 2. Write it (`finitevolx/_src/timestepping/`)

- **The pure step** in the matching module (`explicit_rk.py`,
  `multistep.py`, `imex.py`, `split_explicit.py`, `semi_lagrangian.py`):
  works on any pytree state through `jax.tree.map`; `rhs_fn(state) ->
  tendency`; returns the new state (plus whatever history the next step
  needs). No Python control flow on traced values; no state outside the
  arguments.
- **The diffrax form** in `diffrax_solvers.py`:
  - explicit Runge–Kutta → subclass `dfx.AbstractERK` with a class-level
    `tableau: ClassVar[dfx.ButcherTableau]` (`c`, `b_sol`, `b_error`,
    `a_lower`) and an `interpolation_cls`, as `RK3SSP` does; write the
    tableau in the docstring;
  - a one-off scheme → implement `dfx.AbstractSolver` (`init`, `step`,
    `func`, `order`, `term_structure`, `interpolation_cls`), as
    `ForwardEulerDfx` does;
  - a scheme with history diffrax cannot carry → an `eqx.Module` with
    `init(rhs_fn, t0, y0, dt) -> (solver, y1)` and
    `step(rhs_fn, t, y, dt) -> (y_next, solver)`, updating the history with
    `eqx.tree_at`, as `AB2Solver` does.
- Docstring: the update formula, order, stages, stability / SSP
  coefficient, and a reference.

## 3. Export and document

`finitevolx/__init__.py` (import + `__all__` under `# Time integration —
pure functional` or `# Time integration — diffrax solvers`),
`::: finitevolx.NewScheme` on `docs/api/time_integration.md`, the scheme on
`docs/time_integration.md` and a usage snippet on
`docs/time_integration_usage.md`; `make capabilities`.

## 4. Tests

- `tests/test_timestepping.py`: add the step to the
  `TestExplicitRKConvergence` / `TestExplicitRKAbsoluteAccuracy`
  parametrisations (`dy/dt = -y`, observed order by halving `dt`), pytree
  states, `jit`, a zero RHS; and a `TestPureFunctionalVsDiffrax` case
  matching the diffrax form.
- `tests/test_diffrax_solvers.py`: the solver in the
  `TestDiffraxExplicitRK` / `TestDiffraxConvergence` /
  `TestDiffraxAbsoluteAccuracy` parametrisations, or a class of its own
  (as `TestAB2Solver`); `SaveAt` and `solve_ocean_pde` if relevant.
- IMEX / split-explicit / semi-Lagrangian: `tests/test_timestepping_imex.py`,
  `tests/test_semi_lagrangian.py`.
- State the expected order and the tolerance's origin in a comment (e.g.
  "observed order above expected − 0.15 between the two finest steps", as
  `TestExplicitRKConvergence` asserts).

## 5. Verify

```bash
uv run pytest -n auto tests/test_timestepping.py tests/test_timestepping_imex.py \
  tests/test_diffrax_solvers.py tests/test_semi_lagrangian.py tests/test_capabilities.py
```

Then the `pre-pr-check` skill.
