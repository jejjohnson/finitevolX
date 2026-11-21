# Multigrid Helmholtz Solver

Geometric multigrid solver for variable-coefficient Helmholtz equations
on masked Arakawa C-grids.

## Solver

::: gridax_jax.MultigridSolver

## Factory

::: gridax_jax.build_multigrid_solver

## Levels for gaussx

The V-cycle runs in gaussx (`gaussx.MultigridPreconditioner`, `gaussx.MultigridSolver`); finitevolX builds the levels from the mask (#474).

::: gridax_jax.multigrid_levels

## Deprecated preconditioner factories

`make_multigrid_preconditioner` and `make_gaussx_multigrid_preconditioner` warn for one release: use `build_multigrid_solver(...).as_preconditioner()`, a `gaussx.MultigridPreconditioner`.

::: gridax_jax._src.solvers.preconditioners.make_multigrid_preconditioner

::: gridax_jax._src.solvers.preconditioners.make_gaussx_multigrid_preconditioner
