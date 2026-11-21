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

## Preconditioner

`build_multigrid_solver(...).as_preconditioner()` is a `gaussx.MultigridPreconditioner`, for `gaussx.CGSolver(preconditioner=...)`.
