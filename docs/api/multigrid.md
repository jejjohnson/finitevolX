# Multigrid Helmholtz Solver

Geometric multigrid solver for variable-coefficient Helmholtz equations
on masked Arakawa C-grids.

## Solver

::: gridax.solvers.MultigridSolver

## Factory

::: gridax.solvers.build_multigrid_solver

## Levels for gaussx

The V-cycle runs in gaussx (`gaussx.MultigridPreconditioner`, `gaussx.MultigridSolver`); finitevolX builds the levels from the mask (#474).

::: gridax.solvers.multigrid_levels

## Preconditioner

`build_multigrid_solver(...).as_preconditioner()` is a `gaussx.MultigridPreconditioner`, for `gaussx.CGSolver(preconditioner=...)`.
