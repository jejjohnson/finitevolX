# Iterative Solvers & Preconditioners

The spectral preconditioner and the masked operators for masked-domain elliptic problems.

## Conjugate Gradient

CG is gaussx's: `gaussx.linear_solve_with_info(gaussx.as_linear_operator(matvec, in_structure=jax.ShapeDtypeStruct(b.shape, b.dtype), negative_definite=True), b, solver=gaussx.CGSolver(rtol=..., atol=..., max_steps=...), preconditioner=lambda r: -M(r))` returns `(x, info)` with `info.iterations`, `info.residual_norm` and `info.converged`, and `gaussx.NystromPreconditioner` on `gaussx.randomized_nystrom(operator, rank, key=key)` is the low-rank preconditioner.

## Preconditioners

::: gridax.solvers.make_spectral_preconditioner

## Helmholtz operator as a gaussx operator

The rectangular and masked Helmholtz operators as structured gaussx linear
operators, for `gaussx.solve`, `gaussx.logdet`, `gaussx.sqrt` and SPDE
(Matérn) priors on the model grid.

::: gridax.solvers.helmholtz_operator

::: gridax.solvers.masked_helmholtz_operator

## T-point masked operators

The negative-definite T-point operators on a wet mask, and the Neumann gauge, that the pressure solve is built from: a zero-flux wall, a Dirichlet wall on the faces, and the removal of the wet mean on each connected wet component.

::: gridax.solvers.masked_neumann_laplacian

::: gridax.solvers.masked_face_dirichlet_laplacian

::: gridax.solvers.wet_components

::: gridax.solvers.project_zero_mean
