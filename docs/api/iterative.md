# Iterative Solvers & Preconditioners

Preconditioned Conjugate Gradient solver and preconditioner factories
for masked-domain elliptic problems.

## Conjugate Gradient

!!! warning "Deprecated: CG is gaussx's"
    `solve_cg`, `CGInfo` and `make_nystrom_preconditioner` warn for one release (#474). Use `gaussx.linear_solve_with_info(gaussx.as_linear_operator(matvec, shape=(n, n), negative_definite=True), rhs.ravel(), solver=gaussx.CGSolver(rtol=..., atol=...), preconditioner=...)`, which returns `(x, gaussx.SolveInfo)` with the same `iterations`, `residual_norm` and `converged`, and `gaussx.NystromPreconditioner` on `gaussx.randomized_nystrom(operator, rank, key=key)`.

::: gridax_jax._src.solvers.iterative.solve_cg

::: gridax_jax._src.solvers.iterative.CGInfo

## Preconditioners

::: gridax_jax.make_spectral_preconditioner

::: gridax_jax._src.solvers.preconditioners.make_nystrom_preconditioner

::: gridax_jax.make_preconditioner

## Helmholtz operator as a gaussx operator

The rectangular and masked Helmholtz operators as structured gaussx linear
operators, for `gaussx.solve`, `gaussx.logdet`, `gaussx.sqrt` and SPDE
(Matérn) priors on the model grid.

::: gridax_jax.helmholtz_operator

::: gridax_jax.masked_helmholtz_operator

## T-point masked operators

The negative-definite T-point operators on a wet mask, and the Neumann gauge, that the pressure solve is built from: a zero-flux wall, a Dirichlet wall on the faces, and the removal of the wet mean on each connected wet component.

::: gridax_jax.masked_neumann_laplacian

::: gridax_jax.masked_face_dirichlet_laplacian

::: gridax_jax.wet_components

::: gridax_jax.project_zero_mean
