# Iterative Solvers & Preconditioners

Preconditioned Conjugate Gradient solver and preconditioner factories
for masked-domain elliptic problems.

## Conjugate Gradient

::: gridax_jax.solve_cg

::: gridax_jax.CGInfo

## Preconditioners

::: gridax_jax.make_spectral_preconditioner

::: gridax_jax.make_nystrom_preconditioner

::: gridax_jax.make_preconditioner

## Helmholtz operator as a gaussx operator

The rectangular and masked Helmholtz operators as structured gaussx linear
operators, for `gaussx.solve`, `gaussx.logdet`, `gaussx.sqrt` and SPDE
(Matérn) priors on the model grid.

::: gridax_jax.helmholtz_operator

::: gridax_jax.masked_helmholtz_operator
