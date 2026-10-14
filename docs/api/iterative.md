# Iterative Solvers & Preconditioners

Preconditioned Conjugate Gradient solver and preconditioner factories
for masked-domain elliptic problems.

## Conjugate Gradient

::: finitevolx.solve_cg

::: finitevolx.CGInfo

## Preconditioners

::: finitevolx.make_spectral_preconditioner

::: finitevolx.make_nystrom_preconditioner

::: finitevolx.make_preconditioner

## Helmholtz operator as a gaussx operator

The rectangular and masked Helmholtz operators as structured gaussx linear
operators, for `gaussx.solve`, `gaussx.logdet`, `gaussx.sqrt` and SPDE
(Matérn) priors on the model grid.

::: finitevolx.helmholtz_operator

::: finitevolx.masked_helmholtz_operator
