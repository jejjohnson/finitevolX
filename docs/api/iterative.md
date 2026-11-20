# Iterative Solvers & Preconditioners

Preconditioned Conjugate Gradient solver and preconditioner factories
for masked-domain elliptic problems.

## Conjugate Gradient

::: gridax_ops.solve_cg

::: gridax_ops.CGInfo

## Preconditioners

::: gridax_ops.make_spectral_preconditioner

::: gridax_ops.make_nystrom_preconditioner

::: gridax_ops.make_preconditioner

## Helmholtz operator as a gaussx operator

The rectangular and masked Helmholtz operators as structured gaussx linear
operators, for `gaussx.solve`, `gaussx.logdet`, `gaussx.sqrt` and SPDE
(Matérn) priors on the model grid.

::: gridax_ops.helmholtz_operator

::: gridax_ops.masked_helmholtz_operator
