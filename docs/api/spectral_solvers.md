# Spectral Poisson / Helmholtz Solvers

Direct spectral solvers for 2-D elliptic PDEs on rectangular domains,
using DST-I (Dirichlet), DCT-II (Neumann), or FFT (periodic) spectral
methods.

These solvers treat **every** input node as an unknown (the DST-I wall is
one node outside the input), so pass them only the unknowns of your
staggering.  On the C-grid, the sesmax-core wrappers
`streamfunction_from_vorticity`, `pressure_from_divergence` and `pv_inversion`
([guide](https://jejjohnson.github.io/sesmax/elliptic-inversions/)) do that for
you from the full `[Ny, Nx]` array; see the
[wall table](../elliptic-solvers/#boundary-conditions) to do it by hand.

## DST-I (Dirichlet BCs)

::: gridax.solvers.solve_poisson_dst

::: gridax.solvers.solve_helmholtz_dst

## DCT-II (Neumann BCs)

::: gridax.solvers.solve_poisson_dct

::: gridax.solvers.solve_helmholtz_dct

## FFT (Periodic BCs)

::: gridax.solvers.solve_poisson_fft

::: gridax.solvers.solve_helmholtz_fft

## Eigenvalue Helpers

::: gridax.solvers.dst1_eigenvalues

::: gridax.solvers.dct2_eigenvalues

::: gridax.solvers.fft_eigenvalues
