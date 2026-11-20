# Spectral Poisson / Helmholtz Solvers

Direct spectral solvers for 2-D elliptic PDEs on rectangular domains,
using DST-I (Dirichlet), DCT-II (Neumann), or FFT (periodic) spectral
methods.

These solvers treat **every** input node as an unknown (the DST-I wall is
one node outside the input), so pass them only the unknowns of your
staggering.  On the C-grid, `streamfunction_from_vorticity`,
`pressure_from_divergence` and `pv_inversion` do that for you from the full
`[Ny, Nx]` array; see the
[wall table](../elliptic_solvers.md#boundary-conditions).

## DST-I (Dirichlet BCs)

::: gridax_jax.solve_poisson_dst

::: gridax_jax.solve_helmholtz_dst

## DCT-II (Neumann BCs)

::: gridax_jax.solve_poisson_dct

::: gridax_jax.solve_helmholtz_dct

## FFT (Periodic BCs)

::: gridax_jax.solve_poisson_fft

::: gridax_jax.solve_helmholtz_fft

## Eigenvalue Helpers

::: gridax_jax.dst1_eigenvalues

::: gridax_jax.dct2_eigenvalues

::: gridax_jax.fft_eigenvalues
