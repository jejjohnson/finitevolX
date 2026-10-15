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

::: finitevolx.solve_poisson_dst

::: finitevolx.solve_helmholtz_dst

## DCT-II (Neumann BCs)

::: finitevolx.solve_poisson_dct

::: finitevolx.solve_helmholtz_dct

## FFT (Periodic BCs)

::: finitevolx.solve_poisson_fft

::: finitevolx.solve_helmholtz_fft

## Eigenvalue Helpers

::: finitevolx.dst1_eigenvalues

::: finitevolx.dct2_eigenvalues

::: finitevolx.fft_eigenvalues
