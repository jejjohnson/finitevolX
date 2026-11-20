# Convenience Wrappers

High-level functions for common elliptic PDE inversions on C-grids.
Each wrapper dispatches to spectral, CG, or capacitance solvers
depending on the ``method`` argument.

## Streamfunction from Vorticity

::: gridax_jax.streamfunction_from_vorticity

## Pressure from Divergence

::: gridax_jax.pressure_from_divergence

## PV Inversion

::: gridax_jax.pv_inversion

## Known Values (Inhomogeneous Dirichlet)

Building blocks behind the wrappers' ``known_values`` / ``known_mask``
arguments, for use with your own solver.  The known values sit on the coast
ring (the dry cells adjacent to the wet mask: the C-grid wall nodes for ψ).

::: gridax_jax.coast_ring

::: gridax_jax.boundary_ring

::: gridax_jax.SolveDomain

::: gridax_jax.KnownValueLifting
