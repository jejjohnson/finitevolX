# Convenience Wrappers

High-level functions for common elliptic PDE inversions on C-grids.
Each wrapper dispatches to spectral, CG, or capacitance solvers
depending on the ``method`` argument.

## Streamfunction from Vorticity

::: finitevolx.streamfunction_from_vorticity

## Pressure from Divergence

::: finitevolx.pressure_from_divergence

## PV Inversion

::: finitevolx.pv_inversion

## Known Values (Inhomogeneous Dirichlet)

Building blocks behind the wrappers' ``known_values`` / ``known_mask``
arguments, for use with your own solver.  The known values sit on the coast
ring (the dry cells adjacent to the wet mask: the C-grid wall nodes for ψ).

::: finitevolx.coast_ring

::: finitevolx.boundary_ring

::: finitevolx.SolveDomain

::: finitevolx.KnownValueLifting
