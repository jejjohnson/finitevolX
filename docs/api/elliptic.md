# Convenience Wrappers

High-level functions for common elliptic PDE inversions on C-grids.
Each wrapper dispatches to spectral, CG, or capacitance solvers
depending on the ``method`` argument.

## Streamfunction from Vorticity

::: gridax_ops.streamfunction_from_vorticity

## Pressure from Divergence

::: gridax_ops.pressure_from_divergence

## PV Inversion

::: gridax_ops.pv_inversion

## Known Values (Inhomogeneous Dirichlet)

Building blocks behind the wrappers' ``known_values`` / ``known_mask``
arguments, for use with your own solver.  The known values sit on the coast
ring (the dry cells adjacent to the wet mask: the C-grid wall nodes for ψ).

::: gridax_ops.coast_ring

::: gridax_ops.boundary_ring

::: gridax_ops.SolveDomain

::: gridax_ops.KnownValueLifting
