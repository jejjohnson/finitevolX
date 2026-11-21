# Elliptic Inversions

The one-call wrappers `streamfunction_from_vorticity`, `pressure_from_divergence` and `pv_inversion` moved to sesmax-core as `sesmax_core.gfd` (#475); gridax no longer has them.
See the [elliptic inversions guide](https://jejjohnson.github.io/sesmax/elliptic-inversions/) and the [`sesmax_core.gfd` API](https://jejjohnson.github.io/sesmax/reference/gfd/).

## Known Values (Inhomogeneous Dirichlet)

Building blocks behind the ``known_values`` / ``known_mask``
arguments of the sesmax inversions, for use with your own solver.  The known values sit on the coast
ring (the dry cells adjacent to the wet mask: the C-grid wall nodes for ψ).

::: gridax.solvers.coast_ring

::: gridax.solvers.boundary_ring

::: gridax.solvers.SolveDomain

::: gridax.solvers.KnownValueLifting
