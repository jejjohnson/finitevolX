# Forcing Operators

!!! warning "Moved to sesmax-core"
    These names moved to `sesmax_core.forcing` (#422). `gridax_jax` keeps them for one release behind a `DeprecationWarning`; import them from sesmax-core instead.

Surface wind stress, bottom drag, and Rayleigh damping for C-grid ocean
models.

## Functional primitives

Pure, stateless functions that compute the core forcing math.  They take
fields **already at the correct stagger** and do no interpolation, ghost-ring
padding, or masking.  The output has the shape of the inputs, so to get a
full `[Ny, Nx]` field slice every array input to the interior `[1:-1, 1:-1]`
first, then pad the result back with `interior()` and apply any mask.

::: gridax_jax._src.forcing.functional.wind_stress_tendency

::: gridax_jax._src.forcing.functional.linear_drag_tendency

::: gridax_jax._src.forcing.functional.quadratic_drag_tendency

::: gridax_jax._src.forcing.functional.rayleigh_tendency

## Base class and composition

`AbstractForcing` is the contract every forcing operator satisfies.  The
composition wrappers combine forcings inside a model's right-hand side:
`ForcingSum` adds independent processes, `ForcingProduct` modulates a forcing
by a fixed or callable factor (sponges, seasonal cycles), and
`TimeVaryingForcing` evaluates time-dependent input fields — analytic or
interpolated with `diffrax.LinearInterpolation` — before calling a
time-agnostic forcing.

::: gridax_jax._src.forcing._base.AbstractForcing

::: gridax_jax._src.forcing._base.ForcingSum

::: gridax_jax._src.forcing._base.ForcingProduct

::: gridax_jax._src.forcing._base.TimeVaryingForcing

## 2D operators

Module operators for `CartesianGrid2D`.  They interpolate T-point inputs to
the velocity faces, delegate the math to the functional primitives, restore
the zero ghost ring, and apply an optional `Mask2D` given at construction.
Multilayer models (`[nl, Ny, Nx]`) reuse them through `multilayer()` or by
calling them on a single layer, e.g. bottom drag on `u[-1], v[-1]`.

::: gridax_jax._src.forcing.wind.WindStress2D

`WindCurl2D` is the vorticity / QG-PV form of the same forcing: the C-grid
curl of the face-interpolated stress divided by `rho0 * H`, at X-points.

::: gridax_jax._src.forcing.wind.WindCurl2D

::: gridax_jax._src.forcing.drag.LinearDrag2D

::: gridax_jax._src.forcing.drag.QuadraticDrag2D

::: gridax_jax._src.forcing.rayleigh.RayleighDamping2D

## 3D operators

Wrappers for `CartesianGrid3D` built on the 2D operators, with an optional
`Mask3D`.  Levels are stacked bottom-up — `k = 0` is the bottom ghost and
`k = Nz - 1` the top ghost.  Each operator follows one of three vertical
patterns:

| Pattern | Operators | Levels written |
|---------|-----------|----------------|
| Top injection | `WindStress3D` | top interior level `k = Nz - 2` only |
| Bottom injection | `LinearDrag3D`, `QuadraticDrag3D` | bottom interior level `k = 1` only |
| Uniform vmap | `RayleighDamping3D` | every interior level (z-ghosts zeroed) |

::: gridax_jax._src.forcing.wind.WindStress3D

::: gridax_jax._src.forcing.drag.LinearDrag3D

::: gridax_jax._src.forcing.drag.QuadraticDrag3D

::: gridax_jax._src.forcing.rayleigh.RayleighDamping3D
