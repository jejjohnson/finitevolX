# Forcing Operators

Surface wind stress, bottom drag, and Rayleigh damping for C-grid ocean
models.

## Functional primitives

Pure, stateless functions that compute the core forcing math.  They take
fields **already at the correct stagger** and do no interpolation, ghost-ring
padding, or masking.  The output has the shape of the inputs, so to get a
full `[Ny, Nx]` field slice every array input to the interior `[1:-1, 1:-1]`
first, then pad the result back with `interior()` and apply any mask.

::: finitevolx.wind_stress_tendency

::: finitevolx.linear_drag_tendency

::: finitevolx.quadratic_drag_tendency

::: finitevolx.rayleigh_tendency

## Base class and composition

`AbstractForcing` is the contract every forcing operator satisfies.  The
composition wrappers combine forcings inside a model's right-hand side:
`ForcingSum` adds independent processes, `ForcingProduct` modulates a forcing
by a fixed or callable factor (sponges, seasonal cycles), and
`TimeVaryingForcing` evaluates time-dependent input fields — analytic or
interpolated with `diffrax.LinearInterpolation` — before calling a
time-agnostic forcing.

::: finitevolx.AbstractForcing

::: finitevolx.ForcingSum

::: finitevolx.ForcingProduct

::: finitevolx.TimeVaryingForcing

## 2D operators

Module operators for `CartesianGrid2D`.  They interpolate T-point inputs to
the velocity faces, delegate the math to the functional primitives, restore
the zero ghost ring, and apply an optional `Mask2D` given at construction.
Multilayer models (`[nl, Ny, Nx]`) reuse them through `multilayer()` or by
calling them on a single layer, e.g. bottom drag on `u[-1], v[-1]`.

::: finitevolx.WindStress2D

::: finitevolx.LinearDrag2D

::: finitevolx.QuadraticDrag2D

::: finitevolx.RayleighDamping2D
