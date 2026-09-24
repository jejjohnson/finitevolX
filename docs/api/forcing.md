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
