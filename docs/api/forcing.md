# Forcing Operators

Surface wind stress, bottom drag, and Rayleigh damping for C-grid ocean
models.

## Functional primitives

Pure, stateless functions that compute the core forcing math.  They take
fields **already at the correct stagger** and do no interpolation, ghost-ring
padding, or masking — the caller applies `interior()` and any mask.

::: finitevolx.wind_stress_tendency

::: finitevolx.linear_drag_tendency

::: finitevolx.quadratic_drag_tendency

::: finitevolx.rayleigh_tendency
