# Vorticity Operators

Relative vorticity and Jacobian operators on Arakawa C-grids.

::: finitevolx.Vorticity2D

::: finitevolx.Vorticity3D

## Jacobian

`ArakawaJacobian2D` is the mask-aware class form: it returns the full
`[..., Ny, Nx]` T-point array with a zero ghost ring and dry cells zeroed by
`mask.h`.  The functional `arakawa_jacobian` returns only the interior
`[..., Ny-2, Nx-2]` and takes no mask.

::: finitevolx.ArakawaJacobian2D

::: finitevolx.arakawa_jacobian
