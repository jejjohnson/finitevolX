# Vorticity Operators

Relative vorticity and Jacobian operators on Arakawa C-grids.

::: finitevolx.Vorticity2D

::: finitevolx.Vorticity3D

## PV flux (functional)

`pv_flux_blend` is the grid-free form of `Vorticity2D.pv_flux_blend`.
`pv_flux_arakawa_lamb` is its deprecated alias: the blend is not the
Arakawa–Lamb (1981) scheme.

::: finitevolx.pv_flux_blend

## Jacobian

`ArakawaJacobian2D` is the mask-aware class form: it returns the full
`[..., Ny, Nx]` T-point array with a zero ghost ring.  Under a mask, `f`
and `g` are treated as zero on dry T-cells (so land `NaN`s cannot reach the
stencil) and dry output cells are exactly zero.  The functional
`arakawa_jacobian` returns only the interior `[..., Ny-2, Nx-2]`, takes no
mask, and reads land values as given.

::: finitevolx.ArakawaJacobian2D

::: finitevolx.arakawa_jacobian
