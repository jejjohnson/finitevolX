# Grid Classes

Arakawa C-grid spatial discretization containers for 1-D, 2-D, and 3-D domains.

## 1-D Grid

::: finitevolx.ArakawaCGrid1D

## 2-D Grid

::: finitevolx.ArakawaCGrid2D

## 3-D Grid

::: finitevolx.ArakawaCGrid3D

## Spherical Grids

Concrete lat-lon C-grids. Beyond the uniform metric inherited from the
curvilinear base (`dx = R·dlon`, `dy = R·dlat`), these expose the
*physical* cell widths, which shrink towards the poles as `cos(lat)`:

| Property | Meaning |
|---|---|
| `dx_T` | zonal cell width at T-points, `R·cos(lat_T)·dlon` |
| `dx_V` | zonal cell width at V-points (and X-points) |
| `dy_T` | meridional cell width, `R·dlat` (uniform) |
| `min_cell_width` | smallest interior cell width — the CFL-limiting length |
| `max_aspect` | largest interior `dy_T / dx_T` — polar cell anisotropy |

`dx_T` and `dx_V` are the raw metric and are not clamped, so a row at a
pole carries a degenerate width of roundoff size and arbitrary sign
(`cos(pi/2)` evaluates to `-4.4e-08` in float32 and `+6.1e-17` in
float64). The two reductions recognise both signs: `min_cell_width`
reports exactly zero for such a row and `max_aspect` reports `inf`. A
width is judged degenerate when it falls within a few ulps of the
equatorial width `R * dlon`, so a genuinely narrow cell is kept. Prefer the reductions when the value feeds a CFL
or resolution guard. Both exclude the ghost ring, and both return 0-d
arrays so they can be used inside `jax.jit`.

::: finitevolx.SphericalGrid2D

::: finitevolx.SphericalGrid3D
