# Raw Stencil Primitives

Pure index-arithmetic stencils for Arakawa C-grids.  No metric scaling,
no ghost-ring padding.  See the [theory page](../stencils/) for the
mathematical background.

## Coefficients (Layer 0)

Offsets and exact coefficients from the Taylor conditions, in `gridax.stencils`: integer offsets for collocated stencils (as `gridax.analysis` uses them), half-integer offsets (`method="staggered"`) for the faces of a C-grid. The two-point staggered stencils are the raw stencils below.

::: gridax.stencils.difference_offsets

::: gridax.stencils.difference_coeffs

::: gridax.stencils.nonuniform_coeffs

With `data="cell"` the samples are cell averages, not point values: `difference_coeffs((-1.5, -0.5, 0.5, 1.5), 0, data="cell")` is the finite-volume reconstruction $(-1, 7, 7, -1)/12$ of `linear_4pts`, where point values give the interpolation $(-1, 9, 9, -1)/16$. `bias="left"` and `bias="right"` give the upwind offsets of advection.

## Higher-order staggered stencils (Layer 1)

Fourth- and sixth-order differences and averages of point values on the C-grid, centred where they fit and shifted inwards at the walls. `accuracy=1` is the named two-point stencil.

::: gridax.stencils.staggered_difference

::: gridax.stencils.staggered_average

## WENO weights and smoothness (Layer 0)

::: gridax.stencils.weno_stencils

::: gridax.stencils.optimal_weights

::: gridax.stencils.smoothness_matrix

## 1-D Difference Stencils

::: gridax_jax.diff_x_fwd_1d

::: gridax_jax.diff_x_bwd_1d

## 2-D Difference Stencils

::: gridax_jax.diff_x_fwd_2d

::: gridax_jax.diff_y_fwd_2d

::: gridax_jax.diff_x_bwd_2d

::: gridax_jax.diff_y_bwd_2d

## 3-D Difference Stencils

::: gridax_jax.diff_x_fwd_3d

::: gridax_jax.diff_y_fwd_3d

::: gridax_jax.diff_x_bwd_3d

::: gridax_jax.diff_y_bwd_3d

## 1-D Averaging Stencils

::: gridax_jax.avg_x_fwd_1d

::: gridax_jax.avg_x_bwd_1d

## 2-D Averaging Stencils (2-point)

::: gridax_jax.avg_x_fwd_2d

::: gridax_jax.avg_y_fwd_2d

::: gridax_jax.avg_x_bwd_2d

::: gridax_jax.avg_y_bwd_2d

## 2-D Averaging Stencils (4-point bilinear)

::: gridax_jax.avg_xfwd_yfwd_2d

::: gridax_jax.avg_xbwd_ybwd_2d

::: gridax_jax.avg_xbwd_yfwd_2d

::: gridax_jax.avg_xfwd_ybwd_2d

## 3-D Averaging Stencils

::: gridax_jax.avg_x_fwd_3d

::: gridax_jax.avg_y_fwd_3d

::: gridax_jax.avg_x_bwd_3d

::: gridax_jax.avg_y_bwd_3d
