# Divergence Operators

Flux-form divergence operators on staggered Arakawa C-grids.

::: gridax_jax.Divergence2D

`Difference2D.divergence(u, v)` computes the same divergence. The free function `divergence_2d(u, v, dx, dy)` is deprecated (#456); without a grid object, write `interior(diff_x_bwd_2d(u) / dx + diff_y_bwd_2d(v) / dy, u)`.
