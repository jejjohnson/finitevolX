# Reconstruction Operators

High-order flux reconstruction methods (upwind, WENO) for advection terms.

## One reconstruction for every scheme and axis

`gridax.advection.reconstruct(q, u, axis=..., scheme=...)` gives the advective flux on the faces along any axis, for any scheme. A scheme is a ladder of stencil widths, widest first, and each face takes the widest one that fits in the array (and, with `mask=`, on wet cells), down to first-order upwind. Every method of `Reconstruction1D/2D/3D` below is one call of it.

::: gridax.advection.reconstruct

::: gridax.advection.Upwind

::: gridax.advection.Centred

::: gridax.advection.WENO

::: gridax.advection.TVD

::: gridax.advection.Tier

## The class forms

::: gridax_jax.Reconstruction1D

::: gridax_jax.Reconstruction2D

::: gridax_jax.Reconstruction3D

## Flux Limiters

::: gridax_jax.minmod

::: gridax_jax.mc

::: gridax_jax.superbee

::: gridax_jax.van_leer
