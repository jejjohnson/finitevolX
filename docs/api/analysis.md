# Analysis kernels (gridax)

Finite differences and remapping on collocated grids, from `gridax.analysis`. They compute in the namespace of their input, so the same call runs on NumPy, on JAX (also under `jax.jit`), or on any backend `array-api-compat` knows, and returns the input's dtype. They are the kernels behind geoops' and xrops' `Gradient`, `Divergence`, `Curl`, `Laplacian` and vertical-remap operators. Their stencil coefficients come from `gridax.stencils` (Layer 0); for fields on the staggered C-grid, use the Layer 1 stencils of `gridax.stencils` or the `Difference*` operators instead (see [the derivative stack](https://jejjohnson.github.io/gridax/stencils/#derivative-stack)).

::: gridax.analysis
