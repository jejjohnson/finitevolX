# Analysis kernels (gridax)

Finite differences and remapping on collocated grids, from `gridax.analysis`. They compute in the namespace of their input, so the same call runs on NumPy, on JAX (also under `jax.jit`), or on any backend `array-api-compat` knows, and returns the input's dtype. They are the kernels behind geoops' and xrops' `Gradient`, `Divergence`, `Curl`, `Laplacian` and vertical-remap operators. For fields on the staggered C-grid, use the `Difference*` operators instead.

::: gridax.analysis
