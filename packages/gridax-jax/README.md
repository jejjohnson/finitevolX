# gridax-jax

Arakawa C-grid operations in JAX on gridax grids: raw stencils, differences and interpolation, vorticity and the Arakawa Jacobian, WENO / TVD reconstruction and advection, diffusion, boundary conditions, elliptic and multigrid solvers, diagnostics and the pipekit operator layer. Every operator is an Equinox module or a pure function, so it runs under `jit`, `vmap` and `grad`.

The grids and masks themselves are in [gridax](../gridax), which runs on NumPy alone; gridax-jax depends on it and re-exports its names.

```python
import jax.numpy as jnp
import gridax_jax as fvx

grid = fvx.CartesianGrid2D.from_interior(64, 64, Lx=1.0, Ly=1.0)
h = jnp.zeros((grid.Ny, grid.Nx))
lap = fvx.Difference2D(grid=grid).laplacian(h)
```

This package was called finitevolX; `import finitevolx` still works for one 0.X release, re-exports these names and warns.
