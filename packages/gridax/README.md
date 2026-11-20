# gridax

Structured grids and land / ocean masks as data, on NumPy.

gridax holds the arrays that describe a grid: Arakawa C-grid containers (Cartesian and spherical, 1-D to 3-D, with one ghost ring) and the staggered wet / dry masks derived from a T-point mask. It installs with NumPy, SciPy and jaxtyping only, so a NumPy-only environment can describe the grid a model ran on. When JAX is installed, every grid and mask is also a JAX pytree, with the grid sizes as static metadata.

```python
import numpy as np
from gridax import CartesianGrid2D, Mask2D

grid = CartesianGrid2D.from_interior(64, 32, Lx=1.0e6, Ly=5.0e5)
depth = np.full((grid.Ny, grid.Nx), 4000.0)
depth[:, :3] = -10.0  # a coast on the west
mask = Mask2D.from_bathymetry(depth, min_depth=10.0)
```

`gridax.analysis` holds the analysis kernels on collocated grids: finite differences (`difference`, `adaptive_difference`, and the 1-D / 2-D / 3-D gradient, divergence, curl and Laplacian) and axis remapping (`remap_axis`, `remap_axis_columns`). They follow the array API, so the same call runs on NumPy, JAX or any other backend and returns the input's dtype. The staggered C-grid operations (advection, diffusion, boundary conditions, solvers) are in gridax-jax, which depends on gridax.
