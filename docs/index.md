# finitevolX

Finite volume operators for JAX on Arakawa C-grids.

## Overview

`finitevolX` provides a suite of finite-volume operators for computational fluid
dynamics on staggered Arakawa C-grids, implemented in JAX for high-performance,
differentiable numerical simulations.

## Features

- **Staggered Grid Support**: Full Arakawa C-grid implementation with T, U, V, and X grid points
- **JAX Integration**: Built on JAX for automatic differentiation and GPU/TPU acceleration
- **Modular Operators**: Clean separation of advection, diffusion, interpolation, reconstruction, vorticity, and elliptic solvers
- **WENO Schemes**: High-order WENO reconstruction methods (3rd and 5th order)
- **Boundary Conditions**: Flexible boundary condition handling with ghost cells
- **Type Safety**: Full type hints with `jaxtyping` for array shapes

## Installation

### From PyPI

```bash
pip install finitevolx
```

### From source

```bash
git clone https://github.com/jejjohnson/gridax
cd finitevolX
pip install -e .
```

### With uv (recommended for development and examples)

```bash
uv sync --all-extras
```

## Quick Start

```python
import jax.numpy as jnp
from gridax_jax import Advection2D, CartesianGrid2D

# Create a 2D Arakawa C-grid
grid = CartesianGrid2D.from_interior(64, 64, 1.0, 1.0)

# Initialize advection operator
advection = Advection2D(grid=grid)

# Create velocity and scalar fields (including one ghost-cell ring)
u = jnp.zeros((66, 66))
v = jnp.zeros((66, 66))
h = jnp.ones((66, 66))

# Compute the advective tendency -div(h * u_vec)
dh_dt = advection(h, u, v)
```

## Precision

Every operation returns the floating dtype of its input: float32 in gives float32 out, and float64 in gives float64 out when JAX's `jax_enable_x64` flag is on. The one case JAX would get wrong silently is float64 data with x64 off, which JAX computes in float32 without a warning. Kernels that take host data (NumPy arrays, such as a reanalysis cube read chunk by chunk) call `gridax_jax.check_dtype` on entry and raise a `TypeError` in that case, so the caller chooses: enable x64 to keep float64, or pass float32 on purpose. Grid metrics are not data: they take the dtype of the field they scale.

::: gridax_jax.check_dtype

## Examples

See the `scripts/` directory for complete double-gyre examples that all use the
current class-based API and save sampled fields to Zarr instead of opening live
plots:

- `scripts/swm_linear.py` - Linear shallow-water model
- `scripts/shallow_water.py` - Nonlinear shallow-water model
- `scripts/qg_1p5_layer.py` - 1.5-layer quasi-geostrophic model

The [Examples page](examples.md) shows the saved before/after figures and the
artifacts written by each script.

## Documentation

For detailed API documentation, see the [API Reference](https://jejjohnson.github.io/gridax/reference/).

## Citation

If you use finitevolX in your research, please cite:

```bibtex
@software{finitevolx,
  author = {Johnson, Juan Emmanuel and Uchida, Takaya},
  title = {finitevolX: Finite volume operators for JAX},
  year = {2024},
  url = {https://github.com/jejjohnson/gridax}
}
```

## License

MIT License - see LICENSE file for details.
