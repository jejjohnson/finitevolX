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
git clone https://github.com/jejjohnson/finitevolX
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
from finitevolx import Advection2D, ArakawaCGrid2D

# Create a 2D Arakawa C-grid
grid = ArakawaCGrid2D.from_interior(64, 64, 1.0, 1.0)

# Initialize advection operator
advection = Advection2D(grid=grid)

# Create velocity and scalar fields (including one ghost-cell ring)
u = jnp.zeros((66, 66))
v = jnp.zeros((66, 66))
h = jnp.ones((66, 66))

# Compute the advective tendency -div(h * u_vec)
dh_dt = advection(h, u, v)
```

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

For detailed API documentation, see the [API Reference](api/reference.md).

## Citation

If you use finitevolX in your research, please cite:

```bibtex
@software{finitevolx,
  author = {Johnson, Juan Emmanuel and Uchida, Takaya},
  title = {finitevolX: Finite volume operators for JAX},
  year = {2024},
  url = {https://github.com/jejjohnson/finitevolX}
}
```

## License

MIT License - see LICENSE file for details.
