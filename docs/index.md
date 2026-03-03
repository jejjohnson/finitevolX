# finitevolX

Finite volume operators for JAX on Arakawa C-grids.

## Overview

`finitevolX` provides a suite of finite-volume operators for computational fluid dynamics on staggered Arakawa C-grids, implemented in JAX for high-performance, differentiable numerical simulations.

## Features

- **Staggered Grid Support**: Full Arakawa C-grid implementation with T, U, V, and X grid points
- **JAX Integration**: Built on JAX for automatic differentiation and GPU/TPU acceleration
- **Modular Operators**: Clean separation of advection, diffusion, interpolation, and reconstruction operators
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

### With uv (recommended for development)

```bash
uv sync --all-extras
```

## Quick Start

```python
import jax.numpy as jnp
from finitevolx import ArakawaCGrid2D, Advection2D

# Create a 2D Arakawa C-grid
grid = ArakawaCGrid2D.from_interior(64, 64, 1.0, 1.0)

# Initialize advection operator
advection = Advection2D(grid=grid)

# Create velocity and scalar fields
u = jnp.zeros((66, 66))  # Includes ghost cells
v = jnp.zeros((66, 66))
h = jnp.ones((66, 66))

# Compute advective tendency
dh_dt = advection.advect(h, u, v)
```

## Examples

See the `scripts/` directory for complete examples:

- `shallow_water.py` - Non-linear shallow water equations
- `swm_linear.py` - Linear shallow water model

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
