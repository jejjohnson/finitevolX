# Finite Volume Tools in JAX (In Progress)
[![CI Tests](https://github.com/jejjohnson/finitevolX/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/finitevolX/actions/workflows/ci.yml)
[![Lint & Format](https://github.com/jejjohnson/finitevolX/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/finitevolX/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/finitevolX/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/finitevolX/actions/workflows/typecheck.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/finitevolx/badge)](https://www.codefactor.io/repository/github/jejjohnson/finitevolx)
[![codecov](https://codecov.io/gh/jejjohnson/finitevolX/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/finitevolX)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> This package provides finite-volume building blocks for staggered Arakawa grids in JAX.
> The current API is class-based and focuses on differences, interpolations, reconstructions,
> vorticity operators, elliptic solvers, and boundary-condition helpers.

---
## Key Features

**Grid & Masks**.
Arakawa C-grid classes (`ArakawaCGrid1D/2D/3D`) with stencil-aware mask utilities for land/ocean boundaries.

**Operators**.
Finite-difference, interpolation, divergence, vorticity, Coriolis, and Jacobian operators on staggered grids. Diagnostic quantities like kinetic energy and Bernoulli potential.

**Advection & Reconstruction**.
Advection operators with pluggable reconstruction methods including upwind, TVD (minmod, van Leer, superbee, MC), and WENO (3rd, 5th, 7th, 9th order) schemes. Mask-aware stencil dispatch for irregular domains.

**Diffusion**.
Harmonic and biharmonic diffusion operators, plus energy/enstrophy-conserving momentum advection in vector-invariant form.

**Boundary Conditions**.
Composable per-face boundary conditions (Dirichlet, Neumann, periodic, sponge, slip, Robin, etc.) with ghost-cell enforcement.

**Time Integration**.
Pure functional time steppers (Euler, Heun/RK2, SSP-RK3, RK4, AB2/AB3, Leapfrog+RAF, IMEX-SSP2, split-explicit) and [diffrax](https://docs.kidger.site/diffrax/)-based Butcher tableau solvers for adaptive stepping, checkpointing, and `SaveAt`. Semi-Lagrangian advection for large-CFL transport.

**Elliptic Solvers**.
Spectral Poisson/Helmholtz solvers (DST, DCT, FFT), capacitance matrix method for masked domains, and preconditioned conjugate gradient.

**Vertical Modes**.
Vertical mode decomposition and multilayer vmap helper for 3D simulations.

---
## 🛠️ Installation<a id="installation"></a>

### `pip`

Install the package directly from GitHub:

```bash
pip install git+https://github.com/jejjohnson/finitevolX
```

### `uv` (recommended)

Install the development, test, documentation, and example dependencies:

```bash
git clone https://github.com/jejjohnson/finitevolX.git
cd finitevolX
uv sync --all-extras
```

---
## ⏩ Examples<a id="examples"></a>

The repository ships three double-gyre examples that use the `finitevolx` API
for spatial operators and `finitevolx.heun_step` for time integration.  Each
script uses `xarray` for preprocessing and postprocessing, writes sampled model
fields to Zarr, and saves an **animated GIF** showing the field evolution over
time.  Each script accepts an optional `--spinup-steps` argument to run a silent
spin-up phase before recording frames, which helps produce a recognisable
double-gyre structure in the animation.

### Linear shallow-water model

Script: [`scripts/swm_linear.py`](./scripts/swm_linear.py)

```bash
uv run python scripts/swm_linear.py
```

Artifacts written by default:

- `outputs/linear_shallow_water_double_gyre.zarr`
- `outputs/linear_shallow_water_double_gyre.gif`

![Linear shallow-water double gyre](docs/images/linear_shallow_water_double_gyre.gif)

### Nonlinear shallow-water model

Script: [`scripts/shallow_water.py`](./scripts/shallow_water.py)

```bash
uv run python scripts/shallow_water.py
```

Artifacts written by default:

- `outputs/shallow_water_double_gyre.zarr`
- `outputs/shallow_water_double_gyre.gif`

![Nonlinear shallow-water double gyre](docs/images/shallow_water_double_gyre.gif)

### 1.5-layer quasi-geostrophic model

Script: [`scripts/qg_1p5_layer.py`](./scripts/qg_1p5_layer.py)

```bash
uv run python scripts/qg_1p5_layer.py
```

Artifacts written by default:

- `outputs/qg_1p5_layer_double_gyre.zarr`
- `outputs/qg_1p5_layer_double_gyre.gif`

![1.5-layer QG double gyre](docs/images/qg_1p5_layer_double_gyre.gif)

The QG example uses basin-scale default parameters that are closer to the
MQGeometry double-gyre benchmark and saves a **relative-vorticity** animation
instead of a streamfunction plot, which produces a more recognizable eddy field.
Run with `--spinup-steps 8000` to start recording after the double-gyre
circulation is well established (~370 days into the simulation).

For a different artifact location during development, pass `--output-dir` to any
of the scripts.

---
## References

**Software**

* [PyFVTool](https://github.com/simulkade/PyFVTool) - Finite Volume Tool in Python
* [jaxinterp2d](https://github.com/adam-coogan/jaxinterp2d) - CartesianGrid interpolator for JAX
* [ndimsplinejax](https://github.com/nmoteki/ndimsplinejax) - SplineGrid interpolator for JAX
* [diffrax](https://docs.kidger.site/diffrax/) - JAX-native ODE/SDE solvers (used by finitevolX's time integration module)

**Algorithms**

* [Thiry et al, 2023](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1715/) | [MQGeometry 1.0](https://github.com/louity/MQGeometry) - the WENO reconstructions applied to the multilayer Quasi-Geostrophic equations, Arakawa Grid masks
* [Roullet & Gaillard, 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002663) | [pyRSW](https://github.com/pvthinker/pyRSW#pyrsw) - the WENO reconstructions applied to the shallow water equations
* [Gottlieb, Shu & Tadmor, 2001](https://doi.org/10.1137/S003614450036757X) - Strong Stability-Preserving High-Order Time Discretization Methods (SSP-RK schemes)
* [Ketcheson, 2008](https://doi.org/10.1137/07070485X) - Highly efficient SSP methods: SSP-RK(10,4) with 10 stages and effective CFL coefficient of 6
