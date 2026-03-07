# Finite Volume Tools in JAX (In Progress)
[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/finitevolx/badge)](https://www.codefactor.io/repository/github/jejjohnson/finitevolx)
[![codecov](https://codecov.io/gh/jejjohnson/finitevolX/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/finitevolX)

> This package provides finite-volume building blocks for staggered Arakawa grids in JAX.
> The current API is class-based and focuses on differences, interpolations, reconstructions,
> vorticity operators, elliptic solvers, and boundary-condition helpers.

---
## Key Features

**Operators**.
This package has simple operators that are useful for calculating differences.
These include the standard `difference` and `laplacian` operators.
It also has some geostrophic specific operators like the `divergence` and `vorticity`.

**Masks**.
This package includes mask utilities that are consistent with Arakawa grids.

**Interpolation**.
This package includes interpolation schemes for moving between grid spaces.

**Reconstructions**.
This package includes reconstruction methods to calculate the fluxes typically found within the advection terms.
For example, the flux found in the vector invariant formulation for the Shallow Water equations ([example](https://jejjohnson.github.io/jaxsw/sw-formulation#vector-invariant-formulation)).
Another example is the flux found in the advection term for the QG equations ([example](https://jejjohnson.github.io/jaxsw/qg-formulation#eq-qg-general)).

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

The repository now ships three double-gyre examples that all use the current
`finitevolx` API, use `xarray` for preprocessing and postprocessing, write
sampled model fields to Zarr, and save a static before/after comparison figure.
None of the scripts opens a free-running live plot.

### Linear shallow-water model

Script: [`scripts/swm_linear.py`](./scripts/swm_linear.py)

```bash
uv run python scripts/swm_linear.py
```

Artifacts written by default:

- `outputs/linear_shallow_water_double_gyre.zarr`
- `outputs/linear_shallow_water_double_gyre.png`

![Linear shallow-water double gyre](docs/images/linear_shallow_water_double_gyre.png)

### Nonlinear shallow-water model

Script: [`scripts/shallow_water.py`](./scripts/shallow_water.py)

```bash
uv run python scripts/shallow_water.py
```

Artifacts written by default:

- `outputs/shallow_water_double_gyre.zarr`
- `outputs/shallow_water_double_gyre.png`

![Nonlinear shallow-water double gyre](docs/images/shallow_water_double_gyre.png)

### 1.5-layer quasi-geostrophic model

Script: [`scripts/qg_1p5_layer.py`](./scripts/qg_1p5_layer.py)

```bash
uv run python scripts/qg_1p5_layer.py
```

Artifacts written by default:

- `outputs/qg_1p5_layer_double_gyre.zarr`
- `outputs/qg_1p5_layer_double_gyre.png`

![1.5-layer QG double gyre](docs/images/qg_1p5_layer_double_gyre.png)

The QG example now uses basin-scale default parameters that are closer to the
MQGeometry double-gyre benchmark and saves a **relative-vorticity** figure
instead of a streamfunction plot, which produces a more recognizable eddy field.

For a different artifact location during development, pass `--output-dir` to any
of the scripts.

---
## References

**Software**

* [kernex](https://github.com/ASEM000/kernex) - differentiable stencils
* [FiniteDiffX](https://github.com/ASEM000/finitediffX) - finite difference tools in JAX
* [PyFVTool](https://github.com/simulkade/PyFVTool) - Finite Volume Tool in Python
* [jaxinterp2d](https://github.com/adam-coogan/jaxinterp2d) - CartesianGrid interpolator for JAX
* [ndimsplinejax](https://github.com/nmoteki/ndimsplinejax) - SplineGrid interpolator for JAX

**Algorithms**

* [Thiry et al, 2023](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1715/) | [MQGeometry 1.0](https://github.com/louity/MQGeometry) - the WENO reconstructions applied to the multilayer Quasi-Geostrophic equations, Arakawa Grid masks
* [Roullet & Gaillard, 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002663) | [pyRSW](https://github.com/pvthinker/pyRSW#pyrsw) - the WENO reconstructions applied to the shallow water equations
