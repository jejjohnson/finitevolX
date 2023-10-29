# Finite Volume Tools in JAX (In Progress)

> This package has some tools for finite volume methods in JAX.
> We use the staggered grids (i.e. Arakawa Grids) to define variables an the nodes, faces, and centers.
> This package includes all of the key operations necessary for interacting with these variables including: 1) differences 2) interpolations, and 3) reconstructions.
> In addition, we provide a way to mask boundaries which are consistent with Arakawa Grids.



---
## Key Features

**Operators**. 
This package has simple operators that are useful for calculating differences. 
These include the standard `difference` and `laplacian` operators.
It also has some geostrophic specific operators like the `geostrophic_gradient`, the `divergence`, and `vorticity`.


**Masks**. 
This package includes some useful masks that are necessary for interacting with the Arakawa C-/D-Grids.


**Interpolation**.
This package includes some simple interpolation schemes for moving between grid spaces.
It has linear interpolation schemes based on the mean which include the `arithmetic`, `geometric`, `harmonic`, and `quadratic`.
We also have some general purpose interpolation schemes for the Cartesian grid including the linear interpolation scheme and splines.


**Reconstructions**.
This package includes some *reconstruction* methods to calculate the fluxes typically found within the advection terms.
For example, the flux found in the vector invariant formulation for the Shallow Water equations ([example](https://jejjohnson.github.io/jaxsw/sw-formulation#vector-invariant-formulation)).
Another example, the flux found in the advection term for the QG equations ([example](https://jejjohnson.github.io/jaxsw/qg-formulation#eq-qg-general)).

---
## Installation

We can install it directly through pip

```bash
pip install git+https://github.com/jejjohnson/finitevolX
```

We also use poetry for the development environment.

```bash
git clone https://github.com/jejjohnson/finitevolX.git
cd finitevolX
conda create -n finitevolx python=3.11 poetry
poetry install
```



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