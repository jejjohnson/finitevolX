## Description

Multi-layer QG and shallow water models require a **vertical coupling matrix** `A` built from layer thicknesses `H` and reduced gravities `g'`, plus its eigendecomposition into **Rossby deformation modes**. The associated **layer-to-mode** (`Cl2m`) and **mode-to-layer** (`Cm2l`) transforms are present in both MQGeometry and qgsw-pytorch but are completely absent from finitevolX.

### Why this matters

For an N-layer QG model, the PV inversion step requires solving N coupled 2-D Helmholtz problems. The A-matrix eigendecomposition decouples these into N *independent* Helmholtz problems (one per vertical mode), each solvable with the existing `solve_helmholtz_dst` / `solve_helmholtz_dct` functions. Without this, multi-layer PV inversion is not feasible.

### The A matrix

For N layers with thicknesses `H = [H_1, ..., H_N]` and reduced gravities `g' = [g'_1, ..., g'_{N-1}]`:

```
A[k, k]   = -(f₀² / (g'[k] * H[k])) - (f₀² / (g'[k-1] * H[k]))   (diagonal)
A[k, k+1] =  (f₀² / (g'[k] * H[k]))                                (super-diagonal)
A[k-1, k] =  (f₀² / (g'[k-1] * H[k-1]))                           (sub-diagonal)
```

Its eigenvalues give the **Rossby deformation radii** `Rd = 1/sqrt(-λ)`.

## References

- [`louity/MQGeometry/qgm.py`](https://github.com/louity/MQGeometry/blob/main/qgm.py) — `compute_auxillary_matrices()` builds A, eigendecomposition, `Cl2m`, `Cm2l`
- [`louity/qgsw-pytorch/src/qg.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/qg.py) — `Cl2m`, `Cm2l` transforms and PV inversion loop

## Proposed API

```python
def build_coupling_matrix(
    H: Float[Array, "Nz"],
    g_prime: Float[Array, "Nz-1"],
    f0: float,
) -> Float[Array, "Nz Nz"]:
    """Build the vertical coupling (A) matrix for a multi-layer QG/SW model.
    
    Parameters
    ----------
    H : Float[Array, "Nz"]
        Layer thicknesses [m].
    g_prime : Float[Array, "Nz-1"]
        Reduced gravities between adjacent layers [m/s²].
    f0 : float
        Coriolis parameter [1/s].
    
    Returns
    -------
    Float[Array, "Nz Nz"]
        Tridiagonal coupling matrix A.
    """


def decompose_vertical_modes(
    A: Float[Array, "Nz Nz"],
) -> tuple[Float[Array, "Nz"], Float[Array, "Nz Nz"], Float[Array, "Nz Nz"]]:
    """Eigendecompose the coupling matrix into vertical modes.
    
    Returns
    -------
    lambdas : Float[Array, "Nz"]
        Eigenvalues. Rossby deformation radii: Rd[k] = 1/sqrt(-λ[k]).
    Cl2m : Float[Array, "Nz Nz"]
        Layer-to-mode transform matrix.
    Cm2l : Float[Array, "Nz Nz"]
        Mode-to-layer transform matrix (inverse of Cl2m).
    """


def layer_to_mode(
    field: Float[Array, "Nz Ny Nx"],
    Cl2m: Float[Array, "Nz Nz"],
) -> Float[Array, "Nz Ny Nx"]:
    """Transform a layered field to modal representation."""


def mode_to_layer(
    field: Float[Array, "Nz Ny Nx"],
    Cm2l: Float[Array, "Nz Nz"],
) -> Float[Array, "Nz Ny Nx"]:
    """Transform a modal field back to layer representation."""
```

## Acceptance Criteria

- [ ] Functions in a new `finitevolx/_src/multilayer.py` module
- [ ] Exports from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_multilayer.py` verifying:
  - `Cm2l @ Cl2m ≈ I` (round-trip identity)
  - `A @ Cm2l ≈ diag(lambdas) @ Cm2l` (eigenvalue equation)
  - 1-layer edge case returns identity transform
  - Deformation radii match known values for simple 2-layer configurations
- [ ] NumPy-style docstrings with physical equations

## Priority

**High** — Core building block for any multi-layer QG or SW model. Unlocks multi-layer PV inversion using the existing spectral solvers.
