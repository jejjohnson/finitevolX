## Description

All three reference repositories include standard physical forcing and dissipation operators that are essential for ocean simulations. These are absent from finitevolX, forcing users to implement them from scratch for every application.

### Missing operators

1. **Wind stress curl** — converts wind stress components `(τx, τy)` to a vorticity source term:
   `F_wind[j, i] = (∂τy/∂x - ∂τx/∂y) / (ρ₀ * H₁)`

2. **Bottom Ekman drag** — linear bottom friction damping the bottom-layer vorticity:
   `F_drag = -r * ζ_bottom`

3. **Laplacian (harmonic) viscosity** — `ν∇²q` for lateral diffusion

4. **Bi-Laplacian (biharmonic) viscosity** — `-ν₄∇⁴q = -ν₄∇²(∇²q)` for scale-selective hyperdiffusion

## References

- [`louity/qgm_pytorch/QGM.py`](https://github.com/louity/qgm_pytorch/blob/main/QGM.py) — `curl_wind`, `a_2` (Laplacian), `a_4` (biharmonic), `delta_ek` (Ekman drag)
- [`louity/MQGeometry/qgm.py`](https://github.com/louity/MQGeometry/blob/main/qgm.py) — `wind_forcing`, `bottom_drag_coef`, biharmonic term
- [`louity/qgsw-pytorch/src/sw.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/sw.py) — `taux`/`tauy` wind stress, `bottom_drag_coef`

## Proposed API

```python
# finitevolx/_src/forcing.py

def wind_stress_curl(
    taux: Float[Array, "Ny Nx"],
    tauy: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    rho0: float = 1025.0,
    H1: float = 1.0,
) -> Float[Array, "Ny Nx"]:
    """Curl of wind stress at T-points (vorticity forcing).
    
    F[j, i] = (∂τy/∂x - ∂τx/∂y) / (ρ₀ * H₁)
    """


def bottom_drag(
    zeta: Float[Array, "Ny Nx"],
    coef: float,
) -> Float[Array, "Ny Nx"]:
    """Linear bottom Ekman drag on vorticity.
    
    F[j, i] = -coef * ζ[j, i]
    """


def laplacian_viscosity(
    q: Float[Array, "Ny Nx"],
    nu: float,
    dx: float,
    dy: float,
    mask: Bool[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Laplacian (harmonic) lateral viscosity: ν∇²q."""


def bilaplacian_viscosity(
    q: Float[Array, "Ny Nx"],
    nu4: float,
    dx: float,
    dy: float,
    mask: Bool[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Bi-Laplacian (biharmonic) lateral viscosity: -ν₄∇⁴q.
    
    Computed as -ν₄ * ∇²(∇²q) using two successive Laplacian applications.
    """
```

## Implementation Notes

- **Wind stress curl**: uses `diff_x_V_to_T` and `diff_y_U_to_T` stencils (or their equivalents) on the wind stress components — note that `τx` lives at U-points and `τy` lives at V-points
- **Bottom drag**: trivially `-coef * zeta`, applied only to the bottom layer in multi-layer models
- **Laplacian viscosity**: use `Difference2D` operators `diff_x_U_to_T + diff_y_V_to_T` applied to the gradients of `q`
- **Bilaplacian**: apply `laplacian_viscosity` twice; intermediate Laplacian needs ghost cells (apply BCs between the two applications)
- All operators write only to `[1:-1, 1:-1]`; ghost ring initialized to zero

## Acceptance Criteria

- [ ] Functions in `finitevolx/_src/forcing.py` (new file)
- [ ] Exports from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_forcing.py` verifying:
  - `wind_stress_curl` of uniform wind field is zero
  - `wind_stress_curl` of linearly varying wind matches analytical result
  - `bottom_drag` scales correctly with coefficient
  - `laplacian_viscosity` of constant field is zero
  - `bilaplacian_viscosity` of quadratic polynomial matches analytical result
- [ ] NumPy-style docstrings with physical equations

## Priority

**Medium** — Essential physics for any ocean simulation. Wind forcing and bottom drag are the first forcing terms added to any QG or SW model.
