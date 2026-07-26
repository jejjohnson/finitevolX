## Description

The Arakawa (1966) Jacobian `J(ψ, q)` is the classical energy- and enstrophy-conserving discretization of PV advection in QG models. It is a **discrete analog** of the continuous Jacobian `∂ψ/∂x * ∂q/∂y - ∂ψ/∂y * ∂q/∂x` that guarantees both energy and enstrophy are conserved at the discrete level — a property that WENO-based upwind schemes do not share.

The Arakawa scheme uses a nine-point stencil combining three alternative forms of the Jacobian (J++, J+×, J×+), each conserving either energy or enstrophy individually, with the average conserving both:

```
J[j,i] = (1/3) * (J++ + J+× + J×+)[j,i]
```

This operator is implemented in `qgm_pytorch` as `jacobi_h(psi, q, dx)` and is the standard advection scheme for QG ocean simulations.

## References

- [`louity/qgm_pytorch/QGM.py`](https://github.com/louity/qgm_pytorch/blob/main/QGM.py) — `jacobi_h(psi, q, dx)` — classic Arakawa (1966) discretization
- Arakawa (1966), *Journal of Computational Physics*, 1(1):119–143

## Proposed API

```python
def arakawa_jacobian(
    psi: Float[Array, "Ny Nx"],
    q: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Arakawa (1966) Jacobian J(ψ, q) for energy- and enstrophy-conserving PV advection.
    
    Computes J(ψ, q) = ∂ψ/∂x * ∂q/∂y - ∂ψ/∂y * ∂q/∂x using the
    average of three nine-point stencil forms (J++, J+×, J×+).
    
    This discretization conserves both kinetic energy and enstrophy.
    Both ψ and q are at T-points; output tendency is at T-points.
    
    Parameters
    ----------
    psi : Float[Array, "Ny Nx"]
        Streamfunction at T-points.
    q : Float[Array, "Ny Nx"]
        Potential vorticity at T-points.
    dx : float
        Grid spacing in x [m].
    dy : float
        Grid spacing in y [m].
    
    Returns
    -------
    Float[Array, "Ny Nx"]
        PV tendency J(ψ, q) at T-points. Written to [1:-1, 1:-1].
    
    References
    ----------
    Arakawa (1966), J. Comput. Phys., 1(1):119-143.
    """


class ArakawaJacobian2D(eqx.Module):
    """Arakawa (1966) Jacobian operator on the Arakawa C-grid."""
    grid: ArakawaCGrid2D

    def __call__(
        self,
        psi: Float[Array, "Ny Nx"],
        q: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Compute J(ψ, q)."""
```

## Implementation Notes

The three Jacobian forms using 9-point stencils (following Arakawa 1966):

```python
# J++ (centred differences of both psi and q)
Jpp = ((psi[j+1,i] - psi[j-1,i]) * (q[j,i+1] - q[j,i-1])
     - (psi[j,i+1] - psi[j,i-1]) * (q[j+1,i] - q[j-1,i])) / (4*dx*dy)

# J+x (centred differences of psi, corner differences of q)
Jpx = (psi[j+1,i] * (q[j+1,i+1] - q[j+1,i-1])
     - psi[j-1,i] * (q[j-1,i+1] - q[j-1,i-1])
     - psi[j,i+1] * (q[j+1,i+1] - q[j-1,i+1])
     + psi[j,i-1] * (q[j+1,i-1] - q[j-1,i-1])) / (4*dx*dy)

# Jx+ (corner differences of psi, centred differences of q)
Jxp = (psi[j+1,i+1] * (q[j,i+1] - q[j+1,i])
     - psi[j-1,i-1] * (q[j-1,i] - q[j,i-1])
     - psi[j-1,i+1] * (q[j,i+1] - q[j-1,i])
     + psi[j+1,i-1] * (q[j+1,i] - q[j,i-1])) / (4*dx*dy)

J = (Jpp + Jpx + Jxp) / 3
```

- Uses `jnp` slice operations; writes only to `[1:-1, 1:-1]`
- Both ψ and q must have ghost cells set by the caller before calling

## Acceptance Criteria

- [ ] `arakawa_jacobian` function and `ArakawaJacobian2D` class in `finitevolx/_src/jacobian.py`
- [ ] Exports from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_jacobian.py` verifying:
  - `J(ψ, ψ) ≈ 0` (anti-symmetry: Jacobian with identical arguments is zero)
  - `J(ψ, q) = -J(q, ψ)` (skew-symmetry)
  - Energy conservation: `sum(ψ * J(ψ, q)) ≈ 0` for periodic BCs
  - Enstrophy conservation: `sum(q * J(ψ, q)) ≈ 0` for periodic BCs
  - Matches analytical result for simple polynomial inputs

## Priority

**Lower** — A classical alternative to upwind WENO for QG PV advection. The WENO-based `Advection2D` already handles QG advection, so this is an enhancement rather than a necessity.
