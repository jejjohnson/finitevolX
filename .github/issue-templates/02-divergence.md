## Description

finitevolX currently has divergence computed *implicitly* inside the `Advection` classes, but there is no standalone divergence operator. All three reference repositories expose explicit divergence functions, which are needed independently of advection for:

- Mass continuity checks in shallow water models: `∂h/∂t + H ∇·u = 0`
- Divergence-free velocity verification
- Computing the velocity divergence as a diagnostic field
- PV equation terms that require the divergence of the flux separately

The divergence on the Arakawa C-grid reads face-normal velocities (U-points and V-points) and produces a T-point tendency:
```
div[j, i] = (u[j, i] - u[j, i-1]) / dx + (v[j, i] - v[j-1, i]) / dy
```

## References

- [`louity/MQGeometry/flux.py`](https://github.com/louity/MQGeometry/blob/main/flux.py) — `div_flux_3pts`, `div_flux_5pts`
- [`louity/qgsw-pytorch/src/finite_diff.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/finite_diff.py) — `div_nofluxbc`

## Proposed API

```python
class Divergence2D(eqx.Module):
    """Divergence operator on the Arakawa C-grid: (u, v) → ∇·u at T-points.
    
    div[j, i] = (u[j, i] - u[j, i-1]) / dx + (v[j, i] - v[j-1, i]) / dy
    
    Reads U-points (east faces) and V-points (north faces);
    writes result to T-points (cell centres).
    """
    grid: ArakawaCGrid2D

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Returns divergence at T-points."""


class Divergence3D(eqx.Module):
    """3-D divergence operator, applied per horizontal layer."""
    grid: ArakawaCGrid3D

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Returns divergence at T-points for each layer."""
```

## Implementation Notes

- The stencil is the **backward** difference: reads west ghost of u (U-point) and south ghost of v (V-point)
- These ghost cells must be set by the caller (via BC layer) before calling divergence
- Relation to existing code: the x-component is `diff_x_U_to_T` and the y-component is `diff_y_V_to_T` from `Difference2D` — `Divergence2D` is simply their sum
- For `div_nofluxbc` behavior, the caller applies no-flux BCs (zero normal velocity) before calling

## Acceptance Criteria

- [ ] `Divergence2D` class in `finitevolx/_src/difference.py`
- [ ] `Divergence3D` class in `finitevolx/_src/difference.py`
- [ ] Exports from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_divergence.py` verifying:
  - `div(grad_perp(ψ)) ≈ 0` for any smooth `ψ` (non-divergent geostrophic flow)
  - `div(∇φ) ≈ ∇²φ` (Laplacian identity)
  - Correct shapes and ghost ring behavior
- [ ] NumPy-style docstrings

## Priority

**High** — Required for shallow water mass continuity and diagnostic checks.
