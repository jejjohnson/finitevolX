## Description

The perpendicular gradient operator `(u, v) = (-∂ψ/∂y, ∂ψ/∂x)` computes geostrophic velocity from a streamfunction `ψ` on the staggered Arakawa C-grid. This forward mapping (streamfunction → velocity) is the **fundamental building block** of every QG model, yet it is currently absent from finitevolX.

The existing `Vorticity2D` class computes the *inverse* direction (`curl(u,v) → ζ`), but there is no operator for the forward direction `ψ → (u, v)`.

The perpendicular gradient appears in **all three** reference repositories:

- The `u` component is `-∂ψ/∂y` (forward difference in y: T→V, then negated)
- The `v` component is `+∂ψ/∂x` (forward difference in x: T→U)

On the Arakawa C-grid with same-index convention:
- `u[j, i+1/2] = -(ψ[j+1, i] - ψ[j, i]) / dy`  → result lives at **U-points**
- `v[j+1/2, i] = +(ψ[j, i+1] - ψ[j, i]) / dx`  → result lives at **V-points**

## References

- [`louity/MQGeometry/fd.py`](https://github.com/louity/MQGeometry/blob/main/fd.py) — `grad_perp` function
- [`louity/qgm_pytorch/QGM.py`](https://github.com/louity/qgm_pytorch/blob/main/QGM.py) — `grad_perp` inline
- [`louity/qgsw-pytorch/src/finite_diff.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/finite_diff.py) — `grad_perp` function

## Proposed API

### Option A: Standalone functions (consistent with existing `Difference2D` pattern)

```python
def grad_perp_x(psi: Float[Array, "Ny Nx"], dy: float) -> Float[Array, "Ny Nx"]:
    """Perpendicular gradient x-component: u = -∂ψ/∂y (T → U points).
    
    u[j, i+1/2] = -(ψ[j+1, i] - ψ[j, i]) / dy
    """

def grad_perp_y(psi: Float[Array, "Ny Nx"], dx: float) -> Float[Array, "Ny Nx"]:
    """Perpendicular gradient y-component: v = +∂ψ/∂x (T → V points).
    
    v[j+1/2, i] = +(ψ[j, i+1] - ψ[j, i]) / dx
    """
```

### Option B: `GradPerp2D` equinox Module (preferred, consistent with `Vorticity2D`)

```python
class GradPerp2D(eqx.Module):
    """Perpendicular gradient on the Arakawa C-grid: ψ → (u, v).
    
    Maps a T-point streamfunction to geostrophic velocities:
      u[j, i+1/2] = -∂ψ/∂y  (U-points)
      v[j+1/2, i] = +∂ψ/∂x  (V-points)
    """
    grid: ArakawaCGrid2D

    def __call__(self, psi: Float[Array, "Ny Nx"]) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Returns (u, v) velocity tuple."""
```

## Implementation Notes

- The stencils are analogous to `diff_y_T_to_V` and `diff_x_T_to_U` in `Difference2D`, but with a sign flip on the y-component
- Must write only to `[1:-1, 1:-1]` of the output array (ghost ring initialized to zero)
- North ghost of `ψ` (T-point) is consumed by u-component at the last interior row
- East ghost of `ψ` (T-point) is consumed by v-component at the last interior column
- Add an optional mask-aware variant for use with `ArakawaCGridMask`

## Acceptance Criteria

- [ ] `GradPerp2D` class in `finitevolx/_src/difference.py` (or a new `finitevolx/_src/operators.py`)
- [ ] Exports `GradPerp2D` from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_grad_perp.py` verifying:
  - `curl(grad_perp(ψ)) ≈ ∇²ψ` (round-trip identity)
  - Correct array shapes and ghost ring behavior
  - Works with both uniform and rectangular grids
- [ ] NumPy-style docstring with half-index stencil formula
- [ ] Type hints using `jaxtyping`

## Priority

**High** — Required by every QG model. This is a trivial addition (two stencil slices) that unlocks the entire QG workflow.
