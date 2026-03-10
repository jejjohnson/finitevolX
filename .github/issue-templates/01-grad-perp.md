## Description

The perpendicular gradient operator `(u, v) = (-∂ψ/∂y, ∂ψ/∂x)` computes geostrophic velocity from a streamfunction `ψ` on the staggered Arakawa C-grid. This forward mapping (streamfunction → velocity) is the **fundamental building block** of every QG model, yet it is currently absent from finitevolX.

The existing `Vorticity2D` class computes the *inverse* direction (`curl(u,v) → ζ`), but there is no operator for the forward direction `ψ → (u, v)`.

The perpendicular gradient appears in **all three** reference repositories:

- The `u` component (zonal velocity) is `-∂ψ/∂y`, at **U-points** `(j, i+1/2)`
- The `v` component (meridional velocity) is `+∂ψ/∂x`, at **V-points** `(j+1/2, i)`

The proper Arakawa C-grid approach (matching the existing `qg_1p5_layer.py` script) uses
a two-step T→X→U/V route through the corner grid:

```
# Step 1: T → X (corner) interpolation
psi_x[j+1/2, i+1/2] = (ψ[j,i] + ψ[j+1,i] + ψ[j,i+1] + ψ[j+1,i+1]) / 4

# Step 2a: diff_y_X_to_U  (u at east faces)
u[j, i+1/2] = -(psi_x[j+1/2, i+1/2] - psi_x[j-1/2, i+1/2]) / dy

# Step 2b: diff_x_X_to_V  (v at north faces)
v[j+1/2, i] = +(psi_x[j+1/2, i+1/2] - psi_x[j+1/2, i-1/2]) / dx
```

A simpler (but less accurate) forward-difference approximation used in the reference
repos produces the result on the opposite faces — `u` at V-points and `v` at U-points:

```
u[j+1/2, i] = -(ψ[j+1, i] - ψ[j, i]) / dy   → V-points  (simple approach)
v[j, i+1/2] = +(ψ[j, i+1] - ψ[j, i]) / dx   → U-points  (simple approach)
```

The preferred finitevolX implementation should use the T→X→U/V route to place `u` on
U-points and `v` on V-points, consistent with the existing `Interpolation2D` and
`Difference2D` operators.

## References

- [`louity/MQGeometry/fd.py`](https://github.com/louity/MQGeometry/blob/main/fd.py) — `grad_perp` function
- [`louity/qgm_pytorch/QGM.py`](https://github.com/louity/qgm_pytorch/blob/main/QGM.py) — `grad_perp` inline
- [`louity/qgsw-pytorch/src/finite_diff.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/finite_diff.py) — `grad_perp` function

## Proposed API

### Option A: Standalone functions (consistent with existing `Difference2D` pattern)

```python
def grad_perp_x(psi: Float[Array, "Ny Nx"], dy: float) -> Float[Array, "Ny Nx"]:
    """Perpendicular gradient x-component: u = -∂ψ/∂y (T → X → U points).
    
    u[j, i] = -(psi_x[j, i] - psi_x[j-1, i]) / dy
    where psi_x is ψ interpolated to X-corners via interp_T_to_X.
    u[j, i] represents the east-face value at physical location (j, i+1/2).
    """

def grad_perp_y(psi: Float[Array, "Ny Nx"], dx: float) -> Float[Array, "Ny Nx"]:
    """Perpendicular gradient y-component: v = +∂ψ/∂x (T → X → V points).
    
    v[j, i] = +(psi_x[j, i] - psi_x[j, i-1]) / dx
    where psi_x is ψ interpolated to X-corners via interp_T_to_X.
    v[j, i] represents the north-face value at physical location (j+1/2, i).
    """
```

### Option B: `GradPerp2D` equinox Module (preferred, consistent with `Vorticity2D`)

```python
class GradPerp2D(eqx.Module):
    """Perpendicular gradient on the Arakawa C-grid: ψ → (u, v).
    
    Maps a T-point streamfunction to geostrophic velocities via the
    T → X (corner) → U/V route:
      u[j, i] = -∂psi_x/∂y  at U-points (east faces, physical location j, i+1/2)
      v[j, i] = +∂psi_x/∂x  at V-points (north faces, physical location j+1/2, i)
    where psi_x is ψ interpolated from T-points to X-corners.
    """
    grid: ArakawaCGrid2D

    def __call__(self, psi: Float[Array, "Ny Nx"]) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Returns (u, v) velocity tuple."""
```

## Implementation Notes

- Follows the same T→X→U/V route already used in `qg_1p5_layer.py`:
  1. `psi_x = Interpolation2D(grid).interp_T_to_X(psi)` — corners at `(j+1/2, i+1/2)`
  2. `u = -Difference2D(grid).diff_y_X_to_U(psi_x)` — east faces at `(j, i+1/2)`
  3. `v = +Difference2D(grid).diff_x_X_to_V(psi_x)` — north faces at `(j+1/2, i)`
- `GradPerp2D` wraps these two operators and avoids re-computing `psi_x` twice
- Must write only to `[1:-1, 1:-1]` of output arrays (ghost ring initialized to zero)
- North ghost of `ψ` (T-point) is consumed by the X-interpolation
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
