## Description

finitevolX has `Dirichlet1D`, `Neumann1D`, `Periodic1D`, `Reflective1D`, `Outflow1D`, and `Sponge1D` boundary conditions, but **no slip-coefficient boundary condition** for ocean model lateral walls.

Ocean models use a mixed boundary condition that interpolates between:
- **Free-slip** (`slip_coef = 1.0`): tangential velocity is unchanged at the boundary, equivalent to zero vorticity flux
- **No-slip** (`slip_coef = 0.0`): tangential velocity is zero at the wall
- **Partial-slip** (`slip_coef ∈ (0, 1)`): a weighted combination

For a south wall with `slip_coef = α`, the ghost cell vorticity is:
```
q_ghost = α * q_interior - (1 - α) * (2 * U_wall / dy)
```

Both `qgm_pytorch` (`bcco`) and `qgsw-pytorch` (`slip_coef`) implement this.

## References

- [`louity/qgm_pytorch/QGM.py`](https://github.com/louity/qgm_pytorch/blob/main/QGM.py) — `bcco` parameter controlling slip
- [`louity/qgsw-pytorch/src/sw.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/sw.py) — `slip_coef` parameter ∈ [0,1]

## Proposed API

```python
class SlipBC1D(eqx.Module):
    """Slip / partial-slip / no-slip lateral wall boundary condition.
    
    Interpolates between free-slip (coefficient=1) and no-slip (coefficient=0)
    by setting the ghost cell value of the vorticity field.
    
    Parameters
    ----------
    coefficient : float
        Slip coefficient in [0, 1].
        - 1.0: free-slip (zero vorticity flux)
        - 0.0: no-slip (zero tangential velocity at wall)
        - (0, 1): partial-slip
    face : Literal["south", "north", "west", "east"]
        Which wall face this BC applies to.
    """
    coefficient: float
    face: str

    def __call__(
        self,
        field: Float[Array, "Ny Nx"],
        u_wall: float = 0.0,
        dy: float = 1.0,
    ) -> Float[Array, "Ny Nx"]:
        """Apply slip BC to the ghost row/column of the vorticity field."""
```

## Implementation Notes

- Extends `finitevolx/_src/bc_1d.py` following the existing BC atom pattern
- Should integrate with `BoundaryConditionSet` and `FieldBCSet` in the existing BC infrastructure
- For free-slip (`coefficient = 1.0`): `q_ghost = q_interior` (reflection)
- For no-slip (`coefficient = 0.0`): `q_ghost = -q_interior - 2*u_wall/dy`
- For partial-slip: `q_ghost = coef * q_interior - (1 - coef) * (2*u_wall/dy)`
- The `u_wall` parameter allows non-zero wall velocity (e.g., moving lid)

## Acceptance Criteria

- [ ] `SlipBC1D` class in `finitevolx/_src/bc_1d.py`
- [ ] Exports from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_boundary.py` verifying:
  - `SlipBC1D(1.0)` is equivalent to a reflective BC (free-slip)
  - `SlipBC1D(0.0)` sets ghost cell to `-q_interior` (no-slip, zero wall velocity)
  - Intermediate values blend correctly
  - Works for all four faces

## Priority

**Medium** — Every ocean model uses slip BCs at lateral walls. Currently no way to express this in finitevolX.
