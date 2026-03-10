## Description

The `ArakawaCGridMask` class already computes `StencilCapability` (which cells can support 2-point, 4-point, or 6-point stencils), but there is no flux function that *uses* this information at runtime. Near domain boundaries, wide stencils are invalid and the code must fall back to narrower stencils to avoid reading outside the physical domain.

Both reference repositories implement this pattern:
- `qgsw-pytorch`: `flux()` function blends reconstructions using `mask_2`, `mask_4`, `mask_6` boolean arrays
- `MQGeometry`: `flux_3pts()` / `flux_5pts()` with boundary fallback via the `distbound1/2/3+` hierarchy

Currently, finitevolX's `Reconstruction1D/2D/3D` classes support masking but do not automatically dispatch to narrower stencils near boundaries. This makes them unsafe for non-rectangular domains without manual padding.

## References

- [`louity/qgsw-pytorch/src/flux.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/flux.py) — `flux()` with `mask_2/4/6` blending
- [`louity/qgsw-pytorch/src/reconstruction.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/reconstruction.py) — all reconstruction functions
- [`louity/MQGeometry/flux.py`](https://github.com/louity/MQGeometry/blob/main/flux.py) — `flux_3pts`, `flux_5pts`
- [`louity/MQGeometry/masks.py`](https://github.com/louity/MQGeometry/blob/main/masks.py) — `Masks` class with `distbound1`, `distbound2`, `distbound3plus`

## Proposed API

```python
def upwind_flux_x(
    q: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    stencil_cap: StencilCapability,
    order: int = 5,
) -> Float[Array, "Ny Nx"]:
    """Upwind flux in x with automatic stencil fallback near boundaries.
    
    At cells where StencilCapability.width < order, falls back to the
    widest available stencil:  5-pt → 3-pt → 1-pt (upwind).
    
    Parameters
    ----------
    q : Float[Array, "Ny Nx"]
        Advected scalar at T-points.
    u : Float[Array, "Ny Nx"]
        Velocity at U-points (east faces).
    stencil_cap : StencilCapability
        Per-cell stencil width capability from ArakawaCGridMask.
    order : int
        Maximum reconstruction order (1, 3, or 5).
    """


def upwind_flux_y(
    q: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    stencil_cap: StencilCapability,
    order: int = 5,
) -> Float[Array, "Ny Nx"]:
    """Upwind flux in y with automatic stencil fallback near boundaries."""
```

## Implementation Notes

- The blending uses `jnp.where` with the `StencilCapability` masks: where stencil width ≥ 6, use WENO-5; where width ≥ 4 and < 6, use WENO-3; else use 1st-order upwind
- All branches compute all orders and blend with `jnp.where` for JAX JIT compatibility (no Python-level branching on array values)
- The `StencilCapability` object already exists; this issue just wires it to the reconstruction logic
- Both x and y directions need variants

## Acceptance Criteria

- [ ] `upwind_flux_x` and `upwind_flux_y` functions in `finitevolx/_src/flux.py` (new file)
- [ ] Exports from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_flux.py` verifying:
  - On a rectangular domain (all cells have width ≥ 6), result matches direct WENO-5 reconstruction
  - Near an island boundary, the flux correctly degrades to lower-order stencil
  - Mass conservation: `sum(div(F)) ≈ 0` for periodic/no-flux boundaries

## Priority

**Medium-High** — Critical for non-rectangular ocean geometries. The `ArakawaCGridMask.StencilCapability` infrastructure is already there; this issue completes the connection.
