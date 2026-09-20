## Description

finitevolX currently has `kinetic_energy` and `bernoulli_potential` in the operators module, but these are limited (depend on `finitediffx`). A comprehensive set of **diagnostic utilities** for verifying conservation properties, validating model output, and monitoring simulation health is needed.

Essential diagnostics for ocean models include:
1. **Kinetic energy** on the staggered C-grid (with proper interpolation of face velocities to cell centres)
2. **Potential energy** (available potential energy from interface displacements)
3. **Enstrophy** (mean squared vorticity, conserved by 2D flow)
4. **Total energy** (sum of KE + PE, should be conserved by non-dissipative models)

Both MQGeometry and qgsw-pytorch compute these diagnostics. They serve as key verification tools for any finite volume ocean simulation.

## References

- [`louity/qgsw-pytorch/src/sw.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/sw.py) — `comp_ke` (kinetic energy on staggered grid)
- [`louity/MQGeometry/qgm.py`](https://github.com/louity/MQGeometry/blob/main/qgm.py) — energy and enstrophy diagnostics in the solver

## Proposed API

```python
# finitevolx/_src/diagnostics.py

def kinetic_energy(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    H: float | Float[Array, "Ny Nx"] = 1.0,
    dx: float = 1.0,
    dy: float = 1.0,
    mask: Bool[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Kinetic energy density at T-points on the staggered C-grid.
    
    KE[j,i] = (H/2) * (interp(u²)[j,i] + interp(v²)[j,i])
    
    Interpolates u² from U-points and v² from V-points to T-points
    before averaging, giving the correct staggered-grid KE.
    
    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Zonal velocity at U-points.
    v : Float[Array, "Ny Nx"]
        Meridional velocity at V-points.
    H : float or array
        Layer thickness [m]. Scalar for uniform depth.
    dx, dy : float
        Grid spacing [m].
    mask : Bool[Array, "Ny Nx"], optional
        Ocean mask. If provided, sets land cells to zero.
    
    Returns
    -------
    Float[Array, "Ny Nx"]
        Kinetic energy density [m³/s²] at T-points.
    """


def potential_energy(
    h: Float[Array, "Ny Nx"],
    g_prime: float,
    H_ref: float,
    mask: Bool[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Available potential energy density at T-points.
    
    APE[j,i] = (g' / 2) * (h[j,i] - H_ref)²
    
    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Interface height or layer thickness anomaly at T-points.
    g_prime : float
        Reduced gravity [m/s²].
    H_ref : float
        Reference layer thickness [m].
    mask : Bool[Array, "Ny Nx"], optional
        Ocean mask.
    
    Returns
    -------
    Float[Array, "Ny Nx"]
        Available potential energy density [m³/s²] at T-points.
    """


def enstrophy(
    q: Float[Array, "Ny Nx"],
    h: Float[Array, "Ny Nx"] | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    mask: Bool[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Enstrophy density at T-points.
    
    E[j,i] = (1/2) * h[j,i] * q[j,i]²
    
    For QG models (h = const), this reduces to E = q²/2.
    
    Parameters
    ----------
    q : Float[Array, "Ny Nx"]
        Potential vorticity at T-points.
    h : Float[Array, "Ny Nx"], optional
        Layer thickness. If None, uses unit thickness.
    dx, dy : float
        Grid spacing [m] (reserved for future weighted enstrophy).
    mask : Bool[Array, "Ny Nx"], optional
        Ocean mask.
    
    Returns
    -------
    Float[Array, "Ny Nx"]
        Enstrophy density at T-points.
    """


def total_energy(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    h: Float[Array, "Ny Nx"],
    g_prime: float,
    H_ref: float,
    dx: float,
    dy: float,
    mask: Bool[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Total mechanical energy density (KE + APE) at T-points."""
```

## Implementation Notes

- **KE staggering**: `u` lives at U-points, `v` at V-points. For correct energy, interpolate each to T-points before squaring — or equivalently use the `Interpolation2D` operators already in finitevolX
- All functions write only to `[1:-1, 1:-1]`; ghost ring is zero
- All scalar integrals can be obtained via `jnp.sum(field[1:-1, 1:-1] * mask[1:-1, 1:-1]) * dx * dy`
- For 3-D multi-layer versions, add `Nz` leading dimension with `jax.vmap`

## Acceptance Criteria

- [ ] Functions in `finitevolx/_src/diagnostics.py`
- [ ] Exports from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_diagnostics.py` verifying:
  - `kinetic_energy(u, v)` of uniform flow matches analytical result
  - `kinetic_energy(u, v)` with zero mask gives zero
  - `enstrophy(q)` of constant vorticity field matches `0.5 * q² * Ny * Nx`
  - `total_energy` decreases monotonically in a dissipative simulation (integration test)
  - Non-constant test: spatially varying u/v/q fields match analytically derived values

## Priority

**Lower** — Not blocking model creation but essential for validating any simulation. The existing operator module has partial coverage (`kinetic_energy`, `bernoulli_potential`) that should be superseded by these cleaner, mask-aware functions.
