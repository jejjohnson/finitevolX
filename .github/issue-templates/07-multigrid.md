## Description

finitevolX has excellent spectral solvers (`solve_helmholtz_dst`, `solve_helmholtz_dct`, `solve_helmholtz_fft`) and the capacitance matrix method for masked domains. However, all spectral solvers assume **constant coefficients**. A multigrid solver for the generalized equation:

```
∇·(c(x,y) ∇u) − λu = rhs
```

where `c` varies spatially, is needed for:
- Shallow water models with spatially varying layer thickness
- Variable-coefficient diffusion
- Non-uniform grid applications
- PV inversion on masked domains without the capacitance matrix overhead

The `qgsw-pytorch` repository provides a full PyTorch implementation with Jacobi smoothing, V-cycle recursion, restriction/prolongation, and direct bottom-level solve. This should be ported to JAX.

## References

- [`louity/qgsw-pytorch/src/helmholtz_multigrid.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/helmholtz_multigrid.py) — Full implementation: `MG_Helmholtz` class with Jacobi smoothing, V-cycle, FMG, restriction/prolongation, mask hierarchy, coefficient hierarchy, bottom-solve

## Proposed API

```python
class MultigridHelmholtz(eqx.Module):
    """Multigrid solver for ∇·(c ∇u) − λu = rhs with variable coefficients.
    
    Supports masked domains (irregular geometries) and spatially varying
    diffusion coefficients c(x, y).
    
    Parameters
    ----------
    grid : ArakawaCGrid2D
        The computational grid.
    coef_u : Float[Array, "Ny Nx"]
        Diffusion coefficient at U-points (east faces).
    coef_v : Float[Array, "Ny Nx"]
        Diffusion coefficient at V-points (north faces).
    mask : Bool[Array, "Ny Nx"] | None
        Ocean mask (True = ocean cell, False = land). None = no masking.
    lambda_ : float
        Helmholtz constant (negative for Helmholtz; 0 for Poisson).
    n_levels : int
        Number of multigrid levels.
    n_smooth : int
        Number of Jacobi smoothing iterations per level.
    """

    def solve(
        self,
        rhs: Float[Array, "Ny Nx"],
        *,
        n_vcycles: int = 10,
        tol: float = 1e-6,
    ) -> Float[Array, "Ny Nx"]:
        """Solve the Helmholtz equation using V-cycles.
        
        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side at T-points.
        n_vcycles : int
            Number of V-cycles to perform.
        tol : float
            Convergence tolerance on the relative residual.
        
        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution at T-points.
        """
```

## Implementation Notes

- Use `jax.lax.fori_loop` for the inner Jacobi smoothing iterations (JIT-compatible)
- Use `jax.lax.while_loop` or `jax.lax.fori_loop` for V-cycle iterations
- **Restriction** (fine→coarse): cell-averaged restriction operator
- **Prolongation** (coarse→fine): bilinear interpolation
- Build coefficient hierarchy at construction time (not at solve time)
- Build mask hierarchy at construction time using `ArakawaCGridMask`
- Bottom-level solve: use direct inversion (`jnp.linalg.solve`) for small coarse grids
- Jacobi smoother: `u ← u + ω * (rhs - Lu) / diag(L)` where `ω ≈ 0.8`

## Acceptance Criteria

- [ ] `MultigridHelmholtz` class in `finitevolx/_src/elliptic/multigrid.py`
- [ ] Export from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_multigrid.py` verifying:
  - Converges to machine precision on a simple Poisson problem with known solution
  - Handles masked domain (island in the domain)
  - Variable coefficient case: `c = 1 + x²` converges correctly
  - JIT-compatible: `jax.jit(solver.solve)(rhs)` runs without recompilation
- [ ] Benchmark showing speedup vs. `solve_cg` for large grids

## Priority

**Medium** — The existing spectral solvers are sufficient for most rectangular domains. This becomes important for variable-coefficient or strongly non-rectangular domain problems.
