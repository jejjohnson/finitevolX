# Epic #115: Solvers & Linear Algebra — Implementation Plan

## Status Summary

| Sub-Issue | Title | State | Action Needed |
|-----------|-------|-------|---------------|
| #85 | TDMA solver | **CLOSED** | None — already in `_src/solvers/tridiagonal.py` |
| #10 | Elliptical solvers | **CLOSED** | None — fully covered by `_src/solvers/elliptic.py` |
| #72 | Capacitance matrix method | **CLOSED** | None — `CapacitanceSolver` + `build_capacitance_solver` exist |
| **#87** | **Streamfunction & pressure solvers** | **OPEN** | **Convenience wrappers needed** (small) |
| **#71** | **Multigrid Helmholtz solver** | **OPEN** | **Major new implementation** (large) |

**Bottom line**: 3 of 5 sub-issues are already resolved. The remaining work is:
1. **#87** — thin convenience wrappers over existing solvers (1-2 days)
2. **#71** — multigrid Helmholtz for variable-coefficient masked domains (1-2 weeks)

---

## Part 1: Issue #87 — Streamfunction & Pressure Solver Wrappers

### Context

Per the owner's own audit comment on #87, the core solver infrastructure already exists. What's missing are **convenience wrappers** that map physics-level operations (streamfunction inversion, pressure correction) to the existing spectral/capacitance/CG solvers.

### 1.1 New Functions

Add to `finitevolx/_src/solvers/elliptic.py`:

#### `streamfunction_from_vorticity`

```python
def streamfunction_from_vorticity(
    zeta: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    bc: str = "dst",
    lambda_: float = 0.0,
    mask: Float[Array, "Ny Nx"] | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
) -> Float[Array, "Ny Nx"]:
    """Invert ∇²ψ − λψ = ζ for the streamfunction.

    Solves the Poisson (λ=0) or Helmholtz (λ≠0) equation to recover the
    streamfunction from relative vorticity.

    For rectangular domains, uses the spectral solver selected by *bc*.
    For masked domains, supply either a pre-built ``capacitance_solver``
    or use ``solve_cg`` via the ``mask`` parameter.

    Parameters
    ----------
    zeta : array, shape (Ny, Nx)
        Relative vorticity (RHS).
    dx, dy : float
        Grid spacings.
    bc : {"dst", "dct", "fft"}
        Boundary condition type for spectral solver.
        "dst" = Dirichlet (ψ=0 on boundary), most common for streamfunction.
    lambda_ : float
        Helmholtz parameter. 0.0 for pure Poisson (streamfunction from ζ).
        Non-zero for QG PV inversion: (∇² − λ²)ψ = q.
    mask : array or None
        If given, uses CG with spectral preconditioner on the masked domain.
    capacitance_solver : CapacitanceSolver or None
        If given, uses the pre-built capacitance solver (fastest for masked domains
        with moderate boundary point count).

    Returns
    -------
    psi : array, shape (Ny, Nx)
        Streamfunction.
    """
```

**Implementation logic:**
1. If `capacitance_solver` is not None → `capacitance_solver(zeta)`
2. Elif `mask` is not None → `solve_cg(masked_laplacian_closure, zeta, preconditioner=make_spectral_preconditioner(...))`
3. Else → `_spectral_solve(zeta, dx, dy, lambda_, bc)`

#### `pressure_from_divergence`

```python
def pressure_from_divergence(
    div_u: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    bc: str = "dct",
    mask: Float[Array, "Ny Nx"] | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²p = ∇·u for the pressure correction.

    Used in pressure-projection methods (Chorin splitting) where the
    divergence of the provisional velocity field must be removed.

    Parameters
    ----------
    div_u : array, shape (Ny, Nx)
        Divergence of the velocity field (RHS).
    dx, dy : float
        Grid spacings.
    bc : {"dct", "dst", "fft"}
        Boundary condition type. "dct" (Neumann, ∂p/∂n=0) is the standard
        choice for pressure with solid walls.
    mask, capacitance_solver : optional
        Same as ``streamfunction_from_vorticity``.

    Returns
    -------
    p : array, shape (Ny, Nx)
        Pressure field.
    """
```

**Implementation logic:** Same dispatch as `streamfunction_from_vorticity` but defaults to `bc="dct"` (Neumann).

#### `pv_inversion`

```python
def pv_inversion(
    pv: Float[Array, "... Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Float[Array, " nl"],
    bc: str = "dst",
    mask: Float[Array, "Ny Nx"] | None = None,
) -> Float[Array, "... Ny Nx"]:
    """QG potential vorticity inversion: solve (∇² − λ²)ψ = q.

    Supports batched/multi-layer PV fields via leading dimensions.
    When lambda_ is an array of shape (nl,), each layer uses its own
    Helmholtz parameter (e.g., from vertical mode decomposition).

    Parameters
    ----------
    pv : array, shape (..., Ny, Nx)
        Potential vorticity field. Leading dimensions are batched.
    dx, dy : float
        Grid spacings.
    lambda_ : float or array of shape (nl,)
        Helmholtz parameter(s). For multi-layer QG, these are
        1/Rd² where Rd is the Rossby deformation radius per mode.
    bc : {"dst", "dct", "fft"}
        Boundary condition type.
    mask : array or None
        Optional domain mask.

    Returns
    -------
    psi : array, shape (..., Ny, Nx)
        Streamfunction field.
    """
```

**Implementation logic:** Uses `jax.vmap` over leading dimensions when `lambda_` is an array.

### 1.2 Public API Exports

Add to `finitevolx/__init__.py`:
- `streamfunction_from_vorticity`
- `pressure_from_divergence`
- `pv_inversion`

### 1.3 Tests

New file: `tests/test_solver_wrappers.py`

```
class TestStreamfunctionFromVorticity:
    test_roundtrip_dst          # ∇²ψ_exact → ζ, invert → ψ ≈ ψ_exact
    test_roundtrip_dct          # Same with Neumann BCs
    test_roundtrip_fft          # Same with periodic BCs
    test_with_capacitance       # Masked domain, pre-built solver
    test_with_mask_cg           # Masked domain, CG fallback
    test_helmholtz_mode         # lambda_ ≠ 0

class TestPressureFromDivergence:
    test_roundtrip_neumann      # ∇²p_exact → div, invert → p ≈ p_exact
    test_zero_divergence        # div=0 → p=0 (up to constant)

class TestPVInversion:
    test_single_layer           # Single QG mode
    test_multilayer_vmap        # Array of lambda_ values
    test_with_mask              # Masked domain
```

### 1.4 Effort Estimate

**Small** — these are thin wrappers routing to existing, tested solvers. The main value is API ergonomics.

---

## Part 2: Issue #71 — Multigrid Helmholtz Solver

### Context

The existing solver stack (spectral + capacitance + CG) handles:
- Rectangular domains with constant coefficients (spectral) — O(N log N)
- Irregular domains with constant coefficients (capacitance) — O(N log N + N_b²)
- General domains (CG) — O(k · N) where k = iterations

What's **missing** is a solver for **variable-coefficient** Helmholtz on **masked domains**:

> ∇·(c(x,y) ∇u) − λ u = rhs

where `c(x,y)` varies in space (e.g., spatially varying diffusivity, bottom topography effects). Spectral methods can't handle variable coefficients; CG can but converges slowly without a good preconditioner. **Geometric multigrid** is the standard approach.

### Reference

The [louity/qgsw-pytorch](https://github.com/louity/qgsw-pytorch) implementation (`helmholtz_multigrid.py`, ~560 lines) provides the reference algorithm. It's PyTorch but the algorithm maps directly to JAX.

### 2.1 New Module: `finitevolx/_src/solvers/multigrid.py`

#### Core Pure Functions

```python
def jacobi_smooth(
    f: Float[Array, "Ny Nx"],
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    mask: Float[Array, "Ny Nx"],
    coef_u: Float[Array, "Ny Nx"],   # diffusion coeff on U-faces
    coef_v: Float[Array, "Ny Nx"],   # diffusion coeff on V-faces
    lambda_: float,
    omega: float = 0.8,
    n_iter: int = 3,
) -> Float[Array, "Ny Nx"]:
    """Weighted Jacobi relaxation for ∇·(c∇f) − λf = rhs on masked grid.

    Stencil (5-point, variable-coefficient):
        L[f]_{j,i} = (c_u[j,i+½] (f[j,i+1] − f[j,i]) − c_u[j,i−½] (f[j,i] − f[j,i−1])) / dx²
                    + (c_v[j+½,i] (f[j+1,i] − f[j,i]) − c_v[j−½,i] (f[j,i] − f[j−1,i])) / dy²
                    − λ f[j,i]

    Updates: f_new = f + ω · (rhs − L[f]) / diag(L)
    """
```

```python
def helmholtz_residual(
    f: Float[Array, "Ny Nx"],
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    mask: Float[Array, "Ny Nx"],
    coef_u: Float[Array, "Ny Nx"],
    coef_v: Float[Array, "Ny Nx"],
    lambda_: float,
) -> Float[Array, "Ny Nx"]:
    """Compute residual r = rhs − L[f] on the masked domain."""
```

```python
def restrict_2d(
    field: Float[Array, "Ny Nx"],
    mask_fine: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny_c Nx_c"]:
    """Cell-centred full-weighting restriction (fine → coarse).

    Coarse cell = weighted average of 4 fine cells in the 2×2 block,
    considering only cells inside the mask.
    """
```

```python
def prolong_2d(
    field: Float[Array, "Ny_c Nx_c"],
    mask_fine: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """Cell-centred prolongation (coarse → fine) via bilinear interpolation.

    Uses a 9-point stencil (nearest 4 coarse cells + diagonal neighbors),
    weighted by mask.
    """
```

#### Multigrid Class

```python
class MultigridHelmholtz2D(eqx.Module):
    """Geometric multigrid V-cycle solver for ∇·(c∇u) − λu = rhs.

    Supports:
    - Variable coefficients c(x,y) on staggered U/V faces
    - Masked (irregular) domains via ArakawaCGridMask
    - V-cycle and Full Multigrid (FMG) algorithms
    - Bottom solve via direct matrix inversion (small coarsest grid)
      or iterated Jacobi (larger coarsest grid)

    Attributes
    ----------
    mask_hierarchy : list[Float[Array, "Ny_l Nx_l"]]
        Mask at each multigrid level (fine → coarse).
    coef_u_hierarchy : list[Float[Array, "Ny_l Nx_l"]]
        U-face coefficients at each level.
    coef_v_hierarchy : list[Float[Array, "Ny_l Nx_l"]]
        V-face coefficients at each level.
    dx_hierarchy : list[float]
        Grid spacing in x at each level.
    dy_hierarchy : list[float]
        Grid spacing in y at each level.
    lambda_ : float
        Helmholtz parameter.
    n_levels : int
        Number of multigrid levels.
    n_pre : int
        Pre-smoothing iterations (default: 3).
    n_post : int
        Post-smoothing iterations (default: 3).
    omega : float
        Jacobi relaxation weight (default: 0.8).
    bottom_solver : str
        "direct" (pinv) or "jacobi" (iterated).
    """

    # --- Stored hierarchies (precomputed) ---
    mask_hierarchy: tuple[Float[Array, "..."], ...]
    coef_u_hierarchy: tuple[Float[Array, "..."], ...]
    coef_v_hierarchy: tuple[Float[Array, "..."], ...]
    dx_hierarchy: tuple[float, ...]
    dy_hierarchy: tuple[float, ...]
    lambda_: float = eqx.field(static=True)
    n_levels: int = eqx.field(static=True)
    n_pre: int = eqx.field(static=True, default=3)
    n_post: int = eqx.field(static=True, default=3)
    omega: float = eqx.field(static=True, default=0.8)
    bottom_solver: str = eqx.field(static=True, default="direct")
    _bottom_pinv: Float[Array, "... ..."] | None = None  # precomputed for direct

    def __call__(
        self,
        rhs: Float[Array, "Ny Nx"],
        x0: Float[Array, "Ny Nx"] | None = None,
        tol: float = 1e-6,
        max_cycles: int = 50,
    ) -> tuple[Float[Array, "Ny Nx"], CGInfo]:
        """Solve using repeated V-cycles until convergence."""

    def v_cycle(
        self,
        f: Float[Array, "Ny Nx"],
        rhs: Float[Array, "Ny Nx"],
        level: int = 0,
    ) -> Float[Array, "Ny Nx"]:
        """Single V-cycle: smooth → restrict → recurse → prolong → smooth."""

    def fmg(
        self,
        rhs: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Full Multigrid: coarsest solve → prolong → V-cycle at each level."""
```

#### Factory Function

```python
def build_multigrid_helmholtz(
    mask: np.ndarray | Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    coef_u: np.ndarray | Float[Array, "Ny Nx"] | None = None,
    coef_v: np.ndarray | Float[Array, "Ny Nx"] | None = None,
    n_levels: int | None = None,
    n_pre: int = 3,
    n_post: int = 3,
    omega: float = 0.8,
    bottom_solver: str = "direct",
) -> MultigridHelmholtz2D:
    """Build a multigrid solver with precomputed level hierarchies.

    Offline computation:
    1. Coarsen mask through levels (2×2 pooling, threshold > 0)
    2. Coarsen coefficients (average pooling on U/V staggered grids)
    3. If bottom_solver="direct", precompute pseudoinverse at coarsest level

    Parameters
    ----------
    mask : array, shape (Ny, Nx)
        Domain mask (True/1 = fluid, False/0 = land). If None, rectangular.
    dx, dy : float
        Fine-grid spacings.
    coef_u, coef_v : array or None
        Variable coefficients on U-faces and V-faces.
        None → constant coefficient = 1.0 everywhere.
    n_levels : int or None
        Number of levels. None → auto (coarsen until min(Ny, Nx) < 4).
    bottom_solver : {"direct", "jacobi"}
        "direct" precomputes pinv at coarsest level (fast, memory-limited).
        "jacobi" uses 50+ Jacobi iterations at bottom (slower, no matrix stored).
    """
```

### 2.2 Implementation Strategy

The implementation should proceed in phases:

#### Phase A: Core Stencil Operations (2-3 days)

1. **`helmholtz_residual`** — the variable-coefficient 5-point stencil
   - This is the fundamental building block
   - Must handle masks correctly (zero flux at land boundaries)
   - Stencil on staggered grid: coefficients on U/V faces, field on T-points

2. **`jacobi_smooth`** — weighted Jacobi iteration
   - Uses `helmholtz_residual` internally for the diagonal extraction
   - `lax.fori_loop` for n_iter iterations (JIT-friendly)
   - Mask-aware: only update fluid cells

3. **Tests for Phase A:**
   - `test_residual_zero_for_exact_solution` — known analytic solution
   - `test_jacobi_reduces_residual` — residual decreases monotonically
   - `test_constant_coef_matches_laplacian` — when c=1, matches `masked_laplacian`

#### Phase B: Grid Transfer Operators (1-2 days)

4. **`restrict_2d`** — fine-to-coarse (full weighting)
   - Cell-centred: coarse[j,i] = mean of fine[2j:2j+2, 2i:2i+2] (mask-weighted)
   - Handle odd-sized grids (pad if necessary)

5. **`prolong_2d`** — coarse-to-fine (bilinear interpolation)
   - 9-point stencil interpolation from coarse to fine grid
   - Mask-aware: zero outside mask

6. **Hierarchy construction** — mask and coefficient coarsening
   - `_compute_mask_hierarchy(mask, n_levels)` — 2×2 max-pooling (any wet cell → wet)
   - `_compute_coef_hierarchy(coef_u, coef_v, masks, n_levels)` — average pooling on staggered grids
   - `_compute_dx_hierarchy(dx, dy, n_levels)` — dx doubles each level

7. **Tests for Phase B:**
   - `test_restrict_prolong_roundtrip` — prolong(restrict(f)) ≈ f (for smooth f)
   - `test_restrict_conserves_integral` — ∑ restrict(f) × area ≈ ∑ f × area
   - `test_mask_hierarchy_all_ocean` — rectangular domain stays all-ocean
   - `test_mask_hierarchy_circular` — circular mask coarsens correctly

#### Phase C: V-Cycle and Convergence (2-3 days)

8. **`v_cycle`** — the core recursive algorithm:
   ```
   v_cycle(f, rhs, level):
       f = jacobi_smooth(f, rhs, level, n_pre)      # pre-smooth
       r = helmholtz_residual(f, rhs, level)         # compute residual
       r_coarse = restrict(r)                        # restrict to coarse
       if level == n_levels - 1:
           e_coarse = bottom_solve(r_coarse)         # direct/Jacobi
       else:
           e_coarse = v_cycle(zeros, r_coarse, level+1)  # recurse
       e_fine = prolong(e_coarse)                    # prolong correction
       f = f + e_fine                                # correct
       f = jacobi_smooth(f, rhs, level, n_post)      # post-smooth
       return f
   ```

9. **Bottom solver:**
   - `"direct"`: Assemble coarsest-level operator as dense matrix, precompute `pinv`
   - `"jacobi"`: Run 50+ Jacobi iterations at coarsest level

10. **Outer iteration:** Repeat V-cycles until `‖r‖ / ‖rhs‖ < tol` or max_cycles reached

11. **Tests for Phase C:**
    - `test_vcycle_convergence_rectangular` — residual < tol on rectangle
    - `test_vcycle_convergence_circular_mask` — residual < tol on circular domain
    - `test_vcycle_variable_coef` — spatially varying c(x,y)
    - `test_roundtrip_known_solution` — L(f_exact) → rhs, solve → f ≈ f_exact

#### Phase D: FMG and Polish (1-2 days)

12. **`fmg`** — Full Multigrid for better initial guess:
    ```
    fmg(rhs):
        rhs_hierarchy = [rhs, restrict(rhs), ..., restrict^(L-1)(rhs)]
        f = bottom_solve(rhs_hierarchy[-1])
        for level from (n_levels-2) down to 0:
            f = prolong(f)
            f = v_cycle(f, rhs_hierarchy[level], level)
        return f
    ```

13. **Factory function** `build_multigrid_helmholtz` — offline precomputation

14. **Public API exports** in `__init__.py`:
    - `MultigridHelmholtz2D`
    - `build_multigrid_helmholtz`

15. **Tests for Phase D:**
    - `test_fmg_faster_than_vcycle` — FMG converges in fewer total cycles
    - `test_fmg_matches_spectral_on_rectangle` — agrees with DST solver
    - `test_build_factory_api` — factory returns working solver

### 2.3 JAX Compatibility Considerations

**Challenge:** The V-cycle is recursive with depth `n_levels`. JAX's `lax.while_loop` / `lax.fori_loop` require fixed-shape arrays, but the grid size halves at each level.

**Approach options (in order of preference):**

1. **Unrolled recursion with padding** — Pad all levels to fine-grid size, use masks to select active regions. Wasteful in memory but simple and fully JIT-compatible. This is what the louity PyTorch reference effectively does (all arrays are allocated at fine-grid size).

2. **`lax.switch` dispatch** — Use `lax.switch(level, [fn_level_0, fn_level_1, ...])` where each function operates on the correct grid size. Requires generating per-level closures during `build_multigrid_helmholtz`.

3. **Static unroll** — Since `n_levels` is static, unroll the recursion at trace time (Python-level loop). Each level traces with its own array shapes. Simple, but recompiles if `n_levels` changes.

**Recommendation:** Start with option 3 (static unroll) — it's simplest and `n_levels` is typically 4-8 and fixed for a given grid. The `eqx.field(static=True)` annotation on `n_levels` ensures recompilation when it changes.

### 2.4 Integration with Existing Solvers

The multigrid solver should integrate cleanly with the existing solver stack:

- **As a CG preconditioner:** A single V-cycle makes an excellent preconditioner for CG. Add a convenience function:
  ```python
  def make_multigrid_preconditioner(mg_solver: MultigridHelmholtz2D) -> Callable:
      """Return a preconditioner that applies one V-cycle."""
      def precond(r):
          return mg_solver.v_cycle(jnp.zeros_like(r), r)
      return precond
  ```
  This can be passed to `solve_cg(matvec, rhs, preconditioner=make_multigrid_preconditioner(mg))`.

- **As a standalone solver:** `mg_solver(rhs)` for direct multigrid solve.

- **For #87 wrappers:** The convenience wrappers can accept a `solver` parameter that dispatches to spectral, capacitance, CG, or multigrid.

### 2.5 Test Plan for Multigrid

New file: `tests/test_multigrid.py`

```
class TestHelmholtzResidual:
    test_zero_residual_for_exact_solution
    test_constant_coef_matches_masked_laplacian
    test_variable_coef_stencil
    test_mask_boundary_zero_flux

class TestJacobiSmoothing:
    test_reduces_high_frequency_error
    test_convergence_rate
    test_respects_mask

class TestGridTransfer:
    test_restrict_constant_field
    test_restrict_conserves_mass
    test_prolong_constant_field
    test_restrict_prolong_smooth
    test_masked_restriction
    test_masked_prolongation
    test_odd_grid_sizes

class TestHierarchyConstruction:
    test_mask_coarsening_rectangular
    test_mask_coarsening_circular
    test_coefficient_coarsening
    test_auto_n_levels

class TestVCycle:
    test_convergence_poisson_rectangle
    test_convergence_helmholtz_rectangle
    test_convergence_poisson_circular_mask
    test_convergence_variable_coef
    test_residual_reduction_per_cycle

class TestFMG:
    test_fmg_convergence
    test_fmg_matches_spectral_rectangle
    test_fmg_on_irregular_domain

class TestMultigridFactory:
    test_build_rectangular
    test_build_with_mask
    test_build_variable_coef
    test_default_coef_is_ones

class TestJAXCompatibility:
    test_jit_vcycle
    test_jit_full_solve
    test_vmap_over_batch    # e.g., multi-layer QG with different lambda per layer
    test_grad_through_solve # differentiability (if needed)

class TestMultigridAsPreconditioner:
    test_mg_preconditioned_cg_converges_faster
```

---

## Recommended Implementation Order

### Step 1: Issue #87 — Convenience Wrappers (do first)
- Small, self-contained, high user-facing value
- Tests exercise existing solver infrastructure
- No new algorithmic complexity

### Step 2: Issue #71, Phase A — Core Stencil Operations
- `helmholtz_residual` and `jacobi_smooth`
- Foundation for everything else; test independently

### Step 3: Issue #71, Phase B — Grid Transfer + Hierarchy
- `restrict_2d`, `prolong_2d`, hierarchy builders
- Can be tested independently of the V-cycle

### Step 4: Issue #71, Phase C — V-Cycle + Outer Loop
- Combine phases A+B into the V-cycle
- This is where convergence tests become meaningful

### Step 5: Issue #71, Phase D — FMG + Factory + Polish
- FMG provides excellent initial guesses
- Factory function for user-friendly API
- Integration as CG preconditioner
- Final test suite, edge cases, documentation

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `finitevolx/_src/solvers/elliptic.py` | **Modify** | Add `streamfunction_from_vorticity`, `pressure_from_divergence`, `pv_inversion` |
| `finitevolx/_src/solvers/multigrid.py` | **Create** | New multigrid module |
| `finitevolx/_src/solvers/__init__.py` | **Modify** | Re-export new symbols (if used internally) |
| `finitevolx/__init__.py` | **Modify** | Export new public API |
| `tests/test_solver_wrappers.py` | **Create** | Tests for #87 wrappers |
| `tests/test_multigrid.py` | **Create** | Tests for #71 multigrid |

---

## Open Questions / Decisions Needed

1. **Multigrid JIT strategy:** Static unroll (recommended) vs. padded arrays vs. `lax.switch`? Static unroll is simplest but recompiles per `n_levels`.

2. **Bottom solver threshold:** At what coarsest grid size should we switch from `pinv` to iterated Jacobi? The louity reference uses `pinv` when the coarsest grid is small (< 8×8). This seems reasonable.

3. **Red-black vs. weighted Jacobi:** The reference uses weighted Jacobi (ω=0.8-0.95). Red-black Gauss-Seidel converges faster (ω=1.0 safe) but is harder to implement in JAX without scatter. Start with Jacobi; consider red-black as optimization.

4. **Differentiability:** Should `MultigridHelmholtz2D.__call__` support `jax.grad`? If so, need implicit differentiation (adjoint solve). This is a nice-to-have but not in the acceptance criteria.

5. **Scope of #87 wrappers:** Should the wrappers also accept a `MultigridHelmholtz2D` solver parameter? Yes — but only implement this after #71 is done. Start with spectral/capacitance/CG dispatch only.

6. **SOR solver:** Mentioned in #87 but the owner notes it's "rarely needed with spectral solvers; low priority." **Skip unless explicitly requested.**
