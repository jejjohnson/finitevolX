# Elliptic Solvers: Usage Guide

This page covers practical usage of finitevolX's elliptic solver module,
from quick-start examples to multi-layer PV inversion on masked domains.

---

## Quick Start

The fastest way to solve an elliptic problem: use a convenience wrapper.

```python
import jax
import jax.numpy as jnp
import finitevolx as fvx

jax.config.update("jax_enable_x64", True)

# 1. Set up a grid
grid = fvx.CartesianGrid2D.from_interior(64, 64, 1e5, 1e5)
dx, dy = grid.dx[0], grid.dy[0]

# 2. Create a vorticity field
j = jnp.arange(64)[:, None]
i = jnp.arange(64)[None, :]
zeta = jnp.sin(jnp.pi * (j + 1) / 65) * jnp.sin(jnp.pi * (i + 1) / 65)

# 3. Invert for streamfunction (Dirichlet BCs)
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")
```

---

## Convenience Wrappers

finitevolX provides three high-level wrappers that handle solver dispatch
internally.  All three accept the same `method`, `mask`, `capacitance_solver`,
and `preconditioner` keyword arguments.

### Streamfunction from Vorticity

Solves $\nabla^2 \psi - \lambda\psi = \zeta$.

=== "Spectral (rectangular domain)"

    ```python
    # Dirichlet BCs (ψ = 0 on the C-grid wall nodes, X index 0 and N-2) — most common
    psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")

    # With Helmholtz parameter (QG inversion)
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, bc="dst", lambda_=-1.0
    )
    ```

=== "CG (masked domain)"

    ```python
    mask = jnp.array(ocean_mask, dtype=float)  # 1=ocean, 0=land
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, method="cg", mask=mask, lambda_=-1.0
    )
    ```

=== "Capacitance (masked domain, fast)"

    ```python
    # One-time precomputation
    solver = fvx.build_capacitance_solver(ocean_mask, dx, dy, lambda_=-1.0)

    # Reuse for many solves
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, method="capacitance", capacitance_solver=solver
    )
    ```

### Pressure from Divergence

Solves $\nabla^2 p = \nabla \cdot \mathbf{u}$ (always Poisson, $\lambda = 0$).

```python
# Neumann BCs (∂p/∂n = 0 on the cell faces) — standard for pressure with solid walls
p = fvx.pressure_from_divergence(div_u, dx, dy, bc="dct")

# The same Neumann wall on a masked basin: zero flux through every coast face
p = fvx.pressure_from_divergence(div_u, dx, dy, method="cg", mask=mask)
```

`bc` is honoured by every method or rejected, never ignored.  With
`method="cg"`, `bc="dct"` (the default) builds the zero-flux operator at T
points and returns the zero-mean solution; `bc="dst"` holds p = 0 on the dry
cells next to the wet ones.  The default preconditioner is the spectral solve
with the same `bc`.  `method="capacitance"` is Dirichlet on the coast only,
so it needs `bc="dst"`; `bc="fft"` is for the rectangular spectral solve
only.

The wrappers know the staggering of what they solve for (ψ at X points,
p at T points), pass only its unknowns to the transform and return them
embedded in zeros, so the wall is the C-grid wall for every `bc` and
`method`; see the [wall table](elliptic_solvers.md#boundary-conditions).
Pass the full `[Ny, Nx]` array: do not slice the interior by hand.

### PV Inversion (Multi-Layer)

Solves $(\nabla^2 - \lambda_k)\,\psi_k = q_k$ for each vertical mode.

=== "Single layer"

    ```python
    psi = fvx.pv_inversion(pv, dx, dy, lambda_=-0.5, bc="dst")
    ```

=== "Multi-layer (per-mode λ)"

    ```python
    # lambda_ array: one value per vertical mode
    lambdas = jnp.array([-0.1, -0.5, -2.0])  # e.g., 1/Rd² per mode

    # pv shape: (nl, Ny, Nx) or (batch, nl, Ny, Nx)
    psi = fvx.pv_inversion(pv, dx, dy, lambda_=lambdas, bc="dst")
    ```

=== "With vertical mode decomposition"

    ```python
    # Decompose layer PV into vertical modes
    H_layers = jnp.array([500.0, 1500.0, 3000.0])
    eigenvalues, modes = fvx.decompose_vertical_modes(H_layers)

    # Transform to mode space
    pv_modes = fvx.layer_to_mode(pv_layers, modes)

    # Invert each mode with its eigenvalue
    psi_modes = fvx.pv_inversion(
        pv_modes, dx, dy, lambda_=eigenvalues, bc="dst"
    )

    # Transform back to layer space
    psi_layers = fvx.mode_to_layer(psi_modes, modes)
    ```


### Known Boundary Values (Inhomogeneous Dirichlet)

!!! tip "Tutorial"
    For the full derivation — notation, the block linear system, pseudocode,
    a from-scratch NumPy implementation, convergence and gradients — see the
    [Known Boundary Values tutorial](notebooks/inhomogeneous_bcs_tutorial.py).

By default every solver assumes $\psi = 0$ on the boundary.  When the
boundary values are known but non-zero — SSH from a parent model, a tide
gauge, reanalysis at an open boundary — pass them as `known_values`.  All
three wrappers accept it:

```python
psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy, method="cg", mask=mask,
    known_values=g,          # (Ny, Nx); only the known cells are read
)
```

**Which cells are known.**  The *coast ring* — the dry cells adjacent to at
least one wet cell, around the outer walls *and* any islands — plus any
extra wet cells you mark with `known_mask`.  The coast is exactly where
`method="cg"` puts its ψ = 0 wall: for ψ with a `Mask2D` it is the C-grid
wall nodes (X rows / columns `0` and `N-2` of a rectangle).  Every wet cell
is solved for.  The returned solution equals `known_values` exactly on the
known cells (so ψ carries the prescribed values on the dry wall nodes);
values of `known_values` elsewhere are ignored.  Inspect the ring with
`fvx.coast_ring(mask)`.  For the T-point `pressure_from_divergence` the
coast is the ghost ring, so the values sit at the ghost-cell centres, half a
cell outside the wall faces; `method="spectral"` refuses known values there.

Under the hood this is the **lifting trick**: put the known values in a
field $\psi_{\text{lift}}$ (zero elsewhere), solve the homogeneous problem

$$
(\nabla^2 - \lambda)\,\psi_{\text{hom}}
  = f - (\nabla^2 - \lambda)\,\psi_{\text{lift}}
$$

on the remaining cells, and return
$\psi = \psi_{\text{lift}} + \psi_{\text{hom}}$.

=== "CG (any mask)"

    ```python
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, lambda_=4.0, method="cg", mask=mask, known_values=g
    )
    ```

=== "Capacitance"

    ```python
    # Build on the SAME wet mask: its unknowns are every wet cell with
    # psi = 0 on the coast, exactly the cells left to solve.
    solver = fvx.build_capacitance_solver(
        np.asarray(mask > 0.5), dx, dy, lambda_=4.0
    )
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, lambda_=4.0, method="capacitance", mask=mask,
        capacitance_solver=solver, known_values=g,
    )
    ```

=== "Spectral (rectangular basin)"

    ```python
    # No mask: solves the psi basin of the rectangle -- wet X nodes
    # [1:-2, 1:-2] -- with the known values on its wall nodes (rows/columns
    # 0 and N-2).  Requires bc="dst".
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, bc="dst", lambda_=4.0, known_values=g
    )
    ```

The spectral path solves exactly the problem `method="cg"` solves with the
rectangle's `Mask2D` (its `xy_corner_strict` is wet on `[1:-2, 1:-2]`), and
with `known_values = 0` it is the ordinary `bc="dst"` solve.

#### Sparse observations: `known_mask`

Interior observations are pinned the same way as the boundary ring
(`method="cg"` only):

```python
obs = jnp.zeros((Ny, Nx), dtype=bool).at[20, 30].set(True).at[25, 45].set(True)
values = jnp.zeros((Ny, Nx)).at[20, 30].set(0.12).at[25, 45].set(-0.05)

psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy, method="cg", mask=mask,
    known_values=values, known_mask=obs,
)
# psi[20, 30] == 0.12 and psi[25, 45] == -0.05 exactly
```

#### Multi-layer PV inversion

`known_values` broadcasts against `pv`: a `(Ny, Nx)` field is shared by all
layers, an `(nl, Ny, Nx)` array gives each layer its own values.  Each
layer's correction uses its own $\lambda_k$.

```python
psi = fvx.pv_inversion(
    pv, dx, dy, lambda_=lambdas, method="cg", mask=mask,
    known_values=psi_boundary,   # (Ny, Nx) or (nl, Ny, Nx)
)
```

#### Boundary conditions as a `BoundaryConditionSet`

`bc` also accepts a `BoundaryConditionSet`.  Its face types choose the
transform (all `Dirichlet1D` → `"dst"`, all zero `Neumann1D` → `"dct"`,
all `Periodic1D` → `"fft"`), its `mask` feeds the mask-based methods, and
non-zero `Dirichlet1D` values become known values on each face's wall
nodes (the coast cells on the row / column just outside the wet cells: X
rows `0` / `N-2` and columns `0` / `N-2` of a rectangle):

```python
bc = fvx.BoundaryConditionSet(
    mask=mask,                                   # a Mask2D, or omit for method="spectral"
    south=fvx.Dirichlet1D("south", value=0.0),
    north=fvx.Dirichlet1D("north", value=0.1),   # prescribed SSH on the north wall
    west=fvx.Dirichlet1D("west", value=0.0),
    east=fvx.Dirichlet1D("east", value=0.0),
)
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc=bc, method="cg")
```

West/east win on a node shared by two faces, as when the set fills ghost
cells.  An explicit `known_values` overrides the face values, and an
all-zero set (`BoundaryConditionSet.closed()`) is the ordinary homogeneous
solve.  `method="spectral"` raises on a `bc.mask` (it solves the
rectangle), and `pressure_from_divergence` raises on non-zero
`Dirichlet1D` faces: a T-point Dirichlet value belongs on the face, which
the cell-centred lifting cannot reach.

#### Using the lifting directly

For your own solver or a solve loop that reuses the setup, use
`SolveDomain` and `KnownValueLifting`:

```python
# Setup (once): derived masks, then a solver on the effective domain
domain = fvx.SolveDomain(mask, known_mask=obs)
lifter = fvx.KnownValueLifting(domain, dx, dy, lambda_=4.0)
eff = domain.effective_mask.astype(float)
A = lambda x: fvx.masked_laplacian(x, eff, dx, dy, lambda_=4.0)

# Per step (JIT-friendly): correct, solve, reconstruct
rhs_corrected, value_lift = lifter.preprocess(rhs, known_values)
psi_hom, _ = fvx.solve_cg(A, rhs_corrected)
psi = lifter.postprocess(psi_hom * eff, value_lift)
```

The solver's unknowns must be exactly `domain.effective_mask` (every wet
cell not in `known_mask`), with zero on the coast: build a CG operator on
`domain.effective_mask`.

For **multigrid**, use it as the CG preconditioner — the wrapper then does
the lifting for you:

```python
mg = fvx.build_multigrid_solver(
    np.asarray(domain.effective_mask, dtype=float), dx, dy, lambda_=4.0
)
psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy, lambda_=4.0, method="cg", mask=mask, known_values=g,
    preconditioner=fvx.make_multigrid_preconditioner(mg),
)
```

A standalone multigrid solve is not a drop-in here: its operator places the
Dirichlet condition on cell faces rather than using `masked_laplacian`'s
stencil, so it does not reproduce the lifted problem.

---

## Direct Spectral Solvers

For maximum control, use the spectral solvers directly.

### Poisson Solvers

```python
# The direct solvers take only the unknowns (see the wall table in the theory page).
# Dirichlet (DST-I): ψ = 0 one node outside the array on every side
psi = fvx.solve_poisson_dst(rhs, dx, dy)

# Neumann (DCT-II): ∂ψ/∂n = 0 half a cell outside the array (zero-mean gauge)
psi = fvx.solve_poisson_dct(rhs, dx, dy)

# Periodic (FFT): doubly-periodic domain (zero-mean gauge)
psi = fvx.solve_poisson_fft(rhs, dx, dy)
```

### Helmholtz Solvers

```python
# (∇² − λ)ψ = f
psi = fvx.solve_helmholtz_dst(rhs, dx, dy, lambda_=-1.0)
psi = fvx.solve_helmholtz_dct(rhs, dx, dy, lambda_=-1.0)
psi = fvx.solve_helmholtz_fft(rhs, dx, dy, lambda_=-1.0)
```

All Helmholtz solvers handle `lambda_=0` internally (via tracer-safe
null-mode guards), so they work correctly inside `jax.vmap`:

```python
# Batched solve with per-layer lambda
lambdas = jnp.array([-0.5, -1.0, -2.0])
rhs_batch = jnp.stack([rhs1, rhs2, rhs3])

psi_batch = jax.vmap(
    lambda r, l: fvx.solve_helmholtz_dst(r, dx, dy, l)
)(rhs_batch, lambdas)
```

---

## Capacitance Matrix Solver

The capacitance method extends spectral solvers to irregular domains
defined by a mask.

### Building the Solver

The offline step precomputes Green's functions and the capacitance matrix.
This is expensive ($N_c$ spectral solves, $N_c$ = number of coast cells) but
only done once.  The solver's unknowns are **every wet cell**, with
$\psi = 0$ on the coast (the dry cells next to water): the same problem
`method="cg"` solves on that mask, so the two agree to CG's tolerance.

```python
import numpy as np

# Binary mask: True = ocean, False = land
ocean_mask = np.ones((64, 64), dtype=bool)
ocean_mask[:5, :] = ocean_mask[-5:, :] = False  # land border
ocean_mask[:, :5] = ocean_mask[:, -5:] = False

# Build solver (offline)
solver = fvx.build_capacitance_solver(
    ocean_mask, dx, dy,
    lambda_=-1.0,     # Helmholtz parameter
    base_bc="dst",    # rectangular base solver (the default)
)
```

`base_bc` picks the rectangular base (`"dst"`, the default, `"fft"` or
`"dct"`).  It only matters beyond the array edge, so every base gives the
same solution when the mask's wet cells stay off the array edge (a dry
ghost ring); a singular base (`"fft"` / `"dct"` at $\lambda = 0$) is
handled exactly, but `"dst"` is non-singular for every $\lambda \ge 0$.

### Using with Mask2D

When you have a `Mask2D`, pass it directly — the solver uses its ψ
(X-point) mask `xy_corner_strict`, so ψ = 0 falls on the C-grid wall nodes:

```python
cgrid_mask = fvx.Mask2D.from_mask(ocean_mask)
solver = fvx.build_capacitance_solver(cgrid_mask, dx, dy, lambda_=-1.0)
```

### Online Solve

```python
psi = solver(rhs)  # JIT-compilable, vmap-compatible
```

The solver returns $\psi = 0$ on every dry cell (the coast included) and
solves the 5-point equation on every wet cell.

---

## Conjugate Gradient Solver

For domains where the capacitance matrix is too large, use the
preconditioned CG solver.

### Basic Usage

```python
mask = jnp.array(ocean_mask, dtype=float)

# Define the operator
def A(x):
    return fvx.masked_laplacian(x, mask, dx, dy, lambda_=-1.0)

# Solve with spectral preconditioner (default)
psi, info = fvx.solve_cg(
    A, rhs * mask,
    preconditioner=fvx.make_spectral_preconditioner(dx, dy, lambda_=-1.0),
    rtol=1e-8,
    atol=1e-8,
)
psi = psi * mask  # zero out land points
```

### Convergence Info

`solve_cg` returns a `CGInfo` named tuple:

```python
psi, info = fvx.solve_cg(A, rhs)
print(f"Converged: {info.converged}")
print(f"Iterations: {info.iterations}")
print(f"Residual norm: {info.residual_norm:.2e}")
```

`solve_cg` does not raise when CG fails (pass `throw=True` for that):
hitting `max_steps`, an indefinite operator (`lambda_ < 0`) or a
preconditioner of the wrong sign all return `converged=False`.
`converged` is checked on the explicit residual of the un-preconditioned
system, `max|rhs - A(x)| <= atol + rtol max|rhs|`, and `residual_norm` is
the L2 norm of that residual. The solve is differentiable with respect to
`lambda_` and `dx`, also with a preconditioner built from them. Its output
can be non-zero on land (the spectral preconditioners ignore the mask), so
multiply by the mask as above; the wrappers do.

### Preconditioners

=== "Spectral (cheapest)"

    ```python
    M_inv = fvx.make_spectral_preconditioner(
        dx, dy, lambda_=-1.0, bc="fft"
    )
    ```

    Uses the rectangular spectral solver as an approximate inverse.
    Nearly free (one FFT pair) and very effective when the domain is
    close to rectangular with constant coefficients.

=== "Nyström (operator-only)"

    ```python
    M_inv = fvx.make_nystrom_preconditioner(
        A, shape=(Ny, Nx), rank=100, key=jax.random.PRNGKey(0)
    )
    psi, info = fvx.solve_cg(A, rhs, preconditioner=M_inv, rtol=1e-8)
    ```

    Builds a low-rank approximate inverse by probing the operator with
    random vectors.  Only needs `matvec` access — useful when you have a
    black-box operator with no analytic structure to exploit.

    !!! warning "Nyström is a niche preconditioner"
        For standard Helmholtz/Poisson problems on known grids, the spectral
        or multigrid preconditioners are significantly more effective.
        Nyström captures only `rank` directions of the inverse; the remaining
        directions receive a scalar fallback, so the iteration count may not
        improve much over unpreconditioned CG.  Consider Nyström only when
        no other preconditioner is available (e.g., a non-standard operator
        known only through `matvec`).

=== "Multigrid (most powerful)"

    ```python
    mg = fvx.build_multigrid_solver(mask, dx, dy, lambda_=1.0, coeff=coeff)
    M_inv = fvx.make_multigrid_preconditioner(mg)
    ```

    A single multigrid V-cycle as an approximate inverse.  Captures both
    high- and low-frequency error across the grid hierarchy.  Handles
    variable coefficients and masked domains natively.  Typically reduces
    CG from hundreds of iterations to 5–10.

=== "Factory (dispatches by name)"

    ```python
    # Dispatches to spectral, nystrom, or multigrid
    M_inv = fvx.make_preconditioner("spectral", dx=dx, dy=dy, lambda_=1.0)
    M_inv = fvx.make_preconditioner("nystrom", matvec=A, shape=(64, 64))
    M_inv = fvx.make_preconditioner("multigrid", mg_solver=mg)
    ```

    Convenient when the preconditioner choice is a configurable parameter.

=== "Custom"

    ```python
    def my_preconditioner(r):
        """Any callable (Ny, Nx) -> (Ny, Nx) that approximates A^{-1}."""
        return some_approximate_inverse(r)

    psi, info = fvx.solve_cg(A, rhs, preconditioner=my_preconditioner)
    ```

!!! tip "Which preconditioner should I use?"
    See the [Preconditioner Decision Guide](elliptic_solvers.md#decision-guide)
    in the theory page for a full comparison.  **TL;DR**: start with spectral
    (free, works well for constant-coefficient near-rectangular problems);
    switch to multigrid for variable coefficients or complex masks.

---

## JIT and vmap Compatibility

All solvers are fully compatible with `jax.jit` and `jax.vmap`.

### JIT Compilation

```python
@jax.jit
def invert_pv(pv_field, lambdas):
    return fvx.pv_inversion(pv_field, dx, dy, lambda_=lambdas, bc="dst")

psi = invert_pv(pv, lambdas)
```

### Batched Solves with vmap

```python
# Solve the same equation for many RHS fields
@jax.jit
def batch_solve(rhs_batch):
    return jax.vmap(lambda r: fvx.solve_poisson_dst(r, dx, dy))(rhs_batch)

psi_batch = batch_solve(rhs_ensemble)  # (n_ensemble, Ny, Nx)
```

### Gradient Through Solves

Spectral solvers are differentiable — you can backpropagate through them:

```python
def loss(rhs):
    psi = fvx.solve_poisson_dst(rhs, dx, dy)
    return jnp.sum(psi ** 2)

grad_rhs = jax.grad(loss)(rhs)
```
