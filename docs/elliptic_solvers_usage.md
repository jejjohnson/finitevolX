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
grid = fvx.ArakawaCGrid2D.from_interior(64, 64, 1e5, 1e5)
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
    # Dirichlet BCs (ψ = 0 on boundary) — most common
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
    solver = fvx.build_capacitance_solver(
        ocean_mask, dx, dy, lambda_=-1.0, base_bc="fft"
    )

    # Reuse for many solves
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, method="capacitance", capacitance_solver=solver
    )
    ```

### Pressure from Divergence

Solves $\nabla^2 p = \nabla \cdot \mathbf{u}$ (always Poisson, $\lambda = 0$).

```python
# Neumann BCs (∂p/∂n = 0) — standard for pressure with solid walls
p = fvx.pressure_from_divergence(div_u, dx, dy, bc="dct")
```

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

---

## Direct Spectral Solvers

For maximum control, use the spectral solvers directly.

### Poisson Solvers

```python
# Dirichlet (DST-I): ψ = 0 on all edges
psi = fvx.solve_poisson_dst(rhs, dx, dy)

# Neumann (DCT-II): ∂ψ/∂n = 0 on all edges (zero-mean gauge)
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
This is expensive ($N_b$ spectral solves) but only done once.

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
    base_bc="fft",    # rectangular base solver
)
```

### Using with ArakawaCGridMask

When you have a `CGridMask`, pass it directly — the solver extracts the
`psi` staggering mask and precomputed boundary indices automatically:

```python
cgrid_mask = fvx.ArakawaCGridMask.from_mask(ocean_mask)
solver = fvx.build_capacitance_solver(
    cgrid_mask, dx, dy, lambda_=-1.0, base_bc="fft"
)
```

### Online Solve

```python
psi = solver(rhs)  # JIT-compilable, vmap-compatible
```

The solver guarantees $\psi = 0$ at all inner-boundary points (ocean cells
adjacent to land).

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
        A, shape=(Ny, Nx), rank=50, key=jax.random.PRNGKey(0)
    )
    ```

    Builds a low-rank approximate inverse by probing the operator with
    random vectors.  Useful when you only have `matvec` access to the
    operator, or when the spectral preconditioner is not effective enough.

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
