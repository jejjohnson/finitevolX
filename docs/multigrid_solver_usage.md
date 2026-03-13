# Multigrid Solver: Usage Guide

This page covers practical usage of finitevolX's multigrid Helmholtz solver,
from quick-start examples to variable-coefficient problems and
differentiable solves.

---

## Quick Start

```python
import jax
import jax.numpy as jnp
import numpy as np
import finitevolx as fvx

jax.config.update("jax_enable_x64", True)

# 1. Define a grid
Ny, Nx = 64, 64
dx, dy = 1.0 / Nx, 1.0 / Ny
mask = np.ones((Ny, Nx))  # rectangular domain

# 2. Build the solver (offline, once)
solver = fvx.build_multigrid_solver(mask, dx, dy, lambda_=10.0)

# 3. Solve (online, JIT-compilable)
rhs = jnp.sin(jnp.pi * jnp.arange(Ny)[:, None] / Ny) * \
      jnp.sin(jnp.pi * jnp.arange(Nx)[None, :] / Nx)
u = solver(rhs)
```

---

## Building the Solver

`build_multigrid_solver` precomputes the entire level hierarchy offline
(mask coarsening, face coefficients, operator diagonals).  This is done
once; the returned solver is then cheap to call repeatedly.

### Rectangular Domain

```python
mask = np.ones((64, 64))
solver = fvx.build_multigrid_solver(mask, dx, dy, lambda_=10.0)
```

### Masked (Irregular) Domain

```python
# Circular basin
Y, X = np.mgrid[:64, :64]
mask = ((X - 32)**2 + (Y - 32)**2 < 28**2).astype(float)

solver = fvx.build_multigrid_solver(mask, dx, dy, lambda_=10.0)
u = solver(rhs * jnp.array(mask))  # zero RHS outside domain
```

### With ArakawaCGridMask

```python
cgrid_mask = fvx.ArakawaCGridMask.from_mask(ocean_mask)
solver = fvx.build_multigrid_solver(cgrid_mask, dx, dy, lambda_=10.0)
```

### Variable Coefficient

```python
# Spatially varying diffusivity
coeff = 1.0 + 0.5 * np.sin(2 * np.pi * X / 64)

solver = fvx.build_multigrid_solver(
    mask, dx, dy,
    lambda_=10.0,
    coeff=coeff,
)
u = solver(rhs)
```

### Controlling the Hierarchy

```python
solver = fvx.build_multigrid_solver(
    mask, dx, dy,
    lambda_=10.0,
    n_levels=3,     # number of grid levels (default: auto)
    n_pre=6,        # pre-smoothing iterations
    n_post=6,       # post-smoothing iterations
    n_coarse=50,    # bottom-solver iterations
    omega=0.95,     # Jacobi relaxation weight
    n_cycles=5,     # V-cycles per solve
)
```

!!! tip "Auto level detection"
    When `n_levels=None` (default), the factory halves both dimensions
    until either would drop below 8.  For a 64x64 grid this gives 4
    levels (64 -> 32 -> 16 -> 8).

---

## Solve Modes

The solver provides three methods with different autodiff characteristics.

### Implicit Differentiation (Default)

```python
u = solver(rhs)  # uses jax.lax.custom_linear_solve
```

The backward pass solves the adjoint equation with multigrid — same cost
as the forward pass, O(1) memory.  **Recommended for most use cases.**

### One-Step Differentiation

```python
u = solver.solve_onestep(rhs)
```

Differentiates through only the last V-cycle.  Cheapest backward pass,
with approximate gradients (error proportional to the convergence rate).

### Unrolled Differentiation

```python
u = solver.solve_unrolled(rhs)
```

Differentiates through all V-cycle iterations.  O(n_cycles) memory cost.

---

## Multigrid as a CG Preconditioner

A single multigrid V-cycle makes an excellent preconditioner for
`solve_cg`, especially for variable-coefficient problems.  See the
[Preconditioner Comparison](elliptic_solvers.md#preconditioners) in the
elliptic solvers docs for a full ranking of all available preconditioners.

```python
# Build solver and preconditioner
mg_solver = fvx.build_multigrid_solver(
    mask, dx, dy, lambda_=10.0, coeff=coeff
)
mg_precond = fvx.make_multigrid_preconditioner(mg_solver)

# Define the operator
mask_jnp = jnp.array(mask)
def A(x):
    """Variable-coefficient Helmholtz operator."""
    # For constant coeff, use masked_laplacian directly.
    # For variable coeff, use the multigrid's internal operator.
    from finitevolx._src.solvers.multigrid import _apply_operator
    return _apply_operator(x, mg_solver.levels[0])

# Solve with multigrid-preconditioned CG
u, info = fvx.solve_cg(
    A, rhs * mask_jnp,
    preconditioner=mg_precond,
    rtol=1e-8,
    atol=1e-8,
)
u = u * mask_jnp
print(f"Converged in {info.iterations} iterations")
```

---

## Differentiable Solves

### Gradient Through the Solve

```python
def loss(rhs):
    u = solver(rhs)  # implicit diff (default)
    return jnp.sum(u ** 2)

grad_rhs = jax.grad(loss)(rhs)
```

### Learning a Spatially Varying Coefficient

```python
import optax

# Parameterise the coefficient field
log_coeff = jnp.zeros((Ny, Nx))  # learnable, initialised to c=1

def forward(log_coeff, rhs):
    coeff = jnp.exp(log_coeff)
    solver = fvx.build_multigrid_solver(
        mask, dx, dy, lambda_=10.0, coeff=np.asarray(coeff)
    )
    return solver(rhs)

def loss_fn(log_coeff):
    u_pred = forward(log_coeff, rhs)
    return jnp.mean((u_pred - u_target) ** 2)

# Note: build_multigrid_solver is offline (numpy), so for gradient-based
# learning you would typically fix the solver and differentiate only
# through the RHS, or rebuild the solver at each outer iteration.
```

### Comparing Gradient Strategies

```python
def loss_implicit(rhs):
    return jnp.sum(solver(rhs) ** 2)

def loss_onestep(rhs):
    return jnp.sum(solver.solve_onestep(rhs) ** 2)

def loss_unrolled(rhs):
    return jnp.sum(solver.solve_unrolled(rhs) ** 2)

g_implicit = jax.grad(loss_implicit)(rhs)
g_onestep = jax.grad(loss_onestep)(rhs)
g_unrolled = jax.grad(loss_unrolled)(rhs)

# g_implicit and g_unrolled should agree closely
# g_onestep may differ by O(rho) ~ 0.1-0.3
```

---

## JIT and vmap Compatibility

### JIT Compilation

```python
import equinox as eqx

@eqx.filter_jit
def solve(solver, rhs):
    return solver(rhs)

u = solve(solver, rhs)  # compiled on first call, fast thereafter
```

### Batched Solves with vmap

```python
# Solve for multiple RHS fields at once
rhs_batch = jnp.stack([rhs1, rhs2, rhs3])  # (3, Ny, Nx)

@eqx.filter_jit
def batch_solve(solver, rhs_batch):
    return jax.vmap(solver)(rhs_batch)

u_batch = batch_solve(solver, rhs_batch)  # (3, Ny, Nx)
```

---

## Tuning Guide

| Parameter | Default | Effect |
|---|---|---|
| `n_cycles` | 5 | More cycles = lower residual, but slower. 3-5 is typical. |
| `n_pre` / `n_post` | 6 | More smoothing = better convergence rate per cycle, at higher cost. |
| `n_coarse` | 50 | Enough to solve the (small) coarsest grid accurately. |
| `omega` | 0.95 | Jacobi weight. Lower (0.6-0.8) for stability, higher for speed. |
| `n_levels` | auto | More levels = cheaper coarse grids, but grid must be divisible by $2^{L-1}$. |

!!! tip "Quick convergence check"
    ```python
    u = solver(rhs)
    from finitevolx._src.solvers.multigrid import _apply_operator
    residual = jnp.linalg.norm(rhs - _apply_operator(u, solver.levels[0]))
    print(f"Residual norm: {residual:.2e}")
    ```
