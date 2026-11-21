# Elliptic Solvers: Usage Guide

This page covers practical usage of finitevolX's elliptic solver module,
from quick-start examples to masked-domain solvers.
The snippets run in order: later ones reuse the names the Quick Start
defines.

---

## Quick Start

The fastest way to solve an elliptic problem on a rectangle: slice the unknowns of the field you solve for, call the spectral solver, and embed the result in zeros.

```python
import gaussx
import jax
import jax.numpy as jnp
import numpy as np
import gridax_jax as fvx

jax.config.update("jax_enable_x64", True)

# 1. A 64 x 64-cell basin, 100 km on a side.  Arrays are [Ny, Nx] = [66, 66]:
#    the interior plus one ghost ring.
grid = fvx.CartesianGrid2D.from_interior(64, 64, 1e5, 1e5)
dx, dy, Ny, Nx = grid.dx, grid.dy, grid.Ny, grid.Nx

# 2. Vorticity at the X points, where psi lives (X node i sits at x = i dx)
x = jnp.arange(Nx)[None, :] * dx
y = jnp.arange(Ny)[:, None] * dy
zeta = jnp.sin(jnp.pi * x / grid.Lx) * jnp.sin(jnp.pi * y / grid.Ly)  # [Ny, Nx]

# 3. Invert for the streamfunction, psi = 0 on the C-grid wall nodes (DST-I on the X basin)
psi = jnp.zeros_like(zeta).at[1:-2, 1:-2].set(
    fvx.solve_helmholtz_dst1(zeta[1:-2, 1:-2], dx, dy)
)  # [Ny, Nx]
```

The solver treats every node it is given as an unknown, so you slice the unknowns of your staggering and embed the result (see [Staggering and walls](#staggering-and-walls)).
The one-call inversions that do this for you (`streamfunction_from_vorticity`, `pressure_from_divergence` and `pv_inversion`) moved to sesmax-core; see below.

The masked examples below use a circular basin with an island, and a
physical Helmholtz parameter $\lambda = 1/R_d^2 > 0$:

```python
# Ocean at T points (True = wet); Mask2D derives the psi (X) and p (T) masks
jj, ii = np.mgrid[:Ny, :Nx]
ocean = np.zeros((Ny, Nx), dtype=bool)
ocean[1:-1, 1:-1] = True
ocean &= (ii - Nx / 2) ** 2 + (jj - Ny / 2) ** 2 < (0.45 * Nx) ** 2
ocean &= (ii - 0.62 * Nx) ** 2 + (jj - 0.55 * Ny) ** 2 > (0.1 * Nx) ** 2
mask = fvx.Mask2D.from_mask(ocean)  # psi uses mask.xy_corner_strict, p uses mask.h

Rd = 2e4             # deformation radius [m]
lambda_ = 1 / Rd**2  # (∇² − λ) is negative definite for λ ≥ 0
```

---

## One-Call Inversions (in sesmax-core)

The one-call wrappers `streamfunction_from_vorticity`, `pressure_from_divergence` and `pv_inversion`, their known-value lifting and `solve_ocean_pde` moved to sesmax-core (`sesmax_core.gfd` and `sesmax_core.solve_pde`).
gridax no longer has them.
In sesmax every argument after `dy` is keyword-only (after `lambda_` for `pv_inversion`), and `method="multigrid"` is new.
See the [elliptic inversions guide](https://jejjohnson.github.io/sesmax/elliptic-inversions/) and the [`sesmax_core.gfd` API](https://jejjohnson.github.io/sesmax/reference/gfd/).

(staggering-and-walls)=
### Staggering and walls

The wrappers know the staggering of what they solve for (ψ at X points, p at T points), pass only its unknowns to the transform and return them embedded in zeros, so the wall is the C-grid wall for every `bc` and `method`.
Calling a spectral solver directly, you make the same choice by slicing; see the [wall table](elliptic_solvers.md#boundary-conditions).

### Known Boundary Values (Inhomogeneous Dirichlet)

By default every solver assumes $\psi = 0$ on the boundary.
When the boundary values are known but non-zero (SSH from a parent model, a tide gauge, reanalysis at an open boundary), the `known_values=` argument of the sesmax inversions handles it; the [Known Boundary Values tutorial](https://jejjohnson.github.io/sesmax/inhomogeneous-bcs-tutorial/) in sesmax derives the method and shows every solver.

Under the hood this is the **lifting trick**: put the known values in a field $\psi_{\text{lift}}$ (zero elsewhere), solve the homogeneous problem

$$
(\nabla^2 - \lambda)\,\psi_{\text{hom}}
  = f - (\nabla^2 - \lambda)\,\psi_{\text{lift}}
$$

on the remaining cells, and return $\psi = \psi_{\text{lift}} + \psi_{\text{hom}}$.
The *coast ring* (`fvx.coast_ring(mask)`) holds the known cells: the dry cells adjacent to at least one wet cell, around the outer walls and any islands, plus any extra wet cells you mark with `known_mask`.

#### Using the lifting directly

`SolveDomain` and `KnownValueLifting` stay in finitevolX, for your own solver or a solve loop that reuses the setup:

```python
# Known values: psi rising from 0 (south) to 1e3 (north) on the coast ring,
# and two interior observations (only the known cells of `known_values` are read)
g = 1e3 * y / grid.Ly * jnp.ones((Ny, Nx))
obs = jnp.zeros((Ny, Nx), dtype=bool).at[20, 30].set(True).at[25, 45].set(True)
values = jnp.zeros((Ny, Nx)).at[20, 30].set(120.0).at[25, 45].set(-50.0)

# Setup (once): derived masks, then a solver on the effective domain
domain = fvx.SolveDomain(mask, known_mask=obs)
lifter = fvx.KnownValueLifting(domain, dx, dy, lambda_=lambda_)
eff = domain.effective_mask.astype(float)
A = lambda x: fvx.masked_laplacian(x, eff, dx, dy, lambda_=lambda_)
A_op = gaussx.as_linear_operator(
    A, in_structure=jax.ShapeDtypeStruct(eff.shape, eff.dtype), negative_definite=True
)
cg = gaussx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=2000)

# Per step (JIT-friendly): correct, solve, reconstruct
known_values = jnp.where(obs, values, g)
rhs_corrected, value_lift = lifter.preprocess(zeta, known_values)
psi_hom, info = gaussx.linear_solve_with_info(A_op, rhs_corrected, solver=cg)
psi = lifter.postprocess(psi_hom * eff, value_lift)
```

The solver's unknowns must be exactly `domain.effective_mask` (every wet cell not in `known_mask`), with zero on the coast: build a CG operator on `domain.effective_mask`.

For **multigrid**, build the hierarchy on the same effective mask and use it as the CG preconditioner on the lifted right-hand side:

```python
mg = fvx.build_multigrid_solver(
    np.asarray(domain.effective_mask, dtype=float), dx, dy, lambda_=lambda_
)
# gaussx solves (-A) x = -b, so its preconditioner is -M^-1 (one V-cycle from zero)
pre = lambda r: -mg.v_cycle(jnp.zeros_like(r), r)
psi_hom, info = gaussx.linear_solve_with_info(
    A_op, rhs_corrected, solver=cg, preconditioner=pre
)
```

The multigrid operator (default `bc="dirichlet"`) is `masked_laplacian`'s stencil with dry neighbours zero, so it is the same operator CG solves.

---

## Direct Spectral Solvers

For maximum control, use the spectral solvers directly.  They treat
**every** input node as an unknown (see the
[wall table](elliptic_solvers.md#boundary-conditions)), so here you do pass
the unknowns of your staggering yourself.

### Poisson Solvers

```python
f_in = zeta[1:-2, 1:-2]  # psi's unknowns: the wet X nodes of the rectangle

# Dirichlet (DST-I): ψ = 0 one node outside the input on every side
psi_in = fvx.solve_poisson_dst(f_in, dx, dy)

# Neumann (DCT-II): ∂p/∂n = 0 half a cell outside the input (zero-mean gauge)
p_in = fvx.solve_poisson_dct(zeta[1:-1, 1:-1], dx, dy)  # any T-point field

# Periodic (FFT): doubly-periodic domain (zero-mean gauge)
psi_per = fvx.solve_poisson_fft(f_in, dx, dy)

# Embed in zeros: the X-point streamfunction with its wall nodes at 0
psi = jnp.zeros_like(zeta).at[1:-2, 1:-2].set(psi_in)
```

### Helmholtz Solvers

```python
# (∇² − λ)ψ = f with λ = 1/Rd² > 0
psi_in = fvx.solve_helmholtz_dst(f_in, dx, dy, lambda_=lambda_)
psi_in = fvx.solve_helmholtz_dct(f_in, dx, dy, lambda_=lambda_)
psi_in = fvx.solve_helmholtz_fft(f_in, dx, dy, lambda_=lambda_)
```

All Helmholtz solvers handle `lambda_=0` internally (via tracer-safe
null-mode guards), so they work correctly inside `jax.vmap`:

```python
# Batched solve with per-layer lambda
lambdas = 1.0 / jnp.array([40e3, 20e3, 10e3]) ** 2
rhs_batch = jnp.stack([f_in, 0.5 * f_in, 0.1 * f_in])

psi_batch = jax.vmap(
    lambda r, l: fvx.solve_helmholtz_dst(r, dx, dy, l)
)(rhs_batch, lambdas)
```

---

## Capacitance Matrix Solver

The capacitance method extends spectral solvers to irregular domains
defined by a mask.

### Building the Solver

The offline step factorises the capacitance matrix.  It costs $N_c$
rectangular solves plus a dense $N_c \times N_c$ factorisation
($N_c$ = number of coast cells) and stores the $N_c \times N_c$ matrix, so
it grows with the coastline: see the measured build times and memory in the
[Solver Comparison](solver_comparison.md).  The solver's unknowns are
**every wet cell**, with $\psi = 0$ on the coast (the dry cells next to
water): the same problem `method="cg"` solves on that mask, so the two
agree to CG's tolerance.

```python
# Binary mask: True = ocean, False = land
ocean_mask = np.ones((Ny, Nx), dtype=bool)
ocean_mask[:5, :] = ocean_mask[-5:, :] = False  # land border
ocean_mask[:, :5] = ocean_mask[:, -5:] = False

# Build solver (offline)
solver = fvx.build_capacitance_solver(
    ocean_mask, dx, dy,
    lambda_=lambda_,  # Helmholtz parameter, λ ≥ 0
    base_bc="dst",    # rectangular base solver (the default)
)
```

`base_bc` picks the rectangular base (`"dst"`, the default, `"fft"` or
`"dct"`).  It only matters beyond the array edge, so every base gives the
same solution when the mask's wet cells stay off the array edge (a dry
ghost ring); `"fft"` and `"dct"` raise when a wet cell touches the edge,
where only `"dst"` reproduces `masked_laplacian`.  A singular base (`"fft"` / `"dct"` at $\lambda = 0$) is
handled exactly, but `"dst"` is non-singular for every $\lambda \ge 0$.
A plain array mask is used as it is: it is the ψ mask, not a T mask.

### Using with Mask2D

When you have a `Mask2D`, pass it directly — the solver uses its ψ
(X-point) mask `xy_corner_strict`, so ψ = 0 falls on the C-grid wall nodes:

```python
cgrid_mask = fvx.Mask2D.from_mask(ocean)
solver = fvx.build_capacitance_solver(cgrid_mask, dx, dy, lambda_=lambda_)
```

### Online Solve

```python
psi = solver(zeta)  # JIT-compilable, vmap-compatible
```

The solver returns $\psi = 0$ on every dry cell (the coast included) and
solves the 5-point equation on every wet cell; the online cost is two
rectangular solves plus an $O(N_c^2)$ back-substitution.

---

## Conjugate Gradient Solver

For domains where the capacitance matrix is too large, use the
preconditioned CG solver, which is gaussx's (`gaussx.linear_solve_with_info` with `gaussx.CGSolver`).

### Basic Usage

```python
psi_mask = jnp.asarray(mask.xy_corner_strict, dtype=float)  # psi's wet nodes
rhs = zeta * psi_mask

# Define the operator: (∇² − λ) with psi = 0 on the dry neighbours
def A(x):
    return fvx.masked_laplacian(x, psi_mask, dx, dy, lambda_=lambda_)

# gaussx solves (-A) x = -b, so a preconditioner approximating A^-1 is passed as -M^-1
A_op = gaussx.as_linear_operator(
    A, in_structure=jax.ShapeDtypeStruct(rhs.shape, rhs.dtype), negative_definite=True
)
spectral = fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="fft")
psi, info = gaussx.linear_solve_with_info(
    A_op,
    rhs,
    solver=gaussx.CGSolver(
        rtol=1e-8, atol=1e-8 * float(jnp.max(jnp.abs(rhs))), max_steps=500
    ),
    preconditioner=lambda r: -spectral(r),
)
psi = psi * psi_mask  # zero out land points
```

Scale `atol` to the right-hand side as above rather than passing
`atol=0.0`: the stopping rule divides each residual entry by
`atol + rtol |rhs_i|`, which is `0` where the masked right-hand side is
zero, and the resulting `NaN` stops CG after one iteration.

### Convergence Info

`gaussx.linear_solve_with_info` returns `(x, info)`:

```python
psi, info = gaussx.linear_solve_with_info(A_op, rhs, solver=gaussx.CGSolver(throw=False))
print(f"Converged: {info.converged}")
print(f"Iterations: {info.iterations}")
print(f"Residual norm: {info.residual_norm:.2e}")
```

`gaussx.CGSolver` raises when CG fails unless you pass `throw=False`:
hitting `max_steps`, an indefinite operator (`lambda_ < 0`) or a
preconditioner of the wrong sign all give `converged=False`.
The solve is differentiable with respect to `lambda_` and `dx`.
Its output can be non-zero on land (the spectral preconditioners ignore the mask), so
multiply by the mask as above; the wrappers do.

### Preconditioners

:::::{tab-set}
::::{tab-item} Spectral (cheapest)

```python
M_inv = fvx.make_spectral_preconditioner(
    dx, dy, lambda_=lambda_, bc="fft"
)
```

Uses the rectangular spectral solver on the whole array as an
approximate inverse.  No setup and one transform pair per iteration,
but it ignores the mask and puts its wall at the array edge, so it is
not exact even on a rectangular basin: CG still takes 15–63 iterations
at 128²–256² (see the [Solver Comparison](solver_comparison.md)).
::::
::::{tab-item} Nyström (operator-only)

```python
# gaussx owns it: a randomised Nyström approximation of the operator, then
# gaussx.NystromPreconditioner on its basis, eigenvalues and shift
lowrank = gaussx.randomized_nystrom(A_op, rank=100, key=jax.random.PRNGKey(0))
```

Builds a low-rank approximate inverse by probing the operator with
random vectors.  Only needs `matvec` access — useful when you have a
black-box operator with no analytic structure to exploit.

:::{warning} Nyström is a niche preconditioner
For standard Helmholtz/Poisson problems on known grids, the spectral
or multigrid preconditioners are significantly more effective.
Nyström captures only `rank` directions of the inverse; the remaining
directions receive a scalar fallback, so the iteration count may not
improve much over unpreconditioned CG.  Consider Nyström only when
no other preconditioner is available (e.g., a non-standard operator
known only through `matvec`).
:::
::::
::::{tab-item} Multigrid (most powerful)

```python
mg = fvx.build_multigrid_solver(mask, dx, dy, lambda_=lambda_)  # coeff= for c(x, y)
pre = mg.as_preconditioner()  # a gaussx.MultigridPreconditioner, on flat vectors
# pass it to gaussx.linear_solve_with_info(..., solver=gaussx.CGSolver(...), preconditioner=pre)
# on the flat finest operator pre.levels[0].operator
```

A single multigrid V-cycle as an approximate inverse.  Captures both
high- and low-frequency error across the grid hierarchy, handles
variable coefficients and masked domains natively, and its operator is
exactly `masked_laplacian` on the same mask.  CG needs 5–7 iterations
at 128² and 256² on a rectangle and an island basin, independent of the
grid size.
::::
::::{tab-item} Custom

```python
def my_preconditioner(r):
    """Any callable (Ny, Nx) -> (Ny, Nx) that approximates A^{-1}."""
    return -r / (2 / dx**2 + 2 / dy**2 + lambda_)  # Jacobi: 1 / diag(A)

psi, info = gaussx.linear_solve_with_info(
    A_op, rhs, solver=gaussx.CGSolver(throw=False),
    preconditioner=lambda r: -my_preconditioner(r),
)
```
::::
:::::

:::{tip} Which preconditioner should I use?
See the [Preconditioner Decision Guide](elliptic_solvers.md#decision-guide)
in the theory page and the measured
[Solver Comparison](solver_comparison.md).  **TL;DR**: on a masked
domain, CG + multigrid takes the fewest iterations and is the fastest
CG; the spectral preconditioners need no setup but 3–13× more
iterations.
:::

---

## JIT and vmap Compatibility

All solvers are fully compatible with `jax.jit` and `jax.vmap`.

### JIT Compilation

```python
def invert_layer(q, lam):
    unknowns = fvx.solve_helmholtz_dst1(q[1:-2, 1:-2], dx, dy, lam)
    return jnp.zeros_like(q).at[1:-2, 1:-2].set(unknowns)

@jax.jit
def invert_pv(pv_field, lambdas):
    return jax.vmap(invert_layer)(pv_field, lambdas)

# One 1/Rd² per layer; pv is [nl, Ny, Nx]
lambdas = 1.0 / jnp.array([40e3, 20e3, 10e3]) ** 2
pv = jnp.stack([zeta, 0.5 * zeta, 0.1 * zeta])
psi = invert_pv(pv, lambdas)
```

### Batched Solves with vmap

```python
# Solve the same equation for many RHS fields
@jax.jit
def batch_solve(rhs_batch):
    return jax.vmap(lambda r: fvx.solve_poisson_dst(r, dx, dy))(rhs_batch)

rhs_ensemble = jnp.stack([f_in, -f_in])
psi_batch = batch_solve(rhs_ensemble)  # (n_ensemble, Ny-3, Nx-3)
```

### Gradient Through Solves

Spectral solvers are differentiable — you can backpropagate through them:

```python
def loss(rhs):
    psi = fvx.solve_poisson_dst(rhs, dx, dy)
    return jnp.sum(psi ** 2)

grad_rhs = jax.grad(loss)(f_in)
```
