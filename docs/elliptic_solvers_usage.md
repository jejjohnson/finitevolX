# Elliptic Solvers: Usage Guide

This page covers practical usage of finitevolX's elliptic solver module,
from quick-start examples to multi-layer PV inversion on masked domains.
The snippets run in order: later ones reuse the names the Quick Start
defines.

---

## Quick Start

The fastest way to solve an elliptic problem: use a convenience wrapper.

```python
import jax
import jax.numpy as jnp
import numpy as np
import finitevolx as fvx

jax.config.update("jax_enable_x64", True)

# 1. A 64 x 64-cell basin, 100 km on a side.  Arrays are [Ny, Nx] = [66, 66]:
#    the interior plus one ghost ring.
grid = fvx.CartesianGrid2D.from_interior(64, 64, 1e5, 1e5)
dx, dy, Ny, Nx = grid.dx, grid.dy, grid.Ny, grid.Nx

# 2. Vorticity at the X points, where psi lives (X node i sits at x = i dx)
x = jnp.arange(Nx)[None, :] * dx
y = jnp.arange(Ny)[:, None] * dy
zeta = jnp.sin(jnp.pi * x / grid.Lx) * jnp.sin(jnp.pi * y / grid.Ly)  # [Ny, Nx]

# 3. Invert for the streamfunction, psi = 0 on the C-grid walls
psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")  # [Ny, Nx]
```

Pass the full `[Ny, Nx]` array and get the full array back: the wrappers
know which nodes are unknowns (see [Staggering and walls](#staggering-and-walls)),
so do not slice the interior by hand.

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

## Convenience Wrappers

finitevolX provides three high-level wrappers that handle solver dispatch
internally.  All three accept the same `method`, `mask`, `capacitance_solver`,
and `preconditioner` keyword arguments.

### Streamfunction from Vorticity

Solves $\nabla^2 \psi - \lambda\psi = \zeta$ for ψ at X points.

=== "Spectral (rectangular domain)"

    ```python
    # Dirichlet BCs (ψ = 0 on the C-grid wall nodes, X index 0 and N-2) — most common
    psi = fvx.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")

    # With a Helmholtz parameter (QG inversion, λ = 1/Rd²)
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, bc="dst", lambda_=lambda_
    )
    ```

=== "CG (masked domain)"

    ```python
    # A Mask2D gives the psi mask xy_corner_strict: psi = 0 on the coast nodes
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, method="cg", mask=mask, lambda_=lambda_
    )
    ```

=== "Capacitance (masked domain, fast)"

    ```python
    # One-time precomputation, for this mask and this λ
    solver = fvx.build_capacitance_solver(mask, dx, dy, lambda_=lambda_)

    # Reuse for many solves
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, lambda_=lambda_, method="capacitance",
        capacitance_solver=solver,
    )
    ```

A plain array `mask=` is taken as the mask of the field being solved for
(ψ: X points); a `Mask2D` supplies the right one itself.

### Pressure from Divergence

Solves $\nabla^2 p = \nabla \cdot \mathbf{u}$ (always Poisson, $\lambda = 0$)
for p at T points.

```python
div_u = zeta  # any T-point field; the Neumann solve drops its mean (gauge)

# Neumann BCs (∂p/∂n = 0 on the cell faces) — standard for pressure with solid walls
p = fvx.pressure_from_divergence(div_u, dx, dy, bc="dct")

# The same Neumann wall on a masked basin: zero flux through every coast face
p = fvx.pressure_from_divergence(div_u, dx, dy, method="cg", mask=mask)
```

`bc` is honoured by every method or rejected, never ignored.  The CG wall
is the spectral one of the same `bc`: with `method="cg"`, `bc="dct"` (the
default) is zero flux through every wet–dry face and returns the solution
with zero mean on each connected wet region; `bc="dst"` holds p = 0 on the
wet–dry *faces*, half a cell out (the DST-II wall).  The default
preconditioner is the spectral solve with the same `bc`.
`method="capacitance"` raises for pressure (it would hold p = 0 at the dry
cell centres, not on the faces), and so do `known_values` / `known_mask`;
`bc="fft"` is for the rectangular spectral solve only.

### Staggering and walls

The wrappers know the staggering of what they solve for (ψ at X points,
p at T points), pass only its unknowns to the transform and return them
embedded in zeros, so the wall is the C-grid wall for every `bc` and
`method`; see the [wall table](elliptic_solvers.md#boundary-conditions).

### PV Inversion (Multi-Layer)

Solves $(\nabla^2 - \lambda_k)\,\psi_k = q_k$ for each vertical mode, with
$\lambda_k = 1/R_{d,k}^2 \ge 0$.

=== "Single layer"

    ```python
    pv = zeta
    psi = fvx.pv_inversion(pv, dx, dy, lambda_=lambda_, bc="dst")
    ```

=== "Multi-layer (per-mode λ)"

    ```python
    # lambda_ array: one value per vertical mode, 1/Rd² per mode
    lambdas = 1.0 / jnp.array([40e3, 20e3, 10e3]) ** 2

    # pv shape: (nl, Ny, Nx) or (batch, nl, Ny, Nx)
    pv = jnp.stack([zeta, 0.5 * zeta, 0.1 * zeta])
    psi = fvx.pv_inversion(pv, dx, dy, lambda_=lambdas, bc="dst")
    ```

=== "With vertical mode decomposition"

    ```python
    # Decompose layer PV into vertical modes
    H_layers = jnp.array([500.0, 1500.0, 3000.0])  # thicknesses [m]
    g_prime = jnp.array([9.81, 0.025, 0.0125])     # reduced gravities [m s⁻²]
    A_vert = fvx.build_coupling_matrix(H_layers, g_prime)
    radii, Cl2m, Cm2l = fvx.decompose_vertical_modes(A_vert, f0=1e-4)
    lambdas_k = 1.0 / radii**2  # 1/Rd_k² ≥ 0 per mode

    # Transform to mode space
    pv_layers = jnp.stack([zeta, 0.5 * zeta, 0.1 * zeta])
    pv_modes = fvx.layer_to_mode(pv_layers, Cl2m)

    # Invert each mode with its own λ_k
    psi_modes = fvx.pv_inversion(pv_modes, dx, dy, lambda_=lambdas_k, bc="dst")

    # Transform back to layer space
    psi_layers = fvx.mode_to_layer(psi_modes, Cm2l)
    ```


### Known Boundary Values (Inhomogeneous Dirichlet)

!!! tip "Tutorial"
    For the full derivation — notation, the block linear system, pseudocode,
    a from-scratch NumPy implementation, convergence and gradients — see the
    [Known Boundary Values tutorial](notebooks/inhomogeneous_bcs_tutorial.py).

By default every solver assumes $\psi = 0$ on the boundary.  When the
boundary values are known but non-zero — SSH from a parent model, a tide
gauge, reanalysis at an open boundary — pass them as `known_values`.
`streamfunction_from_vorticity` and `pv_inversion` accept it
(`pressure_from_divergence` refuses it, see below):

```python
# psi rising from 0 (south) to 1e3 (north): a throughflow of 1e3 m² s⁻¹
g = 1e3 * y / grid.Ly * jnp.ones((Ny, Nx))  # [Ny, Nx]; only the known cells are read
psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy, method="cg", mask=mask, known_values=g
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
`fvx.coast_ring(mask)`.  The T-point `pressure_from_divergence` raises on
known values: its Dirichlet wall is on the cell faces, while the lifting
pins values at cell centres (the ghost cells, half a cell outside).

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
        zeta, dx, dy, lambda_=lambda_, method="cg", mask=mask, known_values=g
    )
    ```

=== "Capacitance"

    ```python
    # Build on the SAME mask: its unknowns are every wet cell with
    # psi = 0 on the coast, exactly the cells left to solve.
    solver = fvx.build_capacitance_solver(mask, dx, dy, lambda_=lambda_)
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, lambda_=lambda_, method="capacitance", mask=mask,
        capacitance_solver=solver, known_values=g,
    )
    ```

=== "Spectral (rectangular basin)"

    ```python
    # No mask: solves the psi basin of the rectangle -- wet X nodes
    # [1:-2, 1:-2] -- with the known values on its wall nodes (rows/columns
    # 0 and N-2).  Requires bc="dst".
    psi = fvx.streamfunction_from_vorticity(
        zeta, dx, dy, bc="dst", lambda_=lambda_, known_values=g
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
values = jnp.zeros((Ny, Nx)).at[20, 30].set(120.0).at[25, 45].set(-50.0)

psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy, method="cg", mask=mask,
    known_values=values, known_mask=obs,
)
# psi[20, 30] == 120.0 and psi[25, 45] == -50.0 exactly
```

#### Multi-layer PV inversion

`known_values` broadcasts against `pv`: a `(Ny, Nx)` field is shared by all
layers, an `(nl, Ny, Nx)` array gives each layer its own values.  Each
layer's correction uses its own $\lambda_k$.

```python
psi = fvx.pv_inversion(
    pv, dx, dy, lambda_=lambdas, method="cg", mask=mask,
    known_values=g,   # (Ny, Nx) or (nl, Ny, Nx)
)
```

#### Boundary conditions as a `BoundaryConditionSet`

`bc` also accepts a `BoundaryConditionSet`.  Its face types choose the
transform (all `Dirichlet1D` → `"dst"`, all zero `Neumann1D` → `"dct"`,
all `Periodic1D` → `"fft"`), its `mask` feeds the mask-based methods, and
non-zero `Dirichlet1D` values become known values on each face's wall
nodes (the coast cells on the row / column just outside the wet cells: X
rows `0` / `N-2` and columns `0` / `N-2` of a rectangle).  That needs a
basin whose outer coast lies on the four walls (a rectangle; island coasts
are allowed and stay at zero): on any other `Mask2D` non-zero face values
raise, so pass `known_values` on `fvx.coast_ring(mask)` instead.  A zero face
value can be a Python number or a concrete array (`jnp.array(0.0)`); a
traced value cannot be inspected and counts as non-zero.

```python
rect_mask = fvx.Mask2D.from_mask(np.pad(np.ones((Ny - 2, Nx - 2), dtype=bool), 1))
bc = fvx.BoundaryConditionSet(
    mask=rect_mask,                              # a Mask2D, or omit for method="spectral"
    south=fvx.Dirichlet1D("south", value=0.0),
    north=fvx.Dirichlet1D("north", value=1e3),   # prescribed psi on the north wall
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
lifter = fvx.KnownValueLifting(domain, dx, dy, lambda_=lambda_)
eff = domain.effective_mask.astype(float)
A = lambda x: fvx.masked_laplacian(x, eff, dx, dy, lambda_=lambda_)

# Per step (JIT-friendly): correct, solve, reconstruct
known_values = jnp.where(obs, values, g)
rhs_corrected, value_lift = lifter.preprocess(zeta, known_values)
psi_hom, info = fvx.solve_cg(A, rhs_corrected, rtol=1e-10, atol=1e-10)
psi = lifter.postprocess(psi_hom * eff, value_lift)
```

The solver's unknowns must be exactly `domain.effective_mask` (every wet
cell not in `known_mask`), with zero on the coast: build a CG operator on
`domain.effective_mask`.

For **multigrid**, use it as the CG preconditioner — the wrapper then does
the lifting for you:

```python
mg = fvx.build_multigrid_solver(
    np.asarray(domain.effective_mask, dtype=float), dx, dy, lambda_=lambda_
)
psi = fvx.streamfunction_from_vorticity(
    zeta, dx, dy, lambda_=lambda_, method="cg", mask=mask,
    known_values=known_values, known_mask=obs,
    preconditioner=fvx.make_multigrid_preconditioner(mg),
)
```

The multigrid operator (default `bc="dirichlet"`) is `masked_laplacian`'s
stencil with dry neighbours zero, so it is the same operator CG solves; the
known-value lifting is done by the wrapper, so a standalone multigrid solve
needs the lifted right-hand side.

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
p_in = fvx.solve_poisson_dct(div_u[1:-1, 1:-1], dx, dy)

# Periodic (FFT): doubly-periodic domain (zero-mean gauge)
psi_per = fvx.solve_poisson_fft(f_in, dx, dy)

# What streamfunction_from_vorticity(zeta, dx, dy, bc="dst") returns
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
preconditioned CG solver.

### Basic Usage

```python
psi_mask = jnp.asarray(mask.xy_corner_strict, dtype=float)  # psi's wet nodes
rhs = zeta * psi_mask

# Define the operator: (∇² − λ) with psi = 0 on the dry neighbours
def A(x):
    return fvx.masked_laplacian(x, psi_mask, dx, dy, lambda_=lambda_)

# Solve with a spectral preconditioner
psi, info = fvx.solve_cg(
    A, rhs,
    preconditioner=fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="fft"),
    rtol=1e-8,
    atol=1e-8 * float(jnp.max(jnp.abs(rhs))),
)
psi = psi * psi_mask  # zero out land points
```

Scale `atol` to the right-hand side as above rather than passing
`atol=0.0`: lineax's stopping rule divides each residual entry by
`atol + rtol |rhs_i|`, which is `0` where the masked right-hand side is
zero, and the resulting `NaN` stops CG after one iteration (`solve_cg` then
reports `converged=False`).

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
        dx, dy, lambda_=lambda_, bc="fft"
    )
    ```

    Uses the rectangular spectral solver on the whole array as an
    approximate inverse.  No setup and one transform pair per iteration,
    but it ignores the mask and puts its wall at the array edge, so it is
    not exact even on a rectangular basin: CG still takes 15–63 iterations
    at 128²–256² (see the [Solver Comparison](solver_comparison.md)).

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
    mg = fvx.build_multigrid_solver(mask, dx, dy, lambda_=lambda_)  # coeff= for c(x, y)
    M_inv = fvx.make_multigrid_preconditioner(mg)
    psi, info = fvx.solve_cg(A, rhs, preconditioner=M_inv, rtol=1e-8)
    ```

    A single multigrid V-cycle as an approximate inverse.  Captures both
    high- and low-frequency error across the grid hierarchy, handles
    variable coefficients and masked domains natively, and its operator is
    exactly `masked_laplacian` on the same mask.  CG needs 5–7 iterations
    at 128² and 256² on a rectangle and an island basin, independent of the
    grid size.

=== "Factory (dispatches by name)"

    ```python
    # Dispatches to spectral, nystrom, or multigrid
    M_inv = fvx.make_preconditioner("spectral", dx=dx, dy=dy, lambda_=lambda_)
    M_inv = fvx.make_preconditioner(
        "nystrom", matvec=A, shape=(Ny, Nx), key=jax.random.PRNGKey(0)
    )
    M_inv = fvx.make_preconditioner("multigrid", mg_solver=mg)
    ```

    Convenient when the preconditioner choice is a configurable parameter.

=== "Custom"

    ```python
    def my_preconditioner(r):
        """Any callable (Ny, Nx) -> (Ny, Nx) that approximates A^{-1}."""
        return -r / (2 / dx**2 + 2 / dy**2 + lambda_)  # Jacobi: 1 / diag(A)

    psi, info = fvx.solve_cg(A, rhs, preconditioner=my_preconditioner)
    ```

!!! tip "Which preconditioner should I use?"
    See the [Preconditioner Decision Guide](elliptic_solvers.md#decision-guide)
    in the theory page and the measured
    [Solver Comparison](solver_comparison.md).  **TL;DR**: on a masked
    domain, CG + multigrid takes the fewest iterations and is the fastest
    CG; the spectral preconditioners need no setup but 3–13× more
    iterations.

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
