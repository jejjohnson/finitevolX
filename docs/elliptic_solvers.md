# Elliptic Solvers for Ocean Models

This page covers the theory behind the elliptic solvers in finitevolX,
explains when each method applies, and provides practical guidance for
choosing a solver.

---

## The Problem

Many ocean and atmosphere modelling tasks require solving 2-D elliptic
partial differential equations of the form

$$
(\nabla^2 - \lambda)\,\psi = f
$$

where $\nabla^2 = \partial^2/\partial x^2 + \partial^2/\partial y^2$ is the
discrete 5-point Laplacian, $\lambda$ is a Helmholtz parameter, and $f$ is a
known right-hand side.

Common physical applications:

| Problem | Equation | $\lambda$ |
|---|---|---|
| **Streamfunction from vorticity** | $\nabla^2 \psi = \zeta$ | 0 (Poisson) |
| **Pressure correction** | $\nabla^2 p = \nabla \cdot \mathbf{u}$ | 0 (Poisson) |
| **QG PV inversion** | $(\nabla^2 - 1/R_d^2)\,\psi = q$ | $1/R_d^2$ (Helmholtz) |
| **Implicit diffusion** | $(\nabla^2 - 1/(\nu\,\Delta t))\,T = -T^n/(\nu\,\Delta t)$ | $1/(\nu\,\Delta t)$ |

---

## Boundary Conditions

The choice of boundary condition determines which spectral transform is used:

| BC type | Physical meaning | Transform | finitevolX solver |
|---|---|---|---|
| **Dirichlet** | $\psi = 0$ on all edges | DST-I | `solve_poisson_dst` / `solve_helmholtz_dst` |
| **Neumann** | $\partial\psi/\partial n = 0$ on all edges | DCT-II | `solve_poisson_dct` / `solve_helmholtz_dct` |
| **Periodic** | $\psi$ wraps in both directions | FFT | `solve_poisson_fft` / `solve_helmholtz_fft` |

!!! tip "When to use which BC"
    - **Dirichlet (DST)**: Streamfunction inversion in a closed basin
      ($\psi = 0$ on coastline).  Most common for vorticity inversion.
    - **Neumann (DCT)**: Pressure correction with solid walls
      ($\partial p / \partial n = 0$).
    - **Periodic (FFT)**: Doubly-periodic domains (idealised studies,
      turbulence simulations).

---

## Spectral Solver Theory

All three spectral solvers follow the same three-step algorithm:

1. **Forward transform**: project $f$ onto the spectral basis.
2. **Spectral division**: divide each coefficient by its eigenvalue.
3. **Inverse transform**: recover $\psi$ in physical space.

The eigenvalues of the discrete 5-point Laplacian depend on the transform:

| Transform | Eigenvalue $\lambda_k$ | Null mode |
|---|---|---|
| DST-I ($N$ interior pts) | $-4/\Delta x^2 \cdot \sin^2\!\bigl(\pi(k+1)/(2(N+1))\bigr)$ | None (all $< 0$) |
| DCT-II ($N$ pts) | $-4/\Delta x^2 \cdot \sin^2\!\bigl(\pi k/(2N)\bigr)$ | $k=0$ ($\lambda_0 = 0$) |
| FFT ($N$ pts) | $-4/\Delta x^2 \cdot \sin^2\!\bigl(\pi k/N\bigr)$ | $k=0$ ($\lambda_0 = 0$) |

The 2-D eigenvalue grid is the outer sum of 1-D eigenvalues:
$\Lambda_{j,i} = \lambda_j^y + \lambda_i^x - \lambda$.

For the Poisson equation ($\lambda = 0$) with Neumann or periodic BCs, the
$(0,0)$ eigenvalue is zero (null space = constant solution).  The solvers
handle this by enforcing a **zero-mean gauge**: the $(0,0)$ spectral
coefficient is set to zero, selecting the unique zero-mean solution.

!!! note "Complexity"
    All spectral solvers run in $O(N_y N_x \log(N_y N_x))$ time —
    the cost of the forward and inverse transforms.  There is no iterative
    convergence loop.

---

## Irregular Domains: Capacitance Matrix Method

Real ocean basins are not rectangles.  The **capacitance matrix method**
(Buzbee, Golub & Nielson, 1970) extends fast spectral solvers to domains
defined by a binary mask.

### Algorithm

Given a mask $M$ (True = ocean, False = land) and $N_b$ inner-boundary
points $\{b_k\}$ (ocean cells adjacent to land):

1. **Precompute** (offline, $O(N_b)$ spectral solves):
   - For each boundary point $b_k$, solve the rectangular problem
     $L_{\text{rect}}\,g_k = e_{b_k}$ to get Green's function $g_k$.
   - Build the capacitance matrix $C_{k,l} = g_l(b_k)$ and invert it.

2. **Solve** (online, one spectral solve + $O(N_b^2)$ correction):
   - Compute the rectangular solution: $u = L_{\text{rect}}^{-1} f$.
   - Enforce $\psi(b_k) = 0$ by solving $C \alpha = u[B]$ for correction
     coefficients.
   - Return $\psi = u - \sum_k \alpha_k g_k$.

The online cost is dominated by the single spectral solve, making this
method nearly as fast as a rectangular solver.

!!! warning "Memory"
    The Green's function matrix is $O(N_b \times N_y \times N_x)$.
    For very large $N_b$ (complex coastlines), consider the CG solver
    instead.

---

## Iterative Solver: Preconditioned Conjugate Gradient

For problems where the capacitance matrix is too large, or when you need
flexibility (e.g., variable coefficients), finitevolX provides a
**preconditioned Conjugate Gradient (CG)** solver via
[Lineax](https://docs.kidger.site/lineax/).

The CG method iteratively minimises the quadratic form associated with the
linear system $A\psi = f$, where $A$ is a symmetric operator (the masked
Laplacian).

### Preconditioners

Preconditioning transforms the system $Ax = b$ into $M^{-1}Ax = M^{-1}b$
where $M^{-1} \approx A^{-1}$.  A good preconditioner clusters the
eigenvalues of $M^{-1}A$ near 1, dramatically reducing the number of CG
iterations.

finitevolX provides three preconditioners, each suited to different problem
characteristics.  The table below ranks them from simplest to most powerful:

| Preconditioner | Setup cost | Per-iteration cost | Constant coeff | Variable coeff | Masked domain |
|---|---|---|---|---|---|
| **Spectral** | None | 1 FFT pair | Excellent | Poor | Fair |
| **Nyström** | $k$ matvecs | $O(kN)$ dot product | Good | Fair | Good |
| **Multigrid** | Offline hierarchy build | 1 V-cycle ($O(N)$) | Excellent | Excellent | Excellent |

#### Spectral Preconditioner

Applies the rectangular spectral solver (DST/DCT/FFT) as an approximate
inverse: $M^{-1} r = L_{\text{rect}}^{-1} r$.

**Pros:**

- Essentially free — one forward + one inverse transform per iteration
- No setup cost
- Exact inverse for rectangular, constant-coefficient problems

**Cons:**

- Ignores the mask: the rectangular solve doesn't know about land cells,
  so the preconditioner becomes less effective as the domain deviates from
  a rectangle
- Cannot handle variable coefficients $c(x,y)$ — it always assumes $c = 1$
- Effectiveness degrades for highly irregular coastlines

**Best for:** Constant-coefficient problems on rectangular or
near-rectangular domains.  This is the default and a good first choice.

#### Nyström Preconditioner

Builds a low-rank approximate inverse by probing the operator with $k$
random vectors.  The resulting preconditioner captures the dominant
spectral directions of $A^{-1}$.

**Pros:**

- Operator-only access: works with any `matvec` callable, no need to know
  the operator's structure
- Can partially capture variable-coefficient and mask effects
- Rank $k$ is tunable — trade setup cost for preconditioner quality

**Cons:**

- Setup requires $k$ operator applications (can be expensive for large $k$)
- Low-rank: misses fine-scale structure, especially for poorly conditioned
  problems
- Per-iteration cost is $O(kN)$ rather than $O(N \log N)$ for spectral
- Quality depends on the rank $k$ relative to the effective rank of
  $A^{-1}$

**Best for:** Problems where the operator is available only as a callable
(no analytic structure to exploit), or as a complement when spectral
preconditioning is insufficient but multigrid is not needed.

#### Multigrid Preconditioner

Applies a single multigrid V-cycle from a zero initial guess.  Because
multigrid captures both high- and low-frequency components of $A^{-1}$
across the grid hierarchy, it provides a spectrally complete approximation.

**Pros:**

- Handles variable coefficients $c(x,y)$ natively — the hierarchy is built
  with the actual operator
- Handles masked/irregular domains natively — masks are coarsened through
  the hierarchy
- $O(N)$ cost per iteration (geometric series over grid levels)
- Typically reduces CG iterations from hundreds to 5–10

**Cons:**

- Requires an offline build step (`build_multigrid_solver`) that
  precomputes the level hierarchy
- Grid dimensions must be divisible by $2^{L-1}$ where $L$ is the number
  of levels
- More complex implementation; slightly higher constant factor than
  spectral

**Best for:** Variable-coefficient problems, complex masked domains, or any
problem where spectral preconditioning converges too slowly.

#### Decision Guide

```
Is c(x,y) spatially varying?
├── Yes → Multigrid preconditioner
│         (spectral and Nyström can't represent variable coefficients well)
└── No (constant coefficient) ↓

Is the domain nearly rectangular?
├── Yes → Spectral preconditioner (cheapest, nearly exact)
└── No (complex mask) ↓

Is the mask moderately complex?
├── Yes → Multigrid preconditioner
│         (captures mask effects through coarsened hierarchy)
└── Operator-only access / quick prototype ↓

Default → Nyström preconditioner (rank 30–100)
          or Multigrid if you can build the hierarchy
```

#### Factory Function

`make_preconditioner` dispatches to any of the three preconditioners based
on a string key, which is convenient when the preconditioner choice is
configurable:

```python
pc = fvx.make_preconditioner("spectral", dx=dx, dy=dy, lambda_=1.0)
pc = fvx.make_preconditioner("nystrom", matvec=A, shape=(64, 64), rank=50)
pc = fvx.make_preconditioner("multigrid", mg_solver=mg)
```

---

## Multi-Layer / Batched Solves

QG PV inversion often requires solving a separate Helmholtz equation for
each vertical mode, each with its own $\lambda_k = 1/R_{d,k}^2$:

$$
(\nabla^2 - \lambda_k)\,\psi_k = q_k, \quad k = 1, \ldots, n_l
$$

finitevolX handles this efficiently via `jax.vmap`:

- **Array `lambda_`**: Pass a 1-D array of shape `(nl,)` and a PV field of
  shape `(..., nl, Ny, Nx)`.  Each layer is solved with its own $\lambda_k$.
- **Scalar `lambda_`**: All layers/batch elements use the same parameter.
- **Batch dimensions**: Leading dimensions beyond `(nl, Ny, Nx)` are
  automatically preserved and vmapped over.

All spectral solvers are **tracer-safe** — they use `jnp.where` guards
instead of Python `if` branches, so they work correctly inside `jax.vmap`
and `jax.jit` even when `lambda_` is a JAX tracer.

---

## Decision Guide

```
Is the domain a full rectangle?
├── Yes → Use spectral solver (method="spectral")
│         ├── ψ = 0 on boundary → bc="dst"
│         ├── ∂ψ/∂n = 0 on boundary → bc="dct"
│         └── Periodic → bc="fft"
└── No (masked/irregular domain) ↓

Is the coastline simple (small N_b)?
├── Yes → Use capacitance method (method="capacitance")
│         Pre-build with build_capacitance_solver()
└── No (complex coastline, large N_b) ↓

Default → Use CG (method="cg")
          Optionally with spectral or Nyström preconditioner
```

---

## References

- Buzbee, Golub & Nielson (1970) — On direct methods for solving Poisson's
  equations (capacitance matrix method)
- Arakawa & Lamb (1977) — Computational design of the basic dynamical
  processes of the UCLA general circulation model
- Vallis (2017) — *Atmospheric and Oceanic Fluid Dynamics*, Ch. 5
  (QG potential vorticity inversion)
- Hestenes & Stiefel (1952) — Methods of conjugate gradients for solving
  linear systems
