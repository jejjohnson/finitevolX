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

Preconditioning transforms the system to improve the condition number,
dramatically reducing the number of CG iterations.

| Preconditioner | How it works | Best for |
|---|---|---|
| **Spectral** | Applies the rectangular spectral solver as $M^{-1} \approx A_{\text{rect}}^{-1}$ | Domains that are close to rectangular |
| **Nyström** | Low-rank approximate inverse from randomised probing of $A$ | Large or complex domains; operator-only access |

The spectral preconditioner is essentially free (one FFT/DST/DCT pair) and
is the default.  The Nyström preconditioner requires $k$ operator
applications during setup but can capture more of the operator's structure.

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
