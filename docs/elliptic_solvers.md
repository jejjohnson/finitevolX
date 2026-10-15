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
| **Dirichlet** | $\psi = 0$ on the virtual nodes one step outside the input array | DST-I | `solve_poisson_dst` / `solve_helmholtz_dst` |
| **Neumann** | $\partial\psi/\partial n = 0$ half a cell outside the first / last input node | DCT-II | `solve_poisson_dct` / `solve_helmholtz_dct` |
| **Periodic** | $\psi$ wraps in both directions | FFT | `solve_poisson_fft` / `solve_helmholtz_fft` |

The direct solvers treat **every** input node as an unknown, so where the
wall ends up depends on what you pass them.  The convenience wrappers
(`streamfunction_from_vorticity`, `pv_inversion`, `pressure_from_divergence`)
slice the unknowns of their C-grid staggering out of the `[Ny, Nx]` array,
solve, and embed the result in zeros, so the wall is the C-grid wall.  On a
`from_interior` grid the T cells are `[1:-1]` and the walls are the faces at
$x = 0$ and $x = L$; X node $i$ sits at $x = i\,\Delta x$, so the walls are
the X nodes `0` and `N-2` (index `N-1` is outside the domain):

| Wrapper (staggering) | `bc` | Unknowns | Transform | Wall |
|---|---|---|---|---|
| ψ (X) | `"dst"` | `[1:-2, 1:-2]` | DST-I | $\psi = 0$ on the wall nodes `0`, `N-2` |
| ψ (X) | `"dct"` | `[0:-1, 0:-1]` | DCT-I | $\partial\psi/\partial n = 0$ about the wall nodes `0`, `N-2` |
| ψ (X) | `"fft"` | `[1:-1, 1:-1]` | FFT | periodic, period `N-2` |
| p (T) | `"dct"` | `[1:-1, 1:-1]` | DCT-II | $\partial p/\partial n = 0$ on the faces |
| p (T) | `"dst"` | `[1:-1, 1:-1]` | DST-II | $p = 0$ on the faces |
| p (T) | `"fft"` | `[1:-1, 1:-1]` | FFT | periodic, period `N-2` |

This is the same wall `method="cg"` imposes through `masked_laplacian` with
the `Mask2D` of the same staggering (`xy_corner_strict` for ψ, `h` for p),
so every combination converges at second order.

!!! tip "When to use which BC"
    - **Dirichlet (DST)**: Streamfunction inversion in a closed basin
      ($\psi = 0$ on coastline).  Most common for vorticity inversion.
    - **Neumann (DCT)**: Pressure correction with solid walls
      ($\partial p / \partial n = 0$).
    - **Periodic (FFT)**: Doubly-periodic domains (idealised studies,
      turbulence simulations).

---

## Inhomogeneous Dirichlet Boundary Conditions

The table above assumes the boundary values are **zero**.  Prescribed,
non-zero values $\psi = g$ on $\partial\Omega$ — SSH from a parent model, a
tide gauge, reanalysis data — reduce to the homogeneous case by the
**lifting trick**.  Split $\psi = \psi_{\text{lift}} + \psi_{\text{hom}}$
with $\psi_{\text{lift}} = g$ on the boundary; by linearity

$$
(\nabla^2 - \lambda)\,\psi_{\text{hom}}
  = f - (\nabla^2 - \lambda)\,\psi_{\text{lift}},
\qquad \psi_{\text{hom}} = 0 \text{ on } \partial\Omega .
$$

**Discretely**, the *known* cells $K$ are the coast ring $C$ — the dry
cells with a wet neighbour, where `masked_laplacian` puts its homogeneous
wall (for ψ, the C-grid wall nodes) — plus any observed wet cells $O$, and
the *effective* unknowns are $E = W \setminus O$, every other wet cell.
Every stencil neighbour of an $E$ cell is in $E$ or $K$, so the 5-point
equations at $E$ split into
$A_{EE}\,\psi_E + A_{EK}\,g_K = f_E$, i.e.

$$
A_{EE}\,\psi_E = f_E - A_{EK}\,g_K .
$$

This is exact block elimination: the lift is simply "$g$ on $K$, zero
elsewhere", the correction $A_{EK} g_K$ is the unmasked stencil applied to
that lift (the lift lives on dry coast cells), and the homogeneous solve
runs on the **effective** mask.  Accuracy is set by the stencil alone
(second order).

| Solver | How it takes known values |
|--------|---------------------------|
| CG | operator on the effective mask ($-A_{EE}$ is SPD for $\lambda \ge 0$) |
| CG + multigrid | multigrid preconditioner built on the effective mask |
| Spectral (DST) | ψ rectangle only: $E$ is the X basin `[1:-2, 1:-2]` and $K$ its wall nodes (rows / columns `0` and `N-2`) |
| Capacitance | solver built on the **wet** mask (its unknowns are every wet cell, $\psi = 0$ on the coast); no observed cells |

In finitevolX, pass `known_values=` (and optionally `known_mask=`) to
`streamfunction_from_vorticity` or `pv_inversion`, or a
`BoundaryConditionSet` as `bc=` (the T-point `pressure_from_divergence`
refuses known values: its Dirichlet wall is on the cell faces); see the
[usage guide](elliptic_solvers_usage.md#known-boundary-values-inhomogeneous-dirichlet).
The [Known Boundary Values tutorial](notebooks/inhomogeneous_bcs_tutorial.py)
derives all of this from scratch — notation, block structure, pseudocode, a
NumPy implementation, a convergence study, and gradients with respect to
the boundary data.

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

Given a mask $M$ (True = ocean, False = land) and the $N_b$ coast points
$\{b_k\}$ (land cells adjacent to ocean, where $\psi = 0$; every ocean cell
is an unknown, as in `masked_laplacian` / CG):

1. **Precompute** (offline, $O(N_b)$ spectral solves):
   - For each boundary point $b_k$, solve the rectangular problem
     $L_{\text{rect}}\,g_k = e_{b_k}$ to get Green's function $g_k$.
   - Build the capacitance matrix $C_{k,l} = g_l(b_k)$ and invert it.

2. **Solve** (online, two rectangular solves + an $O(N_b^2)$ correction):
   - Compute the rectangular solution: $u = L_{\text{rect}}^{-1} f$.
   - Enforce $\psi(b_k) = 0$ by solving $C \alpha = u[B]$ for correction
     coefficients (the precomputed factorisation of $C$).
   - Return $\psi = L_{\text{rect}}^{-1}(f - \sum_k \alpha_k e_{b_k})$, a
     second rectangular solve, so the Green's functions are never stored.

The online cost is two rectangular solves plus the $O(N_b^2)$
back-substitution: measured on a 2-vCPU CPU it is 5–11× the spectral solve
on a rectangle, as fast as CG + multigrid at 128² and 2–3.6× faster at 256²
(see the [Solver Comparison](solver_comparison.md)).

!!! warning "Setup cost and memory"
    The solver stores the $N_b \times N_b$ capacitance matrix, and the setup
    costs $N_b$ rectangular solves plus an $O(N_b^3)$ factorisation.  For a
    circular basin with an island at 256² ($N_b = 795$) that is 5.8 MiB and
    about 4 s (9 MiB and 5 s for the 1 020 coast nodes of the 256²
    rectangle); both grow with the length of the coastline, so for very
    long coastlines or a mask that changes, use CG + multigrid.

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

| Preconditioner | Setup cost | Per-iteration cost | CG iterations, 128² / 256² (measured) |
|---|---|---|---|
| **Spectral** (FFT / DST) | None | 1 transform pair | 31–63 (FFT), 15–48 (DST): grows with $N$ |
| **Nyström** | $k$ matvecs | $O(kN)$ dot product | not measured (low rank: little gain on these problems) |
| **Multigrid** | Offline hierarchy build (20–60 ms) | 1 V-cycle ($O(N)$) | 5–7, independent of $N$ |

The iteration counts are for a rectangle and a circular basin with an
island, $\lambda \in \{0, 4\}$, $\max|r| \le 2\times10^{-8}\max|f|$;
unpreconditioned CG takes 340–760.  The full table is in the
[Solver Comparison](solver_comparison.md).

#### Spectral Preconditioner

Applies the rectangular spectral solver (DST/DCT/FFT) as an approximate
inverse: $M^{-1} r = L_{\text{rect}}^{-1} r$.

**Pros:**

- Essentially free — one forward + one inverse transform per iteration
- No setup cost
- Exact inverse only when the operator *is* the transform's: it works on
  the whole array with its wall at the array edge, so even on the ψ basin
  of a rectangle (wall nodes `0` and `N-2`) CG still needs 15 (DST) to
  63 (FFT) iterations

**Cons:**

- Ignores the mask: the rectangular solve doesn't know about land cells,
  so the preconditioner becomes less effective as the domain deviates from
  a rectangle
- Cannot handle variable coefficients $c(x,y)$ — it always assumes $c = 1$
- Effectiveness degrades for highly irregular coastlines

**Best for:** a quick, setup-free solve.  It is the wrappers' default
preconditioner; CG + multigrid needs 3–13× fewer iterations.

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
- Grid-independent iteration counts: CG + multigrid takes 5–7 iterations at
  128² and 256² (unpreconditioned CG: 340–760)
- The V-cycle is symmetric (`n_pre == n_post`), as CG requires

**Cons:**

- Requires an offline build step (`build_multigrid_solver`, NumPy /
  SciPy) that precomputes the level hierarchy; rebuild it when the mask,
  coefficient or $\lambda$ changes
- More complex implementation

**Best for:** masked domains (it is the fastest CG measured) and
variable-coefficient problems.  Any grid shape coarsens (no divisibility
constraint).

#### Decision Guide

```
Is c(x,y) spatially varying?
├── Yes → Multigrid preconditioner
│         (spectral and Nyström can't represent variable coefficients well)
└── No (constant coefficient) ↓

Can you build the multigrid hierarchy (the operator is the 5-point
masked Helmholtz operator)?
├── Yes → Multigrid preconditioner (5–7 iterations, fastest CG)
└── No ↓

Is the operator close to the rectangular constant-coefficient one?
├── Yes → Spectral preconditioner (no setup, 15–63 iterations)
└── Operator-only access → Nyström preconditioner (rank 30–100)
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

Fixed mask and λ, many solves, setup affordable (N_b up to ~1000)?
├── Yes → Use capacitance method (method="capacitance")
│         Pre-build with build_capacitance_solver()
└── No (mask or λ changes, very long coastline) ↓

Default → Use CG (method="cg")
          with preconditioner=make_multigrid_preconditioner(mg)
          (the default spectral preconditioner needs no setup but
          3–13× more iterations)
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
