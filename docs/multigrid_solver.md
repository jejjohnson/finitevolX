# Multigrid Helmholtz Solver

This page covers the theory behind the geometric multigrid solver in
finitevolX, its variable-coefficient Helmholtz operator, and the three
differentiation strategies for computing gradients through the solve.

---

## The Problem

The spectral and capacitance-matrix solvers in finitevolX handle
**constant-coefficient** Helmholtz equations efficiently.  However, many
ocean modelling tasks require solving the **variable-coefficient**
generalisation:

$$
\nabla \cdot \bigl(c(x,y)\,\nabla u\bigr) - \lambda\,u = f
$$

where $c(x,y)$ is a spatially varying coefficient (e.g. layer thickness,
diffusivity, or Rossby-radius field) and the domain may be irregular
(masked).

| Application | Equation | $c(x,y)$ | $\lambda$ |
|---|---|---|---|
| **Variable-thickness QG** | $\nabla \cdot (H\,\nabla \psi) - \psi/R_d^2 = q$ | Layer thickness $H(x,y)$ | $1/R_d^2$ |
| **Spatially varying diffusion** | $\nabla \cdot (\kappa\,\nabla T) = f$ | Diffusivity $\kappa(x,y)$ | 0 |
| **Topographic PV inversion** | $\nabla \cdot (f_0^2/N^2 \cdot \nabla \psi) - \beta y\,\psi = q$ | Stratification-dependent | Varies |

Spectral methods cannot handle spatially varying $c(x,y)$ because the
eigenvalue decomposition assumes constant coefficients.  CG can handle it,
but converges slowly without a good preconditioner.  **Geometric multigrid**
is the standard approach: it provides both a fast standalone solver and an
excellent preconditioner for CG.

---

## Geometric Multigrid: Overview

Multigrid accelerates iterative solvers by exploiting a hierarchy of
progressively coarser grids.  The key insight: simple smoothers (e.g.
Jacobi) efficiently damp **high-frequency** error on any grid, but leave
**low-frequency** error untouched.  By restricting the problem to a coarser
grid, those low-frequency components become high-frequency — and can be
damped cheaply.

### The V-Cycle

The fundamental building block is the **V-cycle**, a recursive algorithm
that visits each level of the grid hierarchy:

```
Level 0 (finest)    ●───smooth───●───────────────────────●───smooth───●
                         ↓ restrict                   prolong ↑
Level 1             ·····●───smooth───●───────●───smooth───●·····
                              ↓ restrict   prolong ↑
Level 2 (coarsest)  ··········●───exact solve───●··········
```

**Algorithm** for `v_cycle(u, rhs, level)`:

1. If at the coarsest level: solve exactly (bottom solve).
2. **Pre-smooth**: apply $\nu_1$ weighted Jacobi iterations.
3. **Residual**: compute $r = f - A u$.
4. **Restrict**: transfer $r$ to the coarse grid, $r_c = R\,r$.
5. **Recurse**: solve $A_c\,e_c = r_c$ on the coarse grid (V-cycle), with
   the Galerkin operator $A_c = R\,A\,P$.
6. **Prolongate**: interpolate $e_c$ back to the fine grid.
7. **Correct**: $u \leftarrow u + e_{\text{fine}}$.
8. **Post-smooth**: apply $\nu_2$ weighted Jacobi iterations.

Multiple V-cycles are applied to drive the residual to convergence.

---

## Discrete Operator

### Variable-Coefficient Helmholtz Stencil

On the finest level the operator is discretised using a 5-point
finite-volume stencil.  Face coefficients $c_x$ and $c_y$ live on the
staggered faces between cell centres:

$$
(Au)_{j,i} = \frac{c_{x,j,i}\,(u_{j,i+1} - u_{j,i}) - c_{x,j,i-1}\,(u_{j,i} - u_{j,i-1})}{\Delta x^2}
+ \frac{c_{y,j,i}\,(u_{j+1,i} - u_{j,i}) - c_{y,j-1,i}\,(u_{j,i} - u_{j-1,i})}{\Delta y^2}
- \lambda\,u_{j,i}
$$

Face coefficients are computed by averaging the cell-centre coefficient
$c(x,y)$ to the faces:

$$
c_{x,j,i} = \tfrac{1}{2}\bigl(c_{j,i} + c_{j,i+1}\bigr) \cdot m_{x,j,i}
$$

where $m_{x,j,i} = 1$ only when both cells $(j,i)$ and $(j,i+1)$ are
wet (inside the mask).  Faces between a wet and a dry cell are set by the
boundary condition below.

### Boundary Conditions

`build_multigrid_solver(..., bc=...)` sets the condition on every face
between a wet cell and a dry one (land, or outside the array), on every
level:

| `bc` | Wet–dry face | Use for |
|---|---|---|
| `"dirichlet"` (default) | keeps its coefficient; the dry neighbour's value is $0$ | the streamfunction $\psi$ at X points (QG PV inversion), $\psi = 0$ on the coast |
| `"neumann"` | zero coefficient: no flux | pressure / free surface at T points (pass `mask.h`) |

With `bc="dirichlet"` and `coeff=None` the finest-level operator is exactly
`masked_laplacian` with dry neighbours zero: for $\psi$ on
`Mask2D.xy_corner_strict` the wall sits on the dry X nodes next to the wet
ones (index $0$ and $N-2$ on a rectangle built with `from_interior`). The
wet–dry faces only add to the diagonal,

$$
(Au)_{j,i} \mathrel{-}= \Bigl(\sum_{\text{wet–dry faces}} \frac{c_{j,i}}{h^2}\Bigr)\, u_{j,i},
$$

with $h = \Delta x$ or $\Delta y$ and $c_{j,i}$ the wet cell's coefficient.
The operator is negative definite for every $\lambda \ge 0$.

With `bc="neumann"` the operator annihilates a constant on each connected
wet component, so at $\lambda = 0$ it is singular. The solver then works in
the zero-mean gauge: it projects the right-hand side onto its solvable
(zero-mean) part and returns the zero-mean solution on each component.

Out-of-bounds neighbours are zero-padded, so nothing wraps periodically
across opposite edges.

---

## Components

### Weighted Jacobi Smoother

The smoother updates the solution pointwise:

$$
u^{(k+1)} = u^{(k)} + \omega\,D^{-1}\bigl(f - A\,u^{(k)}\bigr)
$$

where $D = \text{diag}(A)$ is the diagonal of the operator and
$\omega \in (0, 1)$ is the relaxation weight.  The diagonal is
precomputed during the offline build phase.  The default $\omega = 0.8$
damps every mode in the upper half of the 5-point spectrum by at least
$3/5$ per sweep; $\omega$ near 1 barely damps the checkerboard mode
($|1 - 2\omega|$).

!!! note "Why Jacobi, not Gauss-Seidel?"
    Red-black Gauss-Seidel is the traditional multigrid smoother, but it
    requires sequential updates that are difficult to express efficiently
    in JAX.  Weighted Jacobi is fully parallel, composes naturally with
    `jax.lax.fori_loop`, and is symmetric, which keeps the V-cycle
    symmetric.

### Coarsening

A dimension of $n$ points coarsens to $\lceil n/2 \rceil = (n+1) /\!/ 2$,
with coarse point $J$ on fine point $2J$.  Odd sizes coarsen like even
ones, so a standard $(N+2)$ grid (interior plus ghost ring) is as deep as
$N$: $34 \to 17 \to 9 \to 5$, $130 \to 65 \to \dots \to 5$ (6 levels),
$258 \to \dots \to 5$ (7 levels).  `n_levels=None` coarsens while both
dimensions stay $\ge 4$.

**Mask-coarsening rule.** A coarse point is wet if its coincident fine
point $(2J, 2I)$ is wet.  A coarse point whose coincident fine point is dry
but whose interpolation support (fine points $2J-1, 2J, 2J+1$ in each
direction) touches a wet fine point is added too, unless its prolongation
column is a linear combination of the other added columns (checked with a
column-pivoted QR per cluster at build time).  So coasts get an extra layer
of coarse unknowns and narrow channels keep one, while a one-cell lake or
channel at an odd index — up to four coarse points with *identical*
columns — keeps exactly one: $P$ has full column rank, and $A_c$ is
non-singular (Dirichlet) or singular only by the constants (Neumann,
$\lambda = 0$).  The coarse operator sees the actual fine geometry through
the Galerkin product below, not through the coarse mask.

### Prolongation (Coarse to Fine)

Bilinear interpolation between the nested points:

$$
v^f_{2J,2I} = v^c_{J,I}, \qquad
v^f_{2J+1,2I} = \tfrac12\bigl(v^c_{J,I} + v^c_{J+1,I}\bigr), \qquad
v^f_{2J+1,2I+1} = \tfrac14 \textstyle\sum_{\text{4 neighbours}} v^c
$$

with dry coarse values taken as zero, then scaled row-wise:
$P = \mathrm{diag}(s)\,P_0\,\mathrm{diag}(m^c)$, with $s = m^f$ for
`bc="dirichlet"` (the correction tapers to the zero wall value) and
$s = m^f / (P_0\,m^c)$ for `bc="neumann"` (renormalised over wet coarse
points, so constants are interpolated exactly).

### Restriction (Fine to Coarse)

Restriction is the scaled transpose of prolongation, $R = \tfrac14 P^{\mathsf T}$
— away from walls the 9-point full-weighting average with weights
$(4, 2, 2, 1)/16$.  Being the transpose makes $\langle R r, e\rangle =
\tfrac14\langle r, P e\rangle$ exactly, which the symmetry of the V-cycle
relies on.

### Coarse Operators

Every coarse operator is the Galerkin product $A_c = R\,A\,P$, a 9-point
stencil computed once at build time (SciPy sparse products).  It inherits
the fine wall condition, coefficient and geometry, and it keeps the
operator symmetric and negative (semi)definite on every level.

### Bottom Solver

The coarsest level (at most a few dozen unknowns on a square grid) is
solved exactly: its inverse on the wet points is precomputed at build time
from a symmetric eigendecomposition, dropping eigenvalues below
$10^{-10}\max|\mu|$.  That is the exact inverse of a non-singular
operator and the pseudo-inverse of the singular Neumann $\lambda = 0$
operator (returning the zero-mean correction) without the $\sim 10^{13}$
entries its round-off-sized null eigenvalues would otherwise give.

### Symmetry

With $R = \tfrac14 P^{\mathsf T}$, Galerkin coarse operators, an exact
bottom solve and as many pre- as post-smoothing sweeps (`n_pre == n_post`,
the default), one V-cycle from a zero guess, $r \mapsto
\texttt{v\_cycle}(0, r)$, is a symmetric linear operator: $\langle M r_1,
r_2\rangle = \langle r_1, M r_2\rangle$ to round-off.  That is what CG needs
from a preconditioner; `make_multigrid_preconditioner` refuses a solver with
`n_pre != n_post`.

---

## Grid Hierarchy Construction

The `build_multigrid_solver` factory performs offline precomputation
(using NumPy / SciPy) to build the level hierarchy:

1. **Finest operator**: the 5-point stencil from $c(x,y)$, the mask and
   `bc`.
2. **Transfers**: coarse mask and $P$ per level (rules above).
3. **Galerkin coarse operators** $A_c = R A P$ and their diagonals
   $D^{-1}$ for the smoother.
4. **Bottom inverse** on the coarsest wet points.

The resulting `MultigridSolver` is an immutable `equinox.Module` —
all arrays are frozen JAX arrays, and all integer parameters
(`n_levels`, `n_pre`, etc.) are static fields for efficient JIT
compilation.  Any grid shape works with any `n_levels` $\ge 1$
(`n_levels=1` is a direct dense solve, allowed up to 4096 wet points).

---

## Differentiating Through the Solve

A key advantage of implementing multigrid in JAX is that we can compute
gradients of a loss function through the linear solve:

$$
\mathcal{L}(\theta) = \ell\bigl(A(\theta)^{-1} f(\theta)\bigr)
$$

where $\theta$ parameterises the coefficient field $c(x,y)$, the RHS
$f$, or both.  finitevolX provides **three differentiation strategies**
with different cost/accuracy trade-offs.

### Strategy 1: Implicit Differentiation (Default)

The default `__call__` method uses `jax.lax.custom_linear_solve` to
compute gradients via the **implicit function theorem** (IFT).

Given the linear system $A u = f$, the gradient of a scalar loss
$\ell(u)$ with respect to the RHS is:

$$
\frac{\partial \ell}{\partial f} = A^{-T} \frac{\partial \ell}{\partial u}
$$

Since $A$ is symmetric ($A = A^T$), the adjoint solve is just another
multigrid call with the same operator.

**Properties:**

- Forward: $K$ V-cycles (identical to unrolled)
- Backward: 1 multigrid solve (the adjoint equation)
- Memory: $O(1)$ — no iteration history stored
- Gradients: **exact** (up to solver tolerance)

!!! tip "When to use"
    This is the default and recommended mode.  Use it whenever you need
    gradients through the solve (e.g. learning $c(x,y)$ or $\lambda$
    from data).

### Strategy 2: One-Step Differentiation

The `solve_onestep` method implements the approach of
Bolte, Pauwels & Vaiter (NeurIPS 2023).  It runs $K$ V-cycles
for convergence, but only differentiates through the **last** cycle:

$$
u_K = V\bigl(\underbrace{V(\cdots V(0, f) \cdots, f)}_{\text{stop\_gradient after } K{-}1 \text{ cycles}},\; f\bigr)
$$

The gradient approximation error is $O(\rho)$ where $\rho$ is the
V-cycle convergence rate ($\approx 0.07$ on the default settings; on a
$34^2$ basin the one-step gradient is within 4 % of the implicit one).

**Properties:**

- Forward: $K$ V-cycles (identical)
- Backward: 1 V-cycle (autodiff through only the last)
- Memory: $O(1\text{ V-cycle})$
- Gradients: **approximate**, error $O(\rho)$

!!! note "When to use"
    When you need cheap gradients and can tolerate small approximation
    error — e.g. in training loops where the solver is called many times
    and the gradient noise from one-step differentiation is small relative
    to stochastic gradient noise.

### Strategy 3: Unrolled Differentiation

The `solve_unrolled` method differentiates through every V-cycle
iteration via `jax.lax.fori_loop`:

**Properties:**

- Forward: $K$ V-cycles
- Backward: $K$ V-cycles (replay all iterations)
- Memory: $O(K)$ — stores intermediate states for backprop
- Gradients: **exact** through the iteration process

!!! note "When to use"
    When you specifically need gradients through the iteration dynamics
    itself (e.g. analysing convergence behaviour, or when the number of
    iterations is very small).

### Comparison

| Mode | Backward cost | Memory | Gradient quality |
|---|---|---|---|
| **Implicit** (`__call__`) | 1 multigrid solve | $O(1)$ | Exact (IFT) |
| **One-step** (`solve_onestep`) | 1 V-cycle autodiff | $O(1)$ | Approximate, $O(\rho)$ error |
| **Unrolled** (`solve_unrolled`) | $K$ V-cycles | $O(K)$ | Exact (through iterations) |

---

## Multigrid as a Preconditioner

A single V-cycle is an excellent preconditioner for CG.  This is useful
when:

- You want the convergence guarantees of CG (Krylov method) but need
  faster convergence than a spectral preconditioner provides.
- The variable-coefficient problem is poorly conditioned.
- You want to combine multigrid's coarse-grid correction with CG's
  global optimality.

`make_multigrid_preconditioner` wraps one V-cycle as a closure compatible
with `solve_cg`; the V-cycle is symmetric (see above), as CG requires.  On a
circular basin with $\lambda = 4$ and `rtol=1e-8`, CG then takes 6
iterations at both $128^2$ and $256^2$, against 33 and 47 with the FFT
spectral preconditioner.

---

## Convergence Theory

Multigrid with a good smoother and an accurate coarse-grid correction
reduces the residual by a fixed factor $\rho$ per V-cycle, independent of
grid size.  Measured per-cycle residual reduction (default V(3,3),
$\omega = 0.8$, circular basin, $\lambda = 4$, Dirichlet, cycles 5–8):

| Grid | Levels | $\rho$ per cycle |
|---|---|---|
| $34^2$ | 4 | 0.064–0.066 |
| $130^2$ | 6 | 0.066–0.069 |
| $258^2$ | 7 | 0.068–0.073 |

(The first cycle reduces it by about 0.02.)  This means:

- 5 V-cycles reduce the residual by about $3 \times 10^{-7}$
- The cost per V-cycle is $O(N)$ where $N = N_y \times N_x$ (each
  level costs a quarter of the previous, geometric series)
- **Total cost**: $O(N)$ — optimal for elliptic solves

Measured on CPU (float64, jitted, circular basin, $\lambda = 4$): 5 V-cycles
take 2.7 ms at $128^2$ and 17 ms at $256^2$; CG with the multigrid
preconditioner (`rtol=1e-8`) 6.6 ms / 6 iterations and 40 ms / 6
iterations, against 27 ms / 33 and 220 ms / 47 for CG with the FFT
preconditioner.

Narrow features ($34^2$, cycles 5–8; an island, and two basins joined by a
one-cell channel in an odd or an even column):

| Mask | Dirichlet $\lambda=0$ / $4$ | Neumann $\lambda=0$ / $4$ | CG + MG iterations |
|---|---|---|---|
| island | 0.080 / 0.059 | 0.148 / 0.095 | 5–6 |
| odd-column channel | 0.088 / 0.053 | 0.475 / 0.232 | 5–7 |
| even-column channel | 0.090 / 0.053 | 0.226 / 0.089 | 5–6 |

Dirichlet stays near the basin rate.  Neumann is slower where a thin
channel carries the flux between basins (the near-null "sloshing" mode is
poorly represented by bilinear coarse functions), most at $\lambda = 0$;
the V-cycle remains a good CG preconditioner there (5–7 iterations at
`rtol=1e-8` on all of these masks), which is the recommended use on such
coasts.

For variable coefficients the Galerkin coarse operators follow the
coefficient, so the rate stays close to these values for moderate
coefficient contrast; strongly anisotropic or high-contrast coefficients
degrade point-Jacobi smoothing.

---

## Decision Guide

```
Is c(x,y) constant (or nearly so)?
├── Yes → Use spectral solver (fastest, O(N log N))
│         Or capacitance method for masked domains
└── No (variable coefficient) ↓

Is the domain rectangular (no mask)?
├── Yes → Multigrid standalone (method="multigrid")
│         build_multigrid_solver(np.ones(...), dx, dy, coeff=c)
└── No (masked/irregular domain) ↓

Is the domain simple with small N_b?
├── Yes → Multigrid standalone (handles masks natively)
└── No (complex domain, poor convergence) ↓

Default → Multigrid-preconditioned CG
          make_multigrid_preconditioner() + solve_cg()
```

---

## References

- Briggs, Henson & McCormick (2000) — *A Multigrid Tutorial*, 2nd ed.
  (standard reference for geometric multigrid)
- Trottenberg, Oosterlee & Schuller (2001) — *Multigrid* (comprehensive
  treatment including variable coefficients)
- Bolte, Pauwels & Vaiter (NeurIPS 2023) — One-step differentiation of
  iterative algorithms
- Blondel et al. (ICML 2022) — Efficient and modular implicit
  differentiation
- Louity — `qgsw-pytorch` (reference PyTorch multigrid Helmholtz
  implementation)
