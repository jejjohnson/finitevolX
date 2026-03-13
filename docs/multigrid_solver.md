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
Level 2 (coarsest)  ··········●───bottom solve───●··········
```

**Algorithm** for `v_cycle(u, rhs, level)`:

1. If at the coarsest level: run many Jacobi iterations (bottom solve).
2. **Pre-smooth**: apply $\nu_1$ weighted Jacobi iterations.
3. **Residual**: compute $r = f - A u$.
4. **Restrict**: transfer $r$ to the coarse grid.
5. **Recurse**: solve $A_c\,e_c = r_c$ on the coarse grid (V-cycle).
6. **Prolongate**: interpolate $e_c$ back to the fine grid.
7. **Correct**: $u \leftarrow u + e_{\text{fine}}$.
8. **Post-smooth**: apply $\nu_2$ weighted Jacobi iterations.

Multiple V-cycles are applied to drive the residual to convergence.

---

## Discrete Operator

### Variable-Coefficient Helmholtz Stencil

The operator is discretised using a 5-point finite-volume stencil on a
cell-centred grid.  Face coefficients $c_x$ and $c_y$ live on the
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
wet (inside the mask).  This ensures the operator respects irregular domain
boundaries.

### Boundary Conditions

Zero normal flux at domain edges is enforced by the face coefficients:
boundary faces have zero coefficient, so no flux crosses the domain
boundary.  Out-of-bounds neighbours are zero-padded as an implementation
convenience (avoids periodic wrapping from `jnp.roll`), but the actual
BC is determined by the zeroed face coefficients, **not** the ghost
values.  This means **no periodic wrapping** at the domain edges — the
natural choice for bounded ocean basins.

---

## Components

### Weighted Jacobi Smoother

The smoother updates the solution pointwise:

$$
u^{(k+1)} = u^{(k)} + \omega\,D^{-1}\bigl(f - A\,u^{(k)}\bigr)
$$

where $D = \text{diag}(A)$ is the diagonal of the operator and
$\omega \in (0, 1)$ is the relaxation weight.  The diagonal is
precomputed during the offline build phase.

!!! note "Why Jacobi, not Gauss-Seidel?"
    Red-black Gauss-Seidel is the traditional multigrid smoother, but it
    requires sequential updates that are difficult to express efficiently
    in JAX.  Weighted Jacobi is fully parallel, composes naturally with
    `jax.lax.fori_loop`, and converges well with $\omega \approx 0.8\text{--}0.95$.

### Restriction (Fine to Coarse)

Cell-centred full-weighting restriction averages each $2 \times 2$ block of
fine cells into one coarse cell, weighted by the mask:

$$
v^c_{J,I} = \frac{\sum_{(j,i) \in \text{block}} m^f_{j,i}\,v^f_{j,i}}
                  {\max\bigl(\sum_{(j,i) \in \text{block}} m^f_{j,i},\; 1\bigr)}
$$

This mask-weighted divisor prevents land cells from contaminating the
coarse-grid values.

### Prolongation (Coarse to Fine)

Bilinear prolongation maps each coarse cell to four fine sub-cells using
9/3/3/1 weights:

$$
v^f_{2J,2I} = \frac{9\,v^c_{J,I} + 3\,v^c_{J,I-1} + 3\,v^c_{J-1,I} + v^c_{J-1,I-1}}
                   {9\,m^c_{J,I} + 3\,m^c_{J,I-1} + 3\,m^c_{J-1,I} + m^c_{J-1,I-1}}
$$

The three other sub-cells use analogous stencils with shifted neighbours.
The mask-weighted divisor again prevents land contamination.

### Bottom Solver

At the coarsest level, the grid is small enough (typically
$8 \times 8$ to $16 \times 16$) that iterated Jacobi converges
quickly.  This avoids the complexity of a dense direct solve while
keeping the implementation pure-JAX.

---

## Grid Hierarchy Construction

The `build_multigrid_solver` factory performs offline precomputation
(using NumPy) to build the level hierarchy:

1. **Mask coarsening**: each level's mask is the 4-point average of the
   fine mask, thresholded at 0.5.
2. **Coefficient interpolation**: cell-centre $c(x,y)$ is averaged to
   face coefficients at each level, then coarsened for the next level.
3. **Diagonal precomputation**: $D^{-1}$ is computed from the face
   coefficients and mask at each level.
4. **Grid spacing doubling**: $\Delta x$ and $\Delta y$ double at
   each coarser level.

The resulting `MultigridSolver` is an immutable `equinox.Module` —
all arrays are frozen JAX arrays, and all integer parameters
(`n_levels`, `n_pre`, etc.) are static fields for efficient JIT
compilation.

!!! warning "Grid size constraint"
    Both grid dimensions must be divisible by $2^{L-1}$ where $L$ is
    the number of multigrid levels.  The factory auto-detects $L$ by
    halving until either dimension would drop below 8.

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
V-cycle convergence rate (typically 0.1--0.3 for multigrid).

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
with `solve_cg`.  The CG iteration then converges in very few steps
(often 5--10 instead of hundreds).

---

## Convergence Theory

For a constant-coefficient Poisson/Helmholtz problem on a rectangular
domain, multigrid with weighted Jacobi smoothing achieves a convergence
rate $\rho \approx 0.1\text{--}0.3$ per V-cycle, independent of grid
size.  This means:

- 5 V-cycles reduce the residual by a factor of $\rho^5 \approx 10^{-3}\text{--}10^{-5}$
- The cost per V-cycle is $O(N)$ where $N = N_y \times N_x$ (each
  level costs half the previous, geometric series)
- **Total cost**: $O(N)$ — optimal for elliptic solves

For variable coefficients and masked domains, the convergence rate
depends on the coefficient contrast and domain geometry, but multigrid
typically remains much faster than unpreconditioned CG.

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
