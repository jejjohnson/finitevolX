# Advection Operators

This page covers the theory behind the advection operators in finitevolX,
explains the flux-form discretisation and the available reconstruction schemes
(upwind, TVD, WENO), and provides guidance for choosing a scheme.

---

## The Problem

Advection moves scalar and vector fields along the flow.  In the
**flux form** (also called the conservative or divergence form), the
transport of a scalar $q$ by a velocity $\mathbf{u}$ is written

$$
\frac{\partial q}{\partial t} + \nabla \cdot (\mathbf{u}\,q) = 0
$$

Using $\nabla \cdot (\mathbf{u}\,q) = \mathbf{u} \cdot \nabla q + q\,\nabla \cdot \mathbf{u}$,
the advective form and flux form differ by a term proportional to $\nabla \cdot \mathbf{u}$.
Flux form is preferred because it:

1. **Conserves mass exactly** in the discrete setting (telescoping property).
2. Handles non-divergent flow correctly without an extra correction term.
3. Is the natural form for finite-volume methods.

On the Arakawa C-grid, the scalar $q$ lives at **T-points** and the velocities
$u$, $v$ live at **U/V-points** (face centres).  The tendency is computed at
T-points and written to `[1:-1, 1:-1]` (or `[2:-2, 2:-2]` for 2-D to avoid
reading ghost flux cells).

---

## Flux-Form Algorithm

### 2-D Case

The discrete tendency at interior T-point $(j, i)$ is

$$
\left.\frac{\partial q}{\partial t}\right|_{j,i}
= -\frac{f_{j,i+\frac{1}{2}}^x - f_{j,i-\frac{1}{2}}^x}{\Delta x}
  -\frac{f_{j+\frac{1}{2},i}^y - f_{j-\frac{1}{2},i}^y}{\Delta y}
$$

where the face-normal fluxes are

$$
f_{j,i+\frac{1}{2}}^x = u_{j,i+\frac{1}{2}} \cdot \hat{q}_{j,i+\frac{1}{2}}, \qquad
f_{j+\frac{1}{2},i}^y = v_{j+\frac{1}{2},i} \cdot \hat{q}_{j+\frac{1}{2},i}
$$

and $\hat{q}$ is the **reconstructed face value** obtained from the cell-centre
values by one of the schemes described below.

### Upwind Flux Dispatch

Face values are computed by `upwind_flux(q_left, q_right, u_face)`.  The
convention is:

- If $u_{\text{face}} \geq 0$: $\hat{q} = q_{\text{left}}$ (upwind is to the west/south).
- If $u_{\text{face}} < 0$: $\hat{q} = q_{\text{right}}$ (upwind is to the east/north).

All higher-order methods ultimately call this dispatcher after computing
improved left and right state estimates.

---

## Reconstruction Schemes

The key design choice is how to estimate the face value $\hat{q}$ from
cell-centre values.  The table below summarises the available methods:

| Method | Order | Monotone | Stencil width | Best for |
|--------|-------|----------|---------------|----------|
| **Upwind 1st-order** | 1 | Yes | 2 pts | Boundary fallback, extreme cases |
| **TVD/Minmod** | 2 | Yes | 4 pts | Simple monotone advection |
| **TVD/van Leer** | 2 | Yes | 4 pts | Good balance: smooth + sharp |
| **TVD/Superbee** | 2 | Yes | 4 pts | Sharpest TVD limiter |
| **TVD/MC** | 2 | Yes | 4 pts | Less dissipative than minmod |
| **WENO3** | 3 | Yes (ENO) | 4 pts | Sharp discontinuities, moderate cost |
| **WENO5** | 5 | Yes (ENO) | 6 pts | **Default** — smooth + sharp |
| **WENOz5** | 5 | Yes (ENO) | 6 pts | Less dissipative than WENO5 |
| **WENO7** | 7 | Yes (ENO) | 8 pts | Very smooth fields |
| **WENO9** | 9 | Yes (ENO) | 10 pts | High-accuracy reference |

### Total Variation Diminishing (TVD)

TVD schemes add a **flux limiter** $\phi(r)$ to a first-order upwind base:

$$
\hat{q}_{i+\frac{1}{2}} = q_i + \tfrac{1}{2}\,\phi(r)\,(q_i - q_{i-1}),
\qquad r = \frac{q_{i+1} - q_i}{q_i - q_{i-1}}
$$

The limiter $\phi(r)$ guarantees TVD stability: the total variation
$\sum_i |q_{i+1} - q_i|$ cannot increase.

| Limiter | $\phi(r)$ | Character |
|---------|-----------|-----------|
| **Minmod** | $\max(0,\min(1,r))$ | Most diffusive; never overshoots |
| **van Leer** | $(r + |r|)/(1 + |r|)$ | Smooth; good for monotone scalars |
| **Superbee** | $\max(0, \max(\min(2r, 1), \min(r, 2)))$ | Sharpest TVD; may create false plateaus |
| **MC** | $\max(0, \min((1+r)/2, 2, 2r))$ | Between minmod and superbee |

!!! tip "When to use TVD"
    TVD schemes are ideal when you need guaranteed monotonicity (e.g., tracer
    concentrations that must stay non-negative) and the field has sharp gradients.
    For smooth fields WENO schemes give higher accuracy at comparable cost.

### Weighted Essentially Non-Oscillatory (WENO)

WENO schemes achieve high-order accuracy in smooth regions while avoiding
spurious oscillations near discontinuities.  The key idea is to compute
several **candidate reconstructions** from different sub-stencils and
blend them with **nonlinear weights** that de-emphasise sub-stencils
containing large gradients.

#### WENO3 (3rd order)

Two 2-point candidate stencils blended with nonlinear weights $\omega_0,
\omega_1$:

$$
q_0 = \tfrac{3}{2}q_i - \tfrac{1}{2}q_{i-1}, \qquad
q_1 = \tfrac{1}{2}q_i + \tfrac{1}{2}q_{i+1}
$$

$$
\hat{q}_{i+\frac{1}{2}} = \omega_0\,q_0 + \omega_1\,q_1
$$

Weights are derived from the smoothness indicators $\beta_k$ (variance
measures of the sub-stencil).  Near discontinuities $\omega_k \to 0$ for
stencils crossing the discontinuity, automatically reducing to 1st-order.

#### WENO5 (5th order)

Three 3-point candidate stencils with optimal weights $(d_0, d_1, d_2)$:

$$
\hat{q}_{i+\frac{1}{2}} = \omega_0\,q_0 + \omega_1\,q_1 + \omega_2\,q_2
$$

Achieves 5th-order accuracy in smooth regions.  This is the **recommended
default** for most ocean PDE work.

#### WENOz5 (improved WENO5)

Borges et al. (2008) modification of WENO5 that uses a higher-order
global smoothness indicator $\tau_5 = |\beta_0 - \beta_2|$ to scale the
weights, achieving less dissipation near smooth extrema while remaining
ENO near discontinuities.

!!! note "WENO vs TVD accuracy"
    On smooth fields: WENO5 ≈ 5th-order error vs TVD ≈ 2nd-order.  On fields
    with shocks or fronts: both reduce to 1st-order near the discontinuity,
    but WENO maintains high order elsewhere while TVD is 2nd-order at best.

---

## Adaptive Stencil Selection (Mask-Aware)

Near irregular boundaries (islands, coastlines), the standard WENO stencils
extend into land cells and produce incorrect results.  finitevolX supports
**mask-aware adaptive stencil selection** via `Mask2D`:

- Each T-point stores a `StencilCapability2D` value indicating the maximum
  symmetric stencil width supported in each direction (2, 4, or 6 points).
- The advection operator queries these masks and selects the appropriate
  sub-scheme (upwind, WENO3, or WENO5) at each cell.
- Fully ocean cells use the full-order scheme; cells near land fall back
  gracefully to lower-order methods.

!!! tip "When to enable mask-aware advection"
    Pass an `Mask2D` to `Advection2D.__init__` whenever the domain
    has land cells.  For fully periodic or rectangular domains without land,
    omit the mask for slightly lower overhead.

---

## Write Region

The 2-D and 3-D advection operators write the tendency to
**`[2:-2, 2:-2]`** (not `[1:-1, 1:-1]`) to avoid reading ghost flux cells
at the domain boundary.  This means two rows/columns of T-points adjacent
to the boundary have zero tendency from the advection operator alone; the
caller must account for this when applying boundary conditions.

The 1-D operator writes to `[2:-2]`.

---

## Quick Usage

```python
import jax.numpy as jnp
from finitevolx import ArakawaCGrid2D, Advection2D

grid = ArakawaCGrid2D.from_interior(64, 64, 1e6, 1e6)

# Default: WENO5 without mask
adv = Advection2D(grid=grid)

# With TVD/van Leer
adv_tvd = Advection2D(grid=grid, method="van_leer")

# With mask-aware WENO5
from finitevolx import Mask2D
mask = Mask2D.from_mask(ocean_mask)
adv_masked = Advection2D(grid=grid, method="weno5", mask=mask)

# Compute tracer tendency: -∇·(u*q)
q = jnp.ones((grid.Ny, grid.Nx))  # T-point tracer
u = jnp.zeros((grid.Ny, grid.Nx)) # U-point velocity
v = jnp.zeros((grid.Ny, grid.Nx)) # V-point velocity

dq_dt = adv(q, u, v)   # shape [Ny, Nx], non-zero only in [2:-2, 2:-2]
```

---

## Decision Guide

```
Do you need guaranteed monotonicity (e.g. positive-definite tracers)?
├── Yes → Use TVD limiter
│         ├── Need sharpest fronts?       → superbee
│         ├── Best smooth + monotone?     → van_leer (recommended)
│         └── Most conservative?         → minmod
└── No ↓

Is accuracy on smooth fields important?
├── Yes → Use WENO
│         ├── Standard, well-tested      → weno5 (recommended default)
│         ├── Less dissipative           → wenoz5
│         ├── Moderate cost, 3rd-order   → weno3
│         └── Reference solution         → weno7 / weno9
└── No → Use upwind1 (1st order, maximum dissipation)

Does the domain have land/islands?
├── Yes → pass Mask2D to Advection2D
└── No  → omit mask (slightly faster)
```

---

## References

- Harten, Lax & van Leer (1983) — Upwind schemes and TVD conditions
- van Leer (1979) — MUSCL flux limiters
- Liu, Osher & Chan (1994) — Weighted Essentially Non-Oscillatory schemes
- Jiang & Shu (1996) — WENO5 scheme
- Borges et al. (2008) — WENOz improved WENO scheme
- Durran (2010) — *Numerical Methods for Fluid Dynamics*, Ch. 5
