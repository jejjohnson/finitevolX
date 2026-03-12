# Time Integration for Ocean Models

This page covers the theory behind the time integration schemes in finitevolX,
explains why certain methods are preferred for ocean and atmosphere modelling,
and provides practical guidance for choosing a scheme.

---

## The Problem

Ocean models advance state variables (velocity, free surface, tracers) by
solving systems of the form

$$
\frac{\partial \mathbf{y}}{\partial t} = \mathbf{F}(\mathbf{y}, t)
$$

where $\mathbf{F}$ contains spatial operators (advection, pressure gradient,
diffusion, Coriolis, etc.) already discretised on the Arakawa C-grid.

The choice of time-stepping method controls **stability**, **accuracy**, and
**efficiency**.  Ocean PDEs have specific characteristics that make some schemes
much more appropriate than others.

---

## Key Constraints in Ocean PDEs

| Constraint | Physical origin | Consequence for time stepping |
|---|---|---|
| **CFL condition** | Fast gravity/barotropic waves | Limits $\Delta t$ unless split or implicit |
| **Multiple timescales** | Barotropic ($c \sim 200$ m/s) vs baroclinic ($c \sim 2$ m/s) | Split-explicit or IMEX preferred |
| **Monotonicity / positivity** | Tracer concentrations must stay non-negative | SSP methods preserve these properties |
| **Long integrations** | Climate runs: $10^6$–$10^8$ steps | Cost per step matters; 1 RHS eval/step is valuable |
| **Stiff vertical diffusion** | Thin surface layers with strong mixing | Implicit treatment avoids tiny $\Delta t$ |

---

## Method Families

### Explicit Runge-Kutta

The workhorses of ocean modelling.  Each step evaluates the RHS $s$ times
(one per *stage*) and combines them to achieve order $p$.

| Method | Order | Stages | SSP | Best for |
|---|---|---|---|---|
| **Forward Euler** | 1 | 1 | C=1 | Debugging, sub-stepping inner loops |
| **Heun / RK2** | 2 | 2 | C=1 | Simple models, moderate accuracy |
| **SSP-RK3** | 3 | 3 | C=1 | **Default choice** — best balance of accuracy, stability, and SSP |
| **Classic RK4** | 4 | 4 | No | High-accuracy reference solutions |
| **SSP-RK(10,4)** | 4 | 10 | C=6 | When you need 4th-order *and* SSP (large CFL) |

**SSP (Strong Stability Preserving)** methods guarantee that any convex
functional (TV norm, positivity, entropy) that is non-increasing under Forward
Euler is also non-increasing under the SSP-RK method, up to a CFL number
scaled by the SSP coefficient $C$.

!!! tip "Recommendation"
    **SSP-RK3** (`rk3_ssp_step` / `RK3SSP`) is the default choice for most
    ocean PDE work.  It is the optimal 3rd-order SSP method (C=1) and is used
    by ROMS, MOM6, and many other models.

### Multistep Methods

Multistep methods reuse RHS evaluations from previous time steps, so they
require only **one new RHS evaluation per step** — half the cost of Heun.
The trade-off is that they need history and a bootstrap phase.

| Method | Order | RHS evals/step | Notes |
|---|---|---|---|
| **Adams-Bashforth 2 (AB2)** | 2 | 1 | Used in MITgcm, NEMO |
| **Adams-Bashforth 3 (AB3)** | 3 | 1 | Higher accuracy, needs 2-step history |
| **Leapfrog + Robert-Asselin** | 2 | 1 | Classic NWP scheme; computational mode damped by filter |

!!! note "Leapfrog trade-offs"
    Leapfrog is a centred (non-dissipative) scheme, making it attractive for
    wave propagation.  However, it supports a spurious computational mode that
    must be damped by the Robert-Asselin filter (parameter $\alpha$, typically
    0.01–0.1).  The filter introduces $O(\alpha)$ dissipation.

### IMEX (Implicit-Explicit)

IMEX methods split the RHS into a non-stiff part treated explicitly and a
stiff part treated implicitly:

$$
\frac{\partial \mathbf{y}}{\partial t}
  = \underbrace{\mathbf{F}_E(\mathbf{y})}_{\text{advection, Coriolis}}
  + \underbrace{\mathbf{F}_I(\mathbf{y})}_{\text{vertical diffusion}}
$$

The implicit part requires solving a linear system at each step (e.g., a
tridiagonal solve for vertical diffusion columns), but this allows much larger
$\Delta t$ than fully explicit treatment of stiff terms.

finitevolX provides **IMEX-SSP2(2,2,2)**: a 2nd-order method where the
explicit part is SSP and the implicit part is A-stable (SDIRK with
$\gamma = 1 - 1/\sqrt{2}$).

### Split-Explicit

The dominant strategy in realistic ocean models (ROMS, MOM6, NEMO).  The
key insight: barotropic gravity waves are 100x faster than baroclinic
internal waves, but the barotropic equations are 2D (cheap).

**Algorithm:**

1. **Subcycle** the 2D barotropic equations with $N$ small Forward-Euler
   steps ($\Delta t_{\text{fast}} = \Delta t_{\text{slow}} / N$).
2. **Time-average** the barotropic solution to filter fast oscillations.
3. **Advance** the 3D baroclinic equations with one slow step using the
   averaged barotropic state.

This decouples the barotropic CFL from the slow timestep, allowing
$\Delta t_{\text{slow}}$ to be set by the (much slower) baroclinic dynamics.

### Semi-Lagrangian

Instead of advancing fields on a fixed grid (Eulerian), semi-Lagrangian
methods trace characteristic curves backward from each grid point, then
interpolate the old field at the departure point.

**Key property: unconditionally stable** — the CFL number can exceed 1,
which is impossible with explicit Eulerian advection.  This makes
semi-Lagrangian attractive for models where the flow speed is high
relative to the grid spacing (e.g., atmospheric models, ECMWF IFS).

The trade-off is that interpolation introduces numerical diffusion, and
monotonicity/conservation requires additional corrections.

---

## Decision Guide

```
Is vertical diffusion stiff?
├── Yes → Use IMEX (imex_ssp2_step) for the stiff/non-stiff split
└── No ↓

Do you have barotropic/baroclinic splitting?
├── Yes → Use split-explicit (split_explicit_step)
└── No ↓

Do you need CFL > 1 for advection?
├── Yes → Use semi-Lagrangian (semi_lagrangian_step)
└── No ↓

Is cost per step critical (very long runs)?
├── Yes → Use AB2 (ab2_step) — 1 RHS eval/step
└── No ↓

Default → SSP-RK3 (rk3_ssp_step / RK3SSP)
```

---

## References

- Shu & Osher (1988) — SSP-RK3 foundations
- Gottlieb, Shu & Tadmor (2001) — Strong stability-preserving methods
- Ketcheson (2008) — SSP-RK(10,4) with low-storage implementations
- Durran (2010) — *Numerical Methods for Fluid Dynamics*
- Pareschi & Russo (2005) — IMEX Runge-Kutta methods
- Shchepetkin & McWilliams (2005) — Split-explicit time stepping in ROMS
- Robert (1966) — The Robert-Asselin time filter
- Staniforth & Cote (1991) — Semi-Lagrangian integration schemes
