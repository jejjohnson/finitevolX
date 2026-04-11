# Spatial Operators

This page covers the theory behind the higher-level spatial operators in
finitevolX: divergence, vorticity (relative and potential), diffusion,
Coriolis, momentum advection (vortex-force form), kinetic energy, and
diagnostic quantities.

---

## Overview

The spatial operators build on the primitive [difference](difference.md) and
[interpolation](interpolation.md) operators to compute physically meaningful
quantities on the Arakawa C-grid.  The table below lists the main operators
and what they compute:

| Operator | Module | Output location | Physical quantity |
|----------|--------|-----------------|-------------------|
| `Divergence2D` | divergence | T-points | $\nabla \cdot (u, v)$ |
| `Vorticity2D.relative_vorticity` | vorticity | X-points | $\zeta = \partial v/\partial x - \partial u/\partial y$ |
| `Vorticity2D.potential_vorticity` | vorticity | X-points | $q = (\zeta + f)/h$ |
| `Vorticity2D.pv_flux_energy_conserving` | vorticity | U/V-points | $(q\,Uh, q\,Vh)$ |
| `Diffusion2D` / `diffusion_2d` | diffusion | T-points | $\nabla \cdot (\kappa\,\nabla h)$ |
| `BiharmonicDiffusion2D` | diffusion | T-points | $-\kappa\,\nabla^4 h$ |
| `Coriolis2D` | coriolis | U/V-points | $(+f\bar{v}, -f\bar{u})$ |
| `MomentumAdvection2D` | momentum | U/V-points | Vortex-force form |
| `arakawa_jacobian` | jacobian | interior | $J(f, g)$ |

---

## Divergence

### Theory

The **flux-form divergence** of the horizontal velocity field at T-points:

$$
\delta_{j,i}
= \frac{u_{j,i+\frac{1}{2}} - u_{j,i-\frac{1}{2}}}{\Delta x}
+ \frac{v_{j+\frac{1}{2},i} - v_{j-\frac{1}{2},i}}{\Delta y}
$$

In the shallow-water model the continuity equation is
$\partial h/\partial t + \nabla \cdot (h\,\mathbf{u}) = 0$, so computing
the divergence of the **mass flux** $(hu, hv)$ gives the thickness tendency
$-\delta(hu, hv)$.

For a non-divergent velocity field ($\delta = 0$), the mass flux form and
the simpler advective form agree.  On a C-grid, the discrete divergence is
exactly zero for any velocity field derived from a streamfunction via
`grad_perp` — this is the key property that makes QG models mass-conservative.

### Usage

```python
from finitevolx import CartesianGrid2D, Divergence2D, divergence_2d

grid = CartesianGrid2D.from_interior(64, 64, 1e6, 1e6)

# Class-based
div_op = Divergence2D(grid=grid)
delta = div_op(u, v)              # T-points

# Functional (no grid object needed)
delta = divergence_2d(u, v, dx=grid.dx, dy=grid.dy)
```

---

## Vorticity

### Relative Vorticity

The **relative vorticity** at X-points (NE corners):

$$
\zeta_{j+\frac{1}{2}, i+\frac{1}{2}}
= \frac{v_{j+\frac{1}{2},i+1} - v_{j+\frac{1}{2},i}}{\Delta x}
- \frac{u_{j+1,i+\frac{1}{2}} - u_{j,i+\frac{1}{2}}}{\Delta y}
$$

This is the discrete **curl** of the 2-D velocity field.  It is placed at
X-points because this is the natural location for `diff_x_V_to_X` and
`diff_y_U_to_X`.

### Potential Vorticity (Shallow Water)

The **shallow-water potential vorticity** at X-points:

$$
q_{j+\frac{1}{2}, i+\frac{1}{2}}
= \frac{\zeta_{j+\frac{1}{2}, i+\frac{1}{2}} + f_{j+\frac{1}{2}, i+\frac{1}{2}}}{h_{j+\frac{1}{2}, i+\frac{1}{2}}}
$$

where $f$ and $h$ are interpolated from T-points to X-points via `T_to_X`.

!!! warning "Thin-layer singularity"
    $q$ is set to `NaN` where $h = 0$ to avoid division by zero.  Guard
    against this with a minimum thickness clamp before calling the PV operator.

### PV Flux (Energy-Conserving)

The Sadourny (1975) **energy-conserving** PV-flux form of the vorticity
advection term moves $(q\,Uh)$ and $(q\,Vh)$ to U/V-points for use in the
momentum equation.  This scheme conserves total kinetic energy exactly for
non-divergent flow.

### Usage

```python
from finitevolx import CartesianGrid2D, Vorticity2D

grid = CartesianGrid2D.from_interior(64, 64, 1e6, 1e6)
vort = Vorticity2D(grid=grid)

# Relative vorticity at X-points
zeta = vort.relative_vorticity(u, v)

# Shallow-water PV at X-points
q = vort.potential_vorticity(u, v, h, f)

# PV flux for momentum advection
qu, qv = vort.pv_flux_energy_conserving(q, u*h, v*h)
```

---

## Diffusion

### Harmonic Diffusion

The **harmonic (Laplacian) diffusion** computes the tracer tendency
$\partial h/\partial t = \nabla \cdot (\kappa\,\nabla h)$ in **flux form**
using forward-then-backward differences:

1. **East-face flux** (forward T → U):
   $$
   F_{j,i+\frac{1}{2}}^x = \kappa\,\frac{h_{j,i+1} - h_{j,i}}{\Delta x}
   $$

2. **North-face flux** (forward T → V):
   $$
   F_{j+\frac{1}{2},i}^y = \kappa\,\frac{h_{j+1,i} - h_{j,i}}{\Delta y}
   $$

3. **Tendency** (backward U/V → T):
   $$
   \frac{\partial h}{\partial t}\bigg|_{j,i}
   = \frac{F_{j,i+\frac{1}{2}}^x - F_{j,i-\frac{1}{2}}^x}{\Delta x}
   + \frac{F_{j+\frac{1}{2},i}^y - F_{j-\frac{1}{2},i}^y}{\Delta y}
   $$

For uniform $\kappa$ this reduces to $\kappa\,\nabla^2 h$.  For spatially
varying $\kappa$ (e.g. near coastlines or mixed-layer parameterisations) the
flux form ensures discrete conservation of the diffused quantity.

**Boundary conditions by default:** Face fluxes at domain walls are zero
(no-flux, closed-wall), giving a homogeneous Neumann BC $\partial h/\partial n = 0$.

### Biharmonic Diffusion

**Biharmonic diffusion** applies the harmonic operator twice:

$$
\frac{\partial h}{\partial t} = -\kappa\,\nabla^4 h
  = -\kappa\,\nabla^2(\nabla^2 h)
$$

It damps small scales much more strongly than large scales:
for a wavenumber $k$, harmonic damping scales as $\kappa k^2$
while biharmonic scales as $\kappa k^4$.  This is used to remove
grid-scale noise without excessively damping the resolved mesoscale.

!!! tip "Which diffusion to use"
    - **Harmonic** (`Diffusion2D`): simple, physically motivated, appropriate for
      tracer diffusion when scale-selectivity is not critical.
    - **Biharmonic** (`BiharmonicDiffusion2D`): preferred for **velocity**
      diffusion in eddy-resolving models because it acts mainly on unresolved
      scales.  Requires a larger $\kappa$ to achieve the same total dissipation.

### Mask Support

All diffusion operators accept optional `mask_h`, `mask_u`, `mask_v` arrays.
Land-cell fluxes are zeroed and land-cell tendencies are zeroed, giving
natural no-flux BCs at irregular coastlines without any additional code.

### Usage

```python
from finitevolx import Diffusion2D, BiharmonicDiffusion2D, diffusion_2d
from finitevolx import CartesianGrid2D

grid = CartesianGrid2D.from_interior(64, 64, 1e6, 1e6)

# Class-based harmonic diffusion (unmasked)
diff_op = Diffusion2D(grid=grid)
dh_dt = diff_op(h, kappa=100.0)

# With masking — mask is a class field, not a per-call kwarg.
# Diffusion applies the intermediate flux-masking pattern internally
# (flux_x *= mask.u, flux_y *= mask.v, tendency *= mask.h).
diff_op_masked = Diffusion2D(grid=grid, mask=mask)
dh_dt = diff_op_masked(h, kappa=100.0)

# Biharmonic (scale-selective)
biharm_op = BiharmonicDiffusion2D(grid=grid)
dh_dt = biharm_op(h, kappa=1e9)

# Functional API (no grid object, mask-free)
dh_dt = diffusion_2d(h, kappa=100.0, dx=grid.dx, dy=grid.dy)
```

---

## Coriolis

### Theory

The Coriolis term in the horizontal momentum equations on a C-grid:

$$
\frac{\partial u}{\partial t}\bigg|_{\text{Cor}}
  = +f_{\text{on-U}}\,\bar{v}_{\text{on-U}}
$$

$$
\frac{\partial v}{\partial t}\bigg|_{\text{Cor}}
  = -f_{\text{on-V}}\,\bar{u}_{\text{on-V}}
$$

The Coriolis parameter $f$ (defined at T-points) is interpolated to U/V-points
by simple x/y averaging.  The cross-velocity $v$ (at V-points) is interpolated
to U-points via the **4-point bilinear average** `Interpolation2D.V_to_U`, and
vice versa.  This is the standard C-grid Coriolis discretisation from Sadourny
(1975), which conserves energy.

### Usage

```python
from finitevolx import CartesianGrid2D, Coriolis2D

grid = CartesianGrid2D.from_interior(64, 64, 1e6, 1e6)
cor = Coriolis2D(grid=grid)

# f at T-points
f = jnp.full((grid.Ny, grid.Nx), 1e-4)

du_cor, dv_cor = cor(u, v, f)
```

---

## Momentum Advection (Vortex-Force Form)

### Theory

The **vector-invariant (vortex-force)** form of the horizontal momentum
advection separates the advection into a vorticity flux and a kinetic-energy
gradient:

$$
\frac{\partial u}{\partial t}\bigg|_{\text{adv}}
  = +(\zeta\,v)_{\text{on-U}} - \frac{\partial K}{\partial x}
$$

$$
\frac{\partial v}{\partial t}\bigg|_{\text{adv}}
  = -(\zeta\,u)_{\text{on-V}} - \frac{\partial K}{\partial y}
$$

where $\zeta = \partial v/\partial x - \partial u/\partial y$ is the relative
vorticity at X-points, $K = \tfrac{1}{2}(\bar{u}^2 + \bar{v}^2)$ is the
kinetic energy at T-points, and $(\zeta\,v)_{\text{on-U}}$ is the
vorticity-flux product interpolated to U-points.

This form has important discrete conservation properties depending on how
the vorticity-flux product is evaluated.  Three schemes are available:

| Scheme | Key property | Reference |
|--------|-------------|-----------|
| `'energy'` (default) | Conserves total KE for non-divergent flow | Sadourny (1975) E-scheme |
| `'enstrophy'` | Conserves potential enstrophy | Sadourny (1975) Z-scheme |
| `'al'` | Conserves both KE and enstrophy | Arakawa & Lamb (1981) |

!!! tip "When to use each scheme"
    - **`'energy'`**: well-tested, used in ROMS and many ocean models.
    - **`'enstrophy'`**: preferred when enstrophy conservation matters (e.g.
      turbulence spindown experiments).
    - **`'al'`**: gold standard for QG-like models; more expensive (computes
      both E- and Z-schemes then blends).

### Usage

```python
from finitevolx import CartesianGrid2D, MomentumAdvection2D

grid = CartesianGrid2D.from_interior(64, 64, 1e6, 1e6)

# Energy-conserving (default)
madv = MomentumAdvection2D(grid=grid)
du_adv, dv_adv = madv(u, v)

# Enstrophy-conserving
madv_z = MomentumAdvection2D(grid=grid)
du_adv, dv_adv = madv_z(u, v, scheme="enstrophy")

# Arakawa-Lamb (both)
du_adv, dv_adv = madv(u, v, scheme="al")
```

---

## Arakawa Jacobian

### Theory

The **Arakawa (1966) Jacobian** $J(f, g) = \partial f/\partial x\,\partial g/\partial y - \partial f/\partial y\,\partial g/\partial x$ is the classical advection operator for QG models.  Unlike simple centred differences, the three-term Arakawa average

$$
J(f, g) = \tfrac{1}{3}(J^{++} + J^{+\times} + J^{\times+})
$$

conserves **energy**, **enstrophy**, and satisfies $J(f, f) = 0$ and
$\int J(f, g)\,\mathrm{d}A = 0$ exactly at the discrete level.

The three terms are:
- $J^{++}$: standard centred (advective) form
- $J^{+\times}$: flux form in one direction
- $J^{\times+}$: flux form in the other direction

!!! note "Output shape"
    `arakawa_jacobian` returns the interior `[..., Ny−2, Nx−2]` without ghost
    cells.  The caller must embed this into the full `[Ny, Nx]` array before
    using the result in a time-stepping loop.

### Usage

```python
from finitevolx import arakawa_jacobian
from finitevolx import CartesianGrid2D

grid = CartesianGrid2D.from_interior(64, 64, 1e6, 1e6)

# QG vorticity advection: J(psi, q)
# psi, q have shape [Ny, Nx] with one ghost cell on each side
Jpsi_q = arakawa_jacobian(psi, q, dx=grid.dx, dy=grid.dy)
# Jpsi_q has shape [Ny-2, Nx-2]
```

---

## Decision Guide

```
What spatial operator do you need?
│
├── Divergence ∇·(u, v)
│   └── Divergence2D  or  divergence_2d
│
├── Relative vorticity ζ = ∂v/∂x − ∂u/∂y
│   ├── Class-based → Vorticity2D.relative_vorticity
│   └── Functional  → Difference2D.curl
│
├── Shallow-water PV q = (ζ+f)/h
│   └── Vorticity2D.potential_vorticity
│
├── Tracer diffusion ∂q/∂t = ∇·(κ∇q)
│   ├── Harmonic   → Diffusion2D  or  diffusion_2d
│   └── Biharmonic → BiharmonicDiffusion2D  (scale-selective)
│
├── Coriolis tendency (f×v, −f×u)
│   └── Coriolis2D
│
├── Momentum advection ∂u/∂t|adv
│   ├── Conservative vortex-force form → MomentumAdvection2D
│   └── QG vorticity advection         → arakawa_jacobian
│
└── Diagnostics (KE, APE, enstrophy, Okubo-Weiss, …)
    └── See [Diagnostics API reference](api/diagnostics.md)
```

---

## References

- Sadourny (1975) — C-grid momentum advection, energy- and enstrophy-conserving schemes
- Arakawa & Lamb (1981) — Combined energy/enstrophy conserving scheme
- Arakawa (1966) — Energy/enstrophy-conserving Jacobian operator
- Smagorinsky (1963) — Horizontal diffusion parameterisation
- Haidvogel & Beckmann (1999) — Momentum advection in ocean models
