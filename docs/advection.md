# Advection Operators

This page covers the theory behind the advection operators in finitevolX:
the flux-form finite-volume discretisation, what the inputs mean (cell
averages), the formal order of the scheme, the available reconstruction
schemes (upwind, TVD, WENO) with their measured accuracy, the mask, wall and
spherical semantics, and a decision guide built from those measurements.

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
T-points; which T-points are written depends on the wall mode (see
[Walls, write region and periodic halos](#walls-write-region-and-periodic-halos)).

---

## Inputs Are Cell Averages

The operators are finite-volume schemes: `h` (the tracer argument) holds the
**cell average**

$$
\bar{q}_{j,i} = \frac{1}{\Delta x\,\Delta y}\int_{y_{j-1/2}}^{y_{j+1/2}}\int_{x_{i-1/2}}^{x_{i+1/2}} q(x, y)\,dx\,dy ,
$$

not the value at the cell centre, and the tendency they return is the
tendency of that cell average.  Every reconstruction below turns the cell
averages $\bar{q}_{i-2}, \dots, \bar{q}_{i+2}$ into an estimate of the
**point value** $q(x_{i+1/2})$ on the face.  The two readings agree to
$O(\Delta x^2)$ ($\bar{q}_i = q(x_i) + \tfrac{\Delta x^2}{24} q''(x_i) + \dots$), so
the distinction only matters for schemes above second order.  Initialise
with cell averages and compare with cell averages to see the nominal order.

The largest polynomial degree $d$ for which the face value of $x^d$ is
reproduced exactly (both flow signs; `scripts/benchmark_advection.py --part exactness`):

| Method | `naive` | `upwind1` | `upwind2` | `upwind3` | `weno3` | `weno5` / `wenoz5` | `weno7` | `weno9` | TVD limiters |
|---|---|---|---|---|---|---|---|---|---|
| fed cell averages | 1 | 0 | 1 | 2 | 1 | 2 | 3 | 4 | 1 |
| fed point values  | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

The WENO schemes are exact to the degree of their candidate stencils
($k-1$ for $k$ candidates), not to the degree of the optimal linear
combination: on a polynomial the nonlinear weights differ from the linear
ones.  They still converge at the nominal rate on smooth data, where the
weights tend to the linear weights (see the order tables below).

Feeding point values instead of averages costs nothing in 1-D with a
constant velocity (the scheme is linear and translation-invariant, so the
two readings have the same truncation order: the conservative
finite-difference equivalence), but with a **spatially varying velocity**
every scheme above second order drops to second order, because the face
value is then multiplied by a different velocity on each face.  Measured
1-D L∞ / L1 orders of the tendency of $e^{\sin 2\pi x}$ (periodic,
$N = 128 \to 256$; `--part convergence`):

| Method | $u = 1$, averages | $u = 1$, point values | $u(x) = 1 + \tfrac12\sin 2\pi x$, averages | $u(x)$, point values |
|---|---|---|---|---|
| `upwind3` | 3.0 / 3.0 | 3.0 / 3.0 | 3.0 / 3.0 | 2.2 / 2.1 |
| `weno5`   | 5.0 / 5.0 | 5.0 / 5.0 | 5.0 / 5.0 | 2.0 / 2.0 |
| `wenoz5`  | 5.0 / 5.0 | 5.0 / 5.0 | 5.0 / 5.0 | 2.0 / 2.0 |
| `weno7`   | 5.1 / 6.1 | 5.1 / 6.1 | 5.1 / 6.1 | 2.0 / 2.0 |
| `weno9`   | 8.6 / 8.8 | 9.0 / 8.9 | 8.4 / 8.8 | 2.0 / 2.0 |

---

## Flux-Form Algorithm

### 2-D Case

The discrete tendency at interior T-point $(j, i)$ is

$$
\left.\frac{\partial \bar{q}}{\partial t}\right|_{j,i}
= -\frac{f_{j,i+\frac{1}{2}}^x - f_{j,i-\frac{1}{2}}^x}{\Delta x}
  -\frac{f_{j+\frac{1}{2},i}^y - f_{j-\frac{1}{2},i}^y}{\Delta y}
$$

where the face-normal fluxes are

$$
f_{j,i+\frac{1}{2}}^x = u_{j,i+\frac{1}{2}} \cdot \hat{q}_{j,i+\frac{1}{2}}, \qquad
f_{j+\frac{1}{2},i}^y = v_{j+\frac{1}{2},i} \cdot \hat{q}_{j+\frac{1}{2},i}
$$

and $\hat{q}$ is the **reconstructed face value** obtained from the cell
averages of row $j$ (column $i$) by one of the 1-D schemes described below.

### Formal Order

The 2-D (and 3-D) operators are **unsplit**: both directions' fluxes are
computed from the same state, and the time integrator (e.g. `rk3_ssp_step`)
advances their sum, so there is no splitting error and the temporal order
is the integrator's.  In space, each face value is a **1-D reconstruction
along the face normal**, and each face flux is the face-midpoint velocity
times that value (midpoint quadrature along the face):

- applied to the 2-D cell averages of one row, a 1-D reconstruction of order
  $k$ returns the *face average* $\frac{1}{\Delta y}\int q(x_{i+1/2}, y)\,dy$
  to $O(\Delta x^k)$, so with a **constant velocity** the scheme keeps the
  reconstruction's order $k$ in the strict finite-volume sense;
- with a **spatially varying velocity**, the midpoint velocity times the
  face-averaged $q$ differs from the face average of $u q$ by
  $O(\Delta y^2)$, so the strict finite-volume order is **2** for every
  scheme above second order.  (Order $k$ would need several Gauss points
  per face, or a genuinely 2-D reconstruction; finitevolX does neither.)

Measured L∞ orders of the 2-D cell-average tendency against the exact one
($N = 64 \to 128$, `--part order2d`): constant $(u, v)$: `upwind3` 3.0,
`weno5` 5.0, `weno9` 8.8; divergence-free $u(y)$, $v(x)$: 2.1, 2.0, 2.0.
In practice the high-order schemes still have far smaller errors on smooth
flows (the second-order term has a small constant): see the
[accuracy benchmark](#measured-accuracy-and-cost).

### Face Values and Upwinding

Every reconstruction is upwinded on the sign of the face velocity: with
$u_{i+1/2} \ge 0$ it uses a left-biased stencil (upwind cell $i$), otherwise
the mirrored right-biased one (upwind cell $i+1$),
`h_face = jnp.where(u >= 0, left_biased, right_biased)`, and the flux is
`h_face * u`.  The `Reconstruction1D/2D/3D` methods (`upwind1_x`,
`weno5_y`, `tvd_x(h, u, limiter=...)`, …) return that flux as a full
`[Ny, Nx]` array with a zero ghost ring.

`upwind_flux(q, u, dim, rec_funcs, mask_hierarchy)` is the **coastal
tier blender**, not a two-state flux: it evaluates every reconstruction in
`rec_funcs` (`{stencil_size: fn(q, u) -> flux}`) and, at each face, keeps
the one whose stencil size the mask hierarchy assigns to the upwind cell:

```python
import jax.numpy as jnp
from finitevolx import CartesianGrid2D, Mask2D, Reconstruction2D, upwind_flux

grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
recon = Reconstruction2D(grid=grid)
mask = Mask2D.from_dimensions(grid.Ny, grid.Nx)
hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
rec_funcs = {2: recon.upwind1_x, 4: recon.weno3_x, 6: recon.weno5_x}
q = jnp.ones((grid.Ny, grid.Nx))
u = jnp.ones((grid.Ny, grid.Nx))
fe = upwind_flux(q, u, dim=1, rec_funcs=rec_funcs, mask_hierarchy=hier)
# fe: [Ny, Nx] east-face flux, ghost ring zero
```

The advection operators build these dictionaries themselves; call it
directly only to compose a custom tier ladder.

---

## Reconstruction Schemes

The key design choice is how to estimate the face value $\hat{q}$ from
the cell averages.  The operators' default is `method="upwind1"`.

| Method | Nominal order | Measured order L∞ / L1 | Cells read per face | Mask tier (fallback) | Non-oscillatory |
|--------|---|---|---|---|---|
| `naive` (centred) | 2 | 2.0 / 2.0 | 2 | — (dry face zeroed) | No (dispersive) |
| `upwind1` | 1 | 1.0 / 1.0 | 1 | 2 | Yes (monotone) |
| `upwind2` | 2 | 2.0 / 2.0 | 2 | 4 (→ `upwind1`) | No |
| `upwind3` | 3 | 3.0 / 3.0 | 3 | 4 (→ `upwind1`) | No (small undershoots) |
| `weno3` | 3 | 1.0 / 2.0 | 3 | 4 (→ `upwind1`) | Essentially (ENO) |
| `weno5` | 5 | 5.0 / 5.0 | 5 | 6 (→ `weno3` → `upwind1`) | Essentially (ENO) |
| `wenoz5` | 5 | 5.0 / 5.0 | 5 | 6 (→ `wenoz3` → `upwind1`) | Essentially (ENO) |
| `weno7` | 7 | 5.1 / 6.1 | 7 | 8 (→ `weno5` → …) | Essentially (ENO) |
| `weno9` | 9 | 8.6 / 8.8 | 9 | 10 (→ `weno7` → …) | Essentially (ENO) |
| `minmod`, `van_leer`, `superbee`, `mc` | 2 | 1.0 / 2.0 | 3 | 4 (→ `upwind1`) | Yes (TVD) |

"Measured order" is the 1-D rate of the tendency of $e^{\sin 2\pi x}$ from
cell averages ($N = 128 \to 256$), a smooth field *with extrema*.  At smooth
extrema the TVD limiters clip to first order (hence L∞ 1, L1 2), and the
Jiang–Shu weights of WENO3 and WENO7 degrade (WENO3-JS is first order in
L∞ there, WENO7-JS fifth); WENO-Z does not.  `Advection3D` and
`SphericalAdvection3D` accept every method except `upwind2` / `upwind3`.

### Total Variation Diminishing (TVD)

TVD schemes add a **flux limiter** $\phi(r)$ to a first-order upwind base.
For $u_{i+1/2} \ge 0$ the code (`Reconstruction*.tvd_x`) evaluates

$$
\hat{q}_{i+\frac{1}{2}} = \bar{q}_i + \tfrac{1}{2}\,\phi(r)\,(\bar{q}_{i+1} - \bar{q}_i),
\qquad r = \frac{\bar{q}_i - \bar{q}_{i-1}}{\bar{q}_{i+1} - \bar{q}_i}
$$

(upwind slope over downwind slope), and the mirror image for $u < 0$:
$\hat{q}_{i+1/2} = \bar{q}_{i+1} + \tfrac12\phi(r)(\bar{q}_i - \bar{q}_{i+1})$
with $r = (\bar{q}_{i+1} - \bar{q}_{i+2}) / (\bar{q}_i - \bar{q}_{i+1})$.
The four limiters are symmetric, $\phi(r)/r = \phi(1/r)$, so this equals the
other common form $\bar{q}_i + \tfrac12\phi(1/r)(\bar{q}_i - \bar{q}_{i-1})$.
The denominator of $r$ is guarded relative to the slopes,
$r = \Delta_{\text{up}} / (\Delta_{\text{down}} + \operatorname{sign}(\Delta_{\text{down}})\,(10^{-6}\lvert\Delta_{\text{up}}\rvert + \sqrt{\text{tiny}}))$:
it scales exactly with the field, keeps the sign of a tiny downwind slope,
and a flat stencil gives $r = 0$ with a finite gradient.

The limiter keeps $\phi$ in the TVD region, so with an SSP time integrator
under its CFL limit the total variation $\sum_i \lvert\bar{q}_{i+1} - \bar{q}_i\rvert$
cannot increase: no new extrema, no undershoots.

| Limiter | $\phi(r)$ | Character |
|---------|-----------|-----------|
| **Minmod** | $\max(0,\min(1,r))$ | Most diffusive; never overshoots |
| **van Leer** | $(r + \lvert r\rvert)/(1 + \lvert r\rvert)$ | Smooth limiter |
| **Superbee** | $\max(0, \max(\min(2r, 1), \min(r, 2)))$ | Sharpest TVD; squares smooth peaks |
| **MC** | $\max(0, \min((1+r)/2, 2, 2r))$ | Between van Leer and superbee |

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
q_0 = \tfrac{3}{2}\bar{q}_i - \tfrac{1}{2}\bar{q}_{i-1}, \qquad
q_1 = \tfrac{1}{2}\bar{q}_i + \tfrac{1}{2}\bar{q}_{i+1}
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

Achieves 5th-order accuracy in smooth regions.

#### WENOz5 (improved WENO5)

Borges et al. (2008) modification of WENO5 that uses a higher-order
global smoothness indicator $\tau_5 = |\beta_0 - \beta_2|$ to scale the
weights, achieving less dissipation near smooth extrema while remaining
ENO near discontinuities.

#### Relative epsilons: the scheme does not depend on the amplitude

Every WENO weight is $g_k / (b_k + \varepsilon)^2$ (JS) or
$g_k (1 + \tau / (b_k + \varepsilon))$ (Z) on the *normalised* indicators
$b_k = \beta_k / \sum_j \beta_j$, with a dimensionless $\varepsilon = 10^{-6}$:
the textbook weights with a floor relative to the stencil's own smoothness,
$\varepsilon \sum_j \beta_j$, instead of an absolute constant.  Rescaling the
field therefore rescales the reconstruction exactly (checked to rounding,
$\sim 10^{-15}$, from amplitude $10^{-9}$ to $10^{3}$ by
`tests/test_scale_covariance.py`; before #243 an absolute $10^{-8}$ moved
WENO tendencies by up to 3 % at amplitude $10^3$, and the TVD guard made
van Leer first order at $10^{-9}$).

Two details keep the gradient exact and finite.  The weights depend only
on ratios of indicators (they are homogeneous of degree 0), so the
normalising sum is held out of the gradient (`stop_gradient`): by Euler's
theorem that changes nothing, and the normalised quantities are O(1) at any
amplitude, float32 included.  A flat stencil ($\sum_j\beta_j = 0$, an
ocean at rest or a plateau) gives the linear weights and a finite gradient.
Cost against the absolute form (CPU, $256^2$, jitted): forward unchanged
within noise; gradient about +25 % for weno5, unchanged for weno9, and
+50 % for the TVD limiters (still the cheapest gradients).

---

## Measured Accuracy and Cost

Solid-body rotation of a Gaussian, $q_0 = e^{-r^2 / 2\sigma^2}$ with
$\sigma = 0.05$ centred at $(\tfrac12, \tfrac34)$, rotated once about
$(\tfrac12, \tfrac12)$ on $[0, 1]^2$ by $(u, v) = 2\pi(\tfrac12 - y, x - \tfrac12)$;
$N = 64$, cell-average initial condition, `Advection2D` (closed walls,
no mask), `rk3_ssp_step` at CFL 0.25 (805 steps), float64.  `rel-L2` is
$\lVert\bar q(1) - \bar q_0\rVert_2 / \lVert\bar q_0\rVert_2$; min / max are
over the interior after one revolution (initially 0 and 0.968).  `ms` is the
fastest of 50 jitted forward tendencies on a $256^2$ grid (one
`Advection2D.__call__`), measured on a shared 2-vCPU Intel Xeon @ 2.10 GHz VM
(JAX 0.10.1, CPU): indicative only.  Regenerate with
`uv run python scripts/benchmark_advection.py --part benchmark`.

| Method | rel-L2 | min | max | ms ($256^2$) |
|---|---|---|---|---|
| `naive` | 0.49 | −0.16 | 0.822 | 0.09 |
| `upwind1` | 0.80 | 0 | 0.145 | 0.10 |
| `upwind2` | 0.57 | −0.11 | 0.617 | 0.17 |
| `upwind3` | 0.13 | −7.7e-3 | 0.815 | 0.21 |
| `weno3` | 0.38 | 0 | 0.473 | 0.65 |
| `weno5` | 3.3e-2 | −1.5e-11 | 0.929 | 0.83 |
| `wenoz5` | 1.5e-2 | −1.8e-8 | 0.949 | 0.82 |
| `weno7` | 5.4e-3 | 0 | 0.959 | 1.35 |
| `weno9` | 6.4e-4 | −7.1e-9 | 0.967 | 1.71 |
| `minmod` | 0.44 | 0 | 0.429 | 0.27 |
| `van_leer` | 0.27 | 0 | 0.600 | 0.38 |
| `superbee` | 0.15 | 0 | 0.817 | 0.50 |
| `mc` | 0.21 | 0 | 0.681 | 0.53 |

What it says:

- **WENO3-JS is a poor choice on its own**: 0.38, worse than every TVD
  limiter except minmod and three times worse than the linear `upwind3`, at
  four times the cost of `upwind3`.  It earns its place as the coastal
  fallback of `weno5`.
- `wenoz5` halves the `weno5` error at the same cost; `weno7` / `weno9`
  cut the `wenoz5` error by another ~3× / ~25× for ~1.6× / ~2× the cost.
- The TVD limiters are the only schemes guaranteed free of undershoots;
  among them superbee and MC are the most accurate on this smooth bump.
- The WENO undershoots are tiny ($\le 2\times10^{-8}$ of the peak) but not
  guaranteed zero; `naive` and `upwind2` are dispersive (undershoots of
  16 % and 11 % of the peak).

---

## Masks: Land-Safe Advection

Every method of `Advection1D/2D/3D` and `SphericalAdvection2D/3D` is
land-safe when the operator is constructed with a mask.  The mask is a
**constructor field** (`Advection2D(grid=grid, mask=mask)`), not a
`__call__` argument: `__init__` pre-builds the adaptive stencil hierarchy
`(2, 4, 6, 8, 10)` in both horizontal directions once, and every call
reuses it.

- **Dry inputs are zeroed** with `jnp.where` before any stencil reads them
  (`h` on dry T-cells, `u` on dry U-faces, `v` on dry V-faces), so `NaN` or
  a fill value on land reaches neither the output nor the gradient.
- **Narrower stencils near the coast.**  Each T-point's
  `StencilCapability2D` gives the widest centred stencil that stays wet in
  each direction (tiers 2, 4, 6, 8, 10 = 1, 3, 5, 7, 9 cells).  Every
  upwind-biased method goes through `upwind_flux` and, at each face, uses
  the highest tier of its own ladder that the upwind cell supports (the
  "Mask tier" column above), so no chosen stencil reads a land cell.
  `naive` reads only the two cells beside the face and needs no ladder.
- **No flux through a dry face.**  The face fluxes on every face with
  `mask.u` / `mask.v` false are zeroed before the divergence, as in
  `Diffusion2D`, so no mass crosses the coastline: in a closed basin
  $\sum A\,\partial_t\bar q = 0$ to rounding.
- **Dry output is exactly zero** (`jnp.where(mask.h, out, 0)`), and an
  all-ocean mask reproduces `mask=None` bit for bit.
- `wall="open"` together with a mask raises `NotImplementedError`.

```python
import jax.numpy as jnp
import numpy as np
from finitevolx import Advection2D, CartesianGrid2D, Mask2D

grid = CartesianGrid2D.from_interior(16, 16, 1.0, 1.0)
wet = np.ones((grid.Ny, grid.Nx), dtype=bool)
wet[:2, :] = wet[-2:, :] = wet[:, :2] = wet[:, -2:] = False  # land walls
wet[7:11, 7:11] = False  # an island
mask = Mask2D.from_mask(wet)
adv = Advection2D(grid=grid, mask=mask)  # hierarchy built once, here

h = jnp.where(mask.h, 1.0, jnp.nan)  # NaN on land is fine
u = jnp.ones((grid.Ny, grid.Nx))  # [Ny, Nx] U-points
v = 0.5 * jnp.ones((grid.Ny, grid.Nx))  # [Ny, Nx] V-points
for method in ("naive", "upwind3", "weno5", "weno9", "van_leer"):
    dh = adv(h, u, v, method=method)  # [Ny, Nx]
    assert bool(jnp.isfinite(dh).all())
    assert abs(float(dh.sum())) < 1e-10  # no mass crosses the coastline
```

---

## Walls, Write Region and Periodic Halos

All advection operators take `wall="closed"` (default) or `wall="open"`.

| Operator | `wall="closed"` writes | `wall="open"` writes |
|---|---|---|
| `Advection1D` | `[2:-2]` | `[1:-1]` |
| `Advection2D`, `SphericalAdvection2D` | `[2:-2, 2:-2]` | `[1:-1, 1:-1]` |
| `Advection3D`, `SphericalAdvection3D` | `[1:-1, 2:-2, 2:-2]` | `[1:-1, 1:-1, 1:-1]` |

- **Closed** treats the domain walls as no-flux and never reads the ghost
  flux cells: the outermost row / column of physical cells gets no tendency
  from advection alone, so the caller must account for it.
- **Open** overwrites the wall faces with the first-order upwind flux of the
  caller-filled ghost ring (`ghost = value` for a Dirichlet inflow, a copy
  for outflow, a wrap for periodic), keeps the requested scheme on the
  interior faces, and writes the full interior; the flux stays single-valued,
  so mass is conserved exactly.
- In 3-D the z axis is a batch axis: the horizontal reconstruction and
  divergence are applied to every level by array slicing (no `vmap`), and the
  ghost slabs `k = 0`, `Nz-1` stay zero.

**Periodic halo.**  For the tendency of the physical cells to equal the
periodic scheme's (full-order stencils across the seam, not the boundary
fallbacks), build the grid with $H - 1$ extra cells on each periodic side,
fill the outer $H$ cells of every field by wrapping, and keep the physical
cells `[H:-H]`.  Measured minimum $H$, the same for the Cartesian and the
spherical operators:

| Method | $H$ (`wall="open"`) | $H$ (`wall="closed"`) |
|---|---|---|
| `upwind1` | 1 | 2 |
| `naive`, `upwind2`, `upwind3`, `weno3`, TVD limiters | 2 | 2 |
| `weno5`, `wenoz5` | 3 | 3 |
| `weno7` | 4 | 4 |
| `weno9` | 5 | 5 |

With `wall="open"` a one-cell ghost ring still conserves mass exactly (the
seam face is first-order upwind, the same on both sides); a halo strictly
between 1 and $H$ does not (the two seam faces then use different boundary
fallbacks).

---

## Spherical Advection

`SphericalAdvection2D/3D` reuse the Cartesian reconstructions (they read no
metric) and change only the divergence:

$$
\frac{\partial \bar{q}}{\partial t} = -\frac{1}{R\cos\varphi}
\left[\frac{\partial (q u)}{\partial \lambda} + \frac{\partial (q v \cos\varphi)}{\partial \varphi}\right]
$$

with $\cos\varphi$ at the T-rows for the cell area and at the V-rows for the
meridional faces.  The divergence telescopes against
`spherical_area_weights` ($R^2\cos\varphi_T\,\Delta\lambda\,\Delta\varphi$):
$\sum A\,\partial_t\bar q$ is the net flux through the faces bounding the
written cells.  With `wall="closed"` that region is `[2:-2, 2:-2]`, so on a
zonally periodic grid the seam-adjacent columns get no tendency; use
`wall="open"` with a wrapped halo (table above) for a conservative periodic
seam.  Mask semantics are the Cartesian ones.

---

## Transports and Other Fluxes

**Transports instead of velocities.**  The face value depends on the
velocity only through its sign, so the operators are linear in `u`, `v`.
Passing positively weighted transports, e.g. `U = h_u * u`, `V = h_v * v`
with a layer thickness interpolated to the faces, therefore returns
$-\nabla\cdot(q\,\mathbf{U})$, the tendency of the thickness-weighted
tracer $h q$ (thickness-weighted advection).  There is no dedicated API for
it: compute the face thickness yourself (e.g. `Interpolation2D.T_to_U` /
`T_to_V`), and divide by the updated thickness if you need $q$.

**`rusanov_flux`** (local Lax–Friedrichs,
$F = \tfrac12 a (q_L + q_R) - \tfrac12\lvert a\rvert_\varepsilon (q_R - q_L)$)
returns a **reduced-shape** array: given `q` of length $N$ along `axis` and
the face velocity `a` on the $N - 1$ faces between adjacent cells, it
returns the flux on those $N - 1$ faces, unlike the other fluxes, which are
full ghost-padded C-grid fields.  Take its divergence with `jnp.diff` along
the same axis:

```python
import jax.numpy as jnp
from finitevolx import (
    Advection2D,
    CartesianGrid2D,
    Interpolation2D,
    interior,
    rusanov_flux,
)

grid = CartesianGrid2D.from_interior(16, 16, 1.0, 1.0)
y, x = jnp.meshgrid(jnp.arange(grid.Ny), jnp.arange(grid.Nx), indexing="ij")
q = jnp.sin(0.4 * x) + jnp.cos(0.3 * y)  # [Ny, Nx] T-points
u = jnp.cos(0.2 * y)  # [Ny, Nx] U-points
v = jnp.sin(0.25 * x)  # [Ny, Nx] V-points

# Face k between T-cells k and k+1 is the U-point (V-point) of index k.
fx = rusanov_flux(q, u[:, :-1], axis=-1, eps=0.0)  # [Ny, Nx-1]
fy = rusanov_flux(q, v[:-1, :], axis=-2, eps=0.0)  # [Ny-1, Nx]
dq = interior(
    -(
        jnp.diff(fx, axis=-1)[1:-1, :] / grid.dx
        + jnp.diff(fy, axis=-2)[:, 1:-1] / grid.dy
    ),
    q,
)  # [Ny, Nx]; reads the wall faces, like wall="open"
# With eps=0 the Rusanov flux of the cell values is first-order upwind:
adv = Advection2D(grid=grid)
assert jnp.allclose(dq, adv(q, u, v, method="upwind1", wall="open"))

# Transports: the operator is linear in the velocity.
h = 2.0 + 0.1 * jnp.cos(0.3 * x)  # [Ny, Nx] layer thickness at T-points
interp = Interpolation2D(grid=grid)
U = interp.T_to_U(h) * u  # [Ny, Nx] face transport
V = interp.T_to_V(h) * v  # [Ny, Nx] face transport
dhq = adv(q, U, V, method="weno5")  # -div(q U): tendency of h q
```

---

## Quick Usage

```python
import jax.numpy as jnp
import numpy as np
from finitevolx import CartesianGrid2D, Advection2D, Mask2D

grid = CartesianGrid2D.from_interior(64, 64, 1e6, 1e6)

# Construct once; the operator is method-agnostic.
adv = Advection2D(grid=grid)

# T-point tracer (cell averages) and U/V-point velocities
q = jnp.ones((grid.Ny, grid.Nx))   # T-point tracer
u = jnp.zeros((grid.Ny, grid.Nx))  # U-point velocity
v = jnp.zeros((grid.Ny, grid.Nx))  # V-point velocity

# Default upwind1 (no mask)
dq_dt = adv(q, u, v)

# Choose a method per-call
dq_dt_tvd = adv(q, u, v, method="van_leer")
dq_dt_weno = adv(q, u, v, method="weno5")

# With a land mask: the mask is a constructor field, so build a second
# operator instance with `mask=`.  The adaptive stencil hierarchy is
# pre-built once in `__init__` and reused on every call (docs/masks.md).
ocean_mask = np.ones((grid.Ny, grid.Nx), dtype=bool)
ocean_mask[20:30, 20:30] = False  # an island
mask = Mask2D.from_mask(ocean_mask)
adv_masked = Advection2D(grid=grid, mask=mask)
dq_dt_masked = adv_masked(q, u, v, method="weno5")
# shape [Ny, Nx], non-zero only in [2:-2, 2:-2] and on wet cells
```

---

## Decision Guide

Built from the [measured accuracy table](#measured-accuracy-and-cost)
(solid-body Gaussian, $N = 64$, one revolution; rel-L2 in brackets).

```
Must the tracer never undershoot (positive-definite, bounded)?
├── Yes → a TVD limiter (no undershoot with SSP-RK under its CFL limit)
│         ├── most accurate on smooth bumps    → superbee (0.15) or mc (0.21)
│         ├── smooth limiter, no peak squaring → van_leer (0.27)
│         └── most dissipative                 → minmod (0.44)
└── No ↓

Is accuracy on smooth fields the priority?
├── Yes → WENO (undershoots <= 2e-8 of the peak here, not guaranteed zero)
│         ├── default choice               → wenoz5 (1.5e-2; weno5 3.3e-2, same cost)
│         ├── more accuracy, ~1.6-2x cost  → weno7 (5.4e-3) / weno9 (6.4e-4)
│         └── not weno3 on its own         → (0.38: worse than upwind3 and
│                                             superbee / mc / van_leer)
└── No → cheap and fairly accurate:  upwind3 (0.13; undershoots ~1 %)
         monotone, most dissipative: upwind1 (0.80)
         avoid naive / upwind2 for tracers (dispersive: 16 % / 11 % undershoots)

Does the domain have land / islands?
├── Yes → construct the operator with mask=Mask2D.from_mask(wet):
│         every method is land-safe (coastal fallback, no flux through land)
└── No  → omit the mask

Periodic in x or y?  → wall="open" with the halo H of the table above.
```

---

## References

- Harten, Lax & van Leer (1983) — Upwind schemes and TVD conditions
- van Leer (1979) — MUSCL flux limiters
- Sweby (1984) — High resolution schemes using flux limiters
- Liu, Osher & Chan (1994) — Weighted Essentially Non-Oscillatory schemes
- Jiang & Shu (1996) — WENO5 scheme
- Shu (1998) — ENO and WENO schemes for hyperbolic conservation laws
  (cell averages vs point values, the conservative finite-difference form)
- Borges et al. (2008) — WENOz improved WENO scheme
- Zhang, Zhang & Shu (2011) — On the order of accuracy of two classes of
  finite volume WENO schemes (dimension-by-dimension vs genuinely 2-D)
- Durran (2010) — *Numerical Methods for Fluid Dynamics*, Ch. 5
