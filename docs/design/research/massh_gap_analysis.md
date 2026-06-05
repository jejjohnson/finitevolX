# finitevolX — Gap Analysis vs. MASSH (VarDyn branch)

**Scope:** primitives and operators only. Reference: `leguillf/MASSH@VarDyn`, model cores
`mapping/models/model_qgsw/` (shallow-water) and `mapping/models/model_qg1l/` (QG1L).
**Date:** 2026-06-05.

This document is self-contained: each gap states the math, shows the MASSH reference
implementation (with `file:line`), proposes a `finitevolx` API, and gives a worked example.
The proposals are written against the existing Arakawa-C-grid / Equinox style in
`finitevolx/_src/operators/` and `_src/advection/`.

!!! warning "Provenance of references"
    The finitevolX-side claims in this doc were **verified against the local source on `main`**
    (version `0.0.42`, 2026-06-05) with concrete `file:line` citations and verbatim signatures.
    The MASSH `file:line` citations come from the upstream `VarDyn` branch and were **not**
    re-verified — treat them as pointers, not exact-line guarantees.

!!! important "Two conventions this doc corrects relative to an earlier draft"
    1. **The public API is flat, not namespaced.** `finitevolx/__init__.py` re-exports every
       symbol at the top level (`fvx.divergence_2d`, `fvx.Advection2D`, `fvx.MomentumAdvection2D`).
       There are **no** `fvx.operators.*` / `fvx.advection.*` submodule namespaces — the earlier
       draft's `fvx.operators.divergence_2d(...)` / `fvx.advection.rusanov_flux(...)` would raise
       `AttributeError`. All examples below use the real flat form.
    2. **The energy/enstrophy-conserving SW vorticity scheme already exists** (see §2). The earlier
       draft listed Arakawa–Lamb (1981) as the headline gap; it is in fact implemented. The
       residual §2 gap is narrower.

-----

## 0. What is NOT a gap (so you don't re-scope it)

Confirmed present in finitevolX, equal to or stronger than MASSH (paths verified):

| Capability | finitevolX (verified) | MASSH equivalent |
|---|---|---|
| Conservative Arakawa Jacobian (1966) | `operators/jacobian.py:22` `arakawa_jacobian` | — (MASSH uses upwind vort. advection only) |
| WENO 3/5/7/9 (+ one-sided `_right`) | `advection/weno.py:297–666`, `advection/reconstruction.py` | `reconstruction.py` WENO-Z 4/6 only |
| Flux limiters (minmod / van_leer / superbee / mc) | `advection/limiters.py:34–123` | — |
| C-grid PV (single + multilayer) | `operators/diagnostics.py` `potential_vorticity:150`, `sw_potential_vorticity:828`, `sw_potential_vorticity_multilayer:873` | scattered in `sw.py` |
| KE / Bernoulli / Okubo–Weiss / enstrophy | `operators/diagnostics.py` `kinetic_energy:35`, `bernoulli_potential:76`, `okubo_weiss:274`, `enstrophy:308`, `potential_enstrophy:332` | `finite_diff.py::comp_ke` only |
| **Energy/enstrophy-conserving SW vortex-force** | `operators/vorticity.py:202` `Vorticity2D.pv_flux_arakawa_lamb` (+ `pv_flux_energy_conserving:128`, `pv_flux_enstrophy_conserving:163`); `diffusion/momentum.py:35` `MomentumAdvection2D` (Sadourny E/Z + AL81 blend) | dissipative upwind only |
| Elliptic stack (CG, capacitance, MG, spectral) + PV inversion | `_src/solvers/*` (incl. **variable-coefficient** multigrid `∇·(c∇u)`) | `helmholtz.py`, `helmholtz_multigrid.py` |
| Split-explicit barotropic/baroclinic time-stepping | `timestepping/split_explicit.py:23` `split_explicit_step` | barotropic *filter* (§3) |
| SSP-RK3 (functional + diffrax) | `timestepping/explicit_rk.py:70` `rk3_ssp_step`; `timestepping/diffrax_solvers.py:102` `RK3SSP` | — |

So the gaps below are genuinely net-new operators, **not** reimplementations — with the
important caveat that §2 is narrower than the earlier draft claimed.

-----

## 1. Rusanov / local Lax–Friedrichs flux for the continuity equation  *(real gap)*

**Status: genuine gap.** No Rusanov / local Lax–Friedrichs / smooth-abs flux exists anywhere in
`advection/` (`face_flux.py`, `flux.py`, `linear.py` checked; grep for `rusanov`/`lax`/`llf`
returns nothing).

### Why

The continuity/height flux path is reconstruction-based (WENO). MASSH keeps a separate
**first-order monotone Rusanov flux** for `h` because WENO's nonlinear smoothness weights make
the *adjoint* fragile — the differentiable-DA use case. A Rusanov flux is the standard robust
fallback and a low-dissipation-budget baseline.

### Math

For a conserved scalar `q` advected at face velocity `a` across face `i+½`, the local
Lax–Friedrichs (Rusanov) numerical flux is

$$
F_{i+\frac12} = \tfrac12\,a\,(q_L + q_R) - \tfrac12\,|a|\,(q_R - q_L),
$$

where `q_L, q_R` are the cell values either side of the face and `|a|` is the local maximum
wave speed (here `|a|`). The dissipation term `½|a|(q_R−q_L)` makes it monotone. For AD
stability the absolute value is replaced by the smooth surrogate `|a| ≈ √(a²+ε²)` (the
`smooth_abs` of §5).

### MASSH reference

`mapping/models/model_qgsw/sw.py:674` — `_h_flux_rusanov1`:

```python
def _h_flux_rusanov1(self, h_tot_phys, velocity, dim):
    # First-order conservative Rusanov flux for h-continuity. Monotone;
    # avoids WENO smoothness weights, which make adjoints fragile.
    h_left, h_right = ((h_tot_phys[..., :, :-1], h_tot_phys[..., :, 1:]) if dim == -1
                       else (h_tot_phys[..., :-1, :], h_tot_phys[..., 1:, :]))
    speed = smooth_abs(velocity)                       # √(v²+ε²)
    return 0.5*velocity*(h_left+h_right) - 0.5*speed*(h_right-h_left)
```

with `smooth_abs` at `mapping/models/model_qgsw/reconstruction.py:15`.

### Proposed finitevolX API

Add to `finitevolx/_src/advection/face_flux.py` (it already holds `uv_center_flux`,
`uv_node_flux`; the upwind path lives in `flux.py::upwind_flux`):

```python
def rusanov_flux(
    q: Float[Array, "..."],          # cell-centered scalar (h-grid)
    a: Float[Array, "..."],          # face-normal velocity on the same axis
    axis: int = -1,
    eps: float = 1e-8,               # smooth-abs floor; set 0 for hard |a|
) -> Float[Array, "..."]:
    """Local Lax–Friedrichs (Rusanov) flux at faces along `axis`.

    F = ½ a (q_L + q_R) − ½ |a|_eps (q_R − q_L),  |a|_eps = sqrt(a² + eps²).
    Returns the flux on the (N−1) interior faces along `axis`.
    """
```

Re-export it flat in `finitevolx/__init__.py` (and add to `__all__`), exactly like the existing
`uv_center_flux` / `divergence_2d`. Keep `eps` exposed: `eps=0` recovers the textbook flux;
`eps>0` is the AD-safe variant. The divergence of the flux reuses `divergence_2d`.

### Example

```python
import finitevolx as fvx
import jax.numpy as jnp

h  = jnp.array(...)                                   # (Ny, Nx) on h-grid
u  = jnp.array(...)                                   # face-normal velocity
Fx = fvx.rusanov_flux(h, u, axis=-1, eps=1e-8)        # flat API
Fy = fvx.rusanov_flux(h, v, axis=-2, eps=1e-8)
dhdt = -fvx.divergence_2d(Fx, Fy, dx, dy)             # existing op (flat)
```

-----

## 2. Upwind vorticity-flux operator + a public AL81 free function  *(narrowed gap)*

!!! warning "The conservative scheme already exists"
    The earlier draft listed the Arakawa–Lamb (1981) energy- and enstrophy-conserving SW scheme
    as the headline gap. **It is already implemented** in finitevolX:
    `operators/vorticity.py:202` `Vorticity2D.pv_flux_arakawa_lamb` (a weighted blend of
    `pv_flux_energy_conserving:128` and `pv_flux_enstrophy_conserving:163`), and the
    momentum-level operator `diffusion/momentum.py:35` `MomentumAdvection2D` offers the Sadourny
    (1975) E-scheme, Z-scheme, and the AL81 blend via a `scheme=` argument — citing
    *Arakawa & Lamb (1981), MWR 109, 18–36* in its docstring. Both are in the public `__all__`.

### What is actually missing

1. **The MASSH-parity *upwind* vorticity flux** (`_omega_adv_upwind3/5`) — a *dissipative*
   reconstruction of (absolute) vorticity for the vortex-force term. finitevolX has the
   *conserving* schemes but no upwind vorticity-advection operator (grep `vorticity_flux`,
   `omega_adv` → nothing public). Useful as a robust, cheap baseline and for parity with MASSH.
2. **A public free-function wrapper** for the AL81 flux. Today it is a *method* on `Vorticity2D`;
   a thin `pv_flux_arakawa_lamb(q, U, V, alpha=1/3)` free function (matching the `divergence_2d`
   style) would make it reachable without instantiating the class.

### Math

In vector-invariant SW the momentum tendency carries `q (k × (h u))` with `q = (ζ+f)/h` and
mass flux `h u`. Arakawa–Lamb (1981) discretize this so the operator conserves total energy and
total potential enstrophy simultaneously; the upwind alternative instead reconstructs vorticity
at faces with an order-3/5 upwind stencil (dissipative). MASSH advects *absolute* vorticity
`ω+f` (it folds `f` into the PV); AL81 does the same with `q=(ζ+f)/h`.

### MASSH reference (the upwind variant — the parity target)

`mapping/models/model_qgsw/sw.py:753` — `_omega_adv_upwind3` (3rd-order upwind on corners):

```python
#   vel > 0: omega_face = (-omega[j-1] + 5*omega[j]   + 2*omega[j+1]) / 6
#   vel < 0: omega_face = ( 2*omega[j] + 5*omega[j+1] -   omega[j+2]) / 6
```

used at `sw.py:690` `advection_momentum`:

```python
omega_Vm, omega_Um = self._omega_adv_upwind3(omega, U_m, V_m)
dt_u =  omega_Vm + self.fstar_ugrid[..., 1:-1, :] * V_m
dt_v = -(omega_Um + self.fstar_vgrid[..., 1:-1]   * U_m)
```

### Proposed finitevolX API

```python
# finitevolx/_src/operators/vorticity.py  (alongside Vorticity2D)
def vorticity_flux_upwind(
    omega: Float[Array, "Ny1 Nx1"],   # absolute vorticity on cell corners (q-grid)
    U: Float[Array, "Ny Nx1"],        # mass flux on u-faces  (h·u)
    V: Float[Array, "Ny1 Nx"],        # mass flux on v-faces  (h·v)
    order: int = 3,                   # 3 or 5
) -> tuple[Float[Array, "Ny Nx1"], Float[Array, "Ny1 Nx"]]:
    """Upwind vortex-force fluxes (omega_V on u-eqn, omega_U on v-eqn).
    Matches MASSH sw.py::_omega_adv_upwind{3,5}. Dissipative."""

def pv_flux_arakawa_lamb(                 # thin free-function wrapper over the existing method
    q: Float[Array, "Ny1 Nx1"], U, V, *, alpha: float = 1.0 / 3.0,
) -> tuple[Float[Array, "Ny Nx1"], Float[Array, "Ny1 Nx"]]:
    """Public free-function form of Vorticity2D.pv_flux_arakawa_lamb (AL81)."""
```

Ship the conservation property as a **test**, not a promise: an `enstrophy_conservation_residual`
diagnostic that shows `d/dt Σ q²/2 → 0` to round-off in a closed basin. (Note `potential_enstrophy`
already exists in `diagnostics.py:332` to build that check on.)

### Example

```python
import finitevolx as fvx
# conserving form via the existing class (works today):
vort = fvx.Vorticity2D(grid=grid, mask=None)
pv   = fvx.sw_potential_vorticity(u, v, h, f, dx, dy)        # existing diagnostic
U, V = h_u * u, h_v * v                                      # mass fluxes
CAu, CAv = vort.pv_flux_arakawa_lamb(pv, U, V)               # energy+enstrophy conserving
# or the MASSH-parity dissipative baseline (the §2 gap):
omega_V, omega_U = fvx.vorticity_flux_upwind(omega_abs, U, V, order=3)
```

-----

## 3. Barotropic-mode (external gravity wave) filter / split  *(operator gap; infrastructure present)*

!!! note "The hard parts already exist"
    finitevolX already has (a) **variable-coefficient** multigrid for `∇·(c∇u) − λu = rhs`
    (`solvers/multigrid.py`, factory `build_multigrid_solver(..., coeff=...)`; the per-face
    coefficients are `cx`/`cy` from `_interpolate_coeff_to_faces`, **not** the
    `coef_ugrid`/`coef_vgrid` the earlier draft named), and (b) a **split-explicit**
    barotropic/baroclinic time-stepper (`timestepping/split_explicit.py:23` `split_explicit_step`)
    plus vertical-mode decomposition (`vertical/vertical_modes.py`). What is missing is the
    specific *implicit barotropic filter operator*; the solver and the mode machinery it needs
    are already there, so this is **wiring + one operator**, not new solver math.

### Why

For multi-layer SW the external gravity wave sets a punishing CFL. MASSH includes an implicit
barotropic filter that solves a Helmholtz problem for the barotropic surface mode each step and
subtracts its fast component, letting you take baroclinic-scale time steps.

### Math

Split the depth-averaged (barotropic) velocity `ū = Σ_k h_k u_k / Σ_k h_k`. The fast
surface-mode correction `w` solves a variable-coefficient Helmholtz/Poisson problem

$$
\frac{1}{g\,\tau\,\Delta t}\,\nabla\!\cdot(h\,\bar{u}^{*}) \;=\; \nabla\!\cdot(h\,\nabla w),
\qquad
\mathrm{filt}_u = -g\,\tau\,\partial_x w,\quad \mathrm{filt}_v = -g\,\tau\,\partial_y w,
$$

added back to the momentum tendency so the divergent fast part is treated implicitly. `τ` is an
implicitness/relaxation parameter.

### MASSH reference

`mapping/models/model_qgsw/sw.py:1019` — `filter_barotropic_waves`:

```python
u_bar_star = (u_star*h_tot_ugrid).sum(-3, keepdims=True)/h_tot_ugrid.sum(-3, keepdims=True)
v_bar_star = (v_star*h_tot_vgrid).sum(-3, keepdims=True)/h_tot_vgrid.sum(-3, keepdims=True)
rhs = 1./(self.g*self.dt*self.tau) * ( jnp.diff(h_tot_ugrid*u_bar_star, -2)/self.dx
                                     + jnp.diff(h_tot_vgrid*v_bar_star, -1)/self.dy )
w_surf_imp = self.helm_solver.solve(rhs, coef_ugrid, coef_vgrid)   # variable-coef Helmholtz
filt_u = -self.g*self.tau*jnp.diff(w_surf_imp, -2)
filt_v = -self.g*self.tau*jnp.diff(w_surf_imp, -1)
return dt_u+filt_u, dt_v+filt_v, dt_h
```

(Inspired by Dukowicz & Smith free-surface split, `doi:10.1029/2000JC900089`.)

### Proposed finitevolX API

A model-level operator that calls the *existing* variable-coef multigrid:

```python
# finitevolx/_src/operators/barotropic.py
def barotropic_filter(
    u_star, v_star,                  # provisional velocities (after explicit RHS)
    h_u, h_v,                        # layer thickness on u-/v-faces
    *, dx, dy, g, tau, dt,
    helm_solve: Callable,            # e.g. build_multigrid_solver(..., coeff=h).solve
) -> tuple[Array, Array]:
    """Subtract the implicit external-gravity-wave (barotropic) mode.
    Returns (filt_u, filt_v) tendency corrections. See MASSH sw.py:1019.
    The variable-coef Helmholtz ∇·(h∇w) is provided by the existing
    multigrid solver (build_multigrid_solver(coeff=...))."""
```

So this gap is **a thin operator over an existing solver**, not new elliptic machinery. (It is
the same variable-coef use-case flagged on the spectraldiffx side; finitevolX is the better home
because its multigrid already does `∇·(c∇u)`.)

-----

## 4. Linear drag and Rayleigh-sponge relaxation operators  *(real gap)*

**Status: genuine gap.** Grep for `linear_drag`, `bottom_drag`, `rayleigh`, `relaxation`,
`sponge`, `nudg`, `restoring` across `_src/` finds only (a) the multigrid Jacobi-smoother
"relaxation" weight (unrelated) and (b) a 1-D `Sponge1D` boundary condition (`boundary/bc_1d.py`)
— **no 2-D relaxation/drag tendency operator**. The diffusion module (`Diffusion2D`,
`BiharmonicDiffusion2D`) provides Laplacian/biharmonic dissipation but no linear drag.

### Why

MASSH treats **bottom drag**, **sponge/Rayleigh damping**, and **BC nudging** as first-class RHS
terms. These are needed for closed-basin spin-down, sponge boundaries, and BFN, and they are the
cross-cutting "nudging primitive" of §6.

### Math

- Linear (Rayleigh) bottom drag on the lowest layer: `∂_t u_N += −r u_N`.
- Sponge/relaxation toward a reference `X_ref` with a spatial weight `W∈[0,1]`:
  `∂_t X += −γ W (X − X_ref)`.

### MASSH reference

Bottom drag, `mapping/models/model_qgsw/sw.py:967`:

```python
def add_bottom_drag(self, du, dv, u, v):
    du = du.at[..., -1, :, :].set(du[..., -1, :, :] - self.bottom_drag_coef*u[..., -1, 1:-1, :])
    dv = dv.at[..., -1, :, :].set(dv[..., -1, :, :] - self.bottom_drag_coef*v[..., -1, :, 1:-1])
    return du, dv
```

Sponge nudging in QG1L (`mapping/models/model_qg1l/jqgm.py`): `sponge_coef * Wbc * (Xb − X)`,
with `Wbc` a boundary weight map.

### Proposed finitevolX API

```python
# finitevolx/_src/operators/relaxation.py
def linear_drag(u, v, *, coef, layer=-1) -> tuple[Array, Array]:
    """Rayleigh bottom drag −coef·u on the chosen layer (default deepest)."""

def rayleigh_relaxation(
    x: Array, x_ref: Array, *, coef: float, weight: Array
) -> Array:
    """Sponge/relaxation tendency −coef·weight·(x − x_ref).
    `weight` is a [0,1] spatial map. One operator behind sponge layers, BC
    forcing, BFN nudging, and tracer restoring. MASSH: jqgm.py sponge term."""
```

Re-export both flat in `__init__.py`. `rayleigh_relaxation` is the single operator behind sponge
layers, BC forcing, BFN nudging, and tracer restoring — define once, reuse everywhere.

!!! note "This is the operator the somax tracer-transport gap needs"
    The companion somax gap analysis flags `forcing_tracer_from_bc` and notes that no
    `rayleigh_relaxation` op exists anywhere yet. **This §4 is that operator's home.** Landing it
    here unblocks tracer-to-BC nudging in somax without per-model bespoke code.

### Example

```python
import finitevolx as fvx
du, dv = fvx.linear_drag(u, v, coef=r)
dssh   = fvx.rayleigh_relaxation(ssh, ssh_bc, coef=gamma, weight=Wbc)
```

-----

## 5. Differentiable positivity / monotonicity guards  *(real gap)*

**Status: genuine gap.** `operators/_utils.py` holds only `_safe_div_cos` (a spherical pole
guard); grep for `smooth_abs`, `smooth_clamp`, `smooth_max`, `softplus` across `_src/` finds
nothing.

### Why

Adjoints break where `jnp.maximum`, `jnp.abs`, or hard clamps appear (zero/undefined gradient).
MASSH uses smooth surrogates so the SW adjoint stays well-defined — directly relevant to the
differentiable-baseline goal. Tiny, but they belong as named, tested primitives so they are used
consistently (and `rusanov_flux` (§1) depends on `smooth_abs`).

### Math

- `smooth_abs(x) = √(x² + ε²)`  (wave speeds, WENO weights).
- `smooth_clamp(x, x_min) = x_min + softplus((x−x_min)·s)/s`  (positive layer thickness).

### MASSH reference

`sw.py:27` `smooth_clamp`, `reconstruction.py:15` `smooth_abs`:

```python
def smooth_clamp(x, x_min, sharpness=10.):
    return x_min + jax.nn.softplus((x - x_min)*sharpness)/sharpness    # grad nonzero everywhere

def smooth_abs(x):
    return jnp.sqrt(x**2 + WENO_EPS**2)
```

### Proposed finitevolX API

```python
# finitevolx/_src/operators/differentiable.py  (new; or extend operators/_utils.py)
def smooth_abs(x, eps=1e-8): ...
def smooth_clamp(x, x_min, sharpness=10.0): ...
def smooth_max(x, y, sharpness=10.0): ...      # generic two-arg variant
```

Document the gradient behavior in each docstring (the entire point). Re-export flat.

-----

## 6. Cross-cutting note: the relaxation/nudging primitive

§4 (sponge/drag), the weight map (xrtoolz), and the Gaspari–Cohn taper (gaussx) are three faces
of one concept:

$$
\text{tendency} \mathrel{+}= \text{coef}\cdot W(x)\cdot (X_{\text{ref}} - X),
$$

with `W` a smooth spatial weight. MASSH uses this exact pattern for sponge layers, BC forcing,
tracer restoring, **and** BFN. Implementing `rayleigh_relaxation` here + `compute_weight_map` in
xrtoolz + `gaspari_cohn` in gaussx closes the most gaps per unit effort and is the prerequisite
for the `BFNCycle` protocol proposed in the pipekit doc.

-----

## Suggested ordering for finitevolX

1. `smooth_abs` / `smooth_clamp` (§5) — trivial; unblocks AD-safe versions of everything else
   (and `rusanov_flux` needs `smooth_abs`).
2. `rusanov_flux` (§1) — small, immediately useful as an AD-robust continuity flux.
3. `rayleigh_relaxation` + `linear_drag` (§4) — unblocks sponge/BC/BFN (and the somax tracer
   nudging gap).
4. `vorticity_flux_upwind` + the public `pv_flux_arakawa_lamb` wrapper (§2) — MASSH parity and a
   one-call conserving entry point (the conserving *scheme* already exists).
5. `barotropic_filter` (§3) — a thin operator over the existing variable-coef multigrid.

## References

- Arakawa & Lamb (1981), *Mon. Wea. Rev.* 109, 18–36 — energy/enstrophy-conserving SW scheme
  (**already implemented**: `operators/vorticity.py:202`, `diffusion/momentum.py:35`).
- Arakawa (1966), *J. Comput. Phys.* 1 — the Jacobian (`operators/jacobian.py:22`).
- Sadourny (1975) — the E/Z schemes finitevolX's `MomentumAdvection2D` also offers.
- Dukowicz & Smith (2000), `doi:10.1029/2000JC900089` — barotropic split (MASSH's cited source).
- **finitevolX (verified, `0.0.42`):** `operators/jacobian.py`, `advection/weno.py`,
  `advection/limiters.py`, `advection/face_flux.py`, `operators/diagnostics.py`,
  `operators/vorticity.py`, `operators/divergence.py`, `solvers/multigrid.py`
  (`build_multigrid_solver(coeff=...)`), `timestepping/split_explicit.py`,
  `timestepping/explicit_rk.py` / `diffrax_solvers.py`.
- **MASSH (`VarDyn`, unverified line numbers):** `mapping/models/model_qgsw/sw.py`,
  `reconstruction.py`; `mapping/models/model_qg1l/jqgm.py`.
