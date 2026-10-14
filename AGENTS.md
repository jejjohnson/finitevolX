# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this repo is

finitevolX is finite-volume operators on staggered Arakawa C-grids in JAX:
grids and land / ocean masks, raw stencils, difference / interpolation /
advection / diffusion operators, boundary conditions, diagnostics and
forcing, elliptic and multigrid solvers, and time integrators built on
diffrax, for 1-D, 2-D, 3-D and multilayer ocean and atmosphere models.
Everything is an Equinox module or a pure function, so it runs under `jit`,
`vmap` and `grad`. Ocean models are built on it, so **finitevolX is a
library of primitives**: new code composes what is here, and anything
genuinely new lands where the next person will find it.

It is one package, `finitevolx/` at the repo root (flat layout, no `src/`).
Everything under `finitevolx/_src/` is private; the public API is the flat
`finitevolx.__all__`, re-exported from `finitevolx/__init__.py` (its order is
a logical grouping, not alphabetical: ruff's `RUF022` is off).

| Area | Path (`finitevolx/_src/`) | Owns |
|---|---|---|
| Grids | `grid/` | `ArakawaCGrid*` (topology) → `CurvilinearGrid*` (uniform metric) → `CartesianGrid1D/2D/3D`, `SphericalGrid2D/3D` (`base.py`, `cartesian.py`, `spherical.py`; `cylindrical.py` is a TODO stub) |
| Masks | `mask/` | `Mask1D/2D/3D` (`cartesian.py`): staggered wet masks, coast classification, `StencilCapability*` (`base.py`), adaptive stencil hierarchies; numpy / scipy construction helpers (`utils.py`) |
| Layer 1: raw stencils | `operators/stencils.py`, `operators/_ghost.py` | `diff_*` / `avg_*` index arithmetic with no metric and no padding; `interior` pads a result back to the full shape |
| Layer 2: functional helpers | `operators/diagnostics.py`, `operators/jacobian.py`, `operators/eos.py`, `operators/reductions.py`, `forcing/functional.py`, `advection/{weno,linear,limiters,reconstruct_fn,face_flux,flux}.py`, … | Pure functions on arrays (`kinetic_energy`, `arakawa_jacobian`, `area_mean`, `weno_5pts`, `upwind_flux`, `linear_drag_tendency`, …); they never take a `Mask2D`, except the reductions (`area_sum`, `masked_mean`, … accept `mask=`) |
| Layer 3: class operators | `operators/` (`difference.py`, `interpolation.py`, `divergence.py`, `vorticity.py`, `coriolis.py`, `diagnostic_operators.py`, `spherical_*.py`), `advection/` (`advection.py`, `reconstruction.py`, `spherical_advection.py`), `diffusion/` | `Difference*`, `Interpolation*`, `Advection*`, `Reconstruction*`, `Diffusion*`, `MomentumAdvection*`, `Vorticity*`, `Coriolis*`, `Energetics2D`, … and their `Spherical*` siblings; mask-aware through a `mask` field |
| Boundary conditions | `boundary/` | Per-face atoms `Dirichlet1D` … `Slip1D` (`bc_1d.py`), `BoundaryConditionSet` (`bc_set.py`), `FieldBCSet` (`bc_field.py`), functional ghost-ring fillers (`boundary.py`) |
| Forcing | `forcing/` | `AbstractForcing`, `ForcingSum` / `ForcingProduct` / `TimeVaryingForcing` (`_base.py`); wind, drag, Rayleigh operators |
| Solvers | `solvers/` | Spectral solvers re-exported from spectraldiffx (`spectral.py`, `spectral_transforms.py`); capacitance wrapper and the `streamfunction_from_vorticity` / `pressure_from_divergence` / `pv_inversion` wrappers (`elliptic.py`); `solve_cg` on lineax (`iterative.py`); preconditioners (`preconditioners.py`); `MultigridSolver` (`multigrid.py`); known-value lifting (`inhomogeneous.py`); tridiagonal solve re-exported from gaussx (`tridiagonal.py`) |
| Time integration | `timestepping/` | Pure step functions (`euler_step`, `rk3_ssp_step`, `ab2_step`, `imex_ssp2_step`, …) and solver classes (diffrax's `RK3SSP`, `SSP_RK104`, `IMEX_SSP2`, …; `AB2Solver`, `LeapfrogRAFSolver`, `SplitExplicitRKSolver`) in `diffrax_solvers.py`; `solve_ocean_pde` (`_solve.py`) |
| Vertical | `vertical/` | `multilayer` (vmap a 2-D operator over layers), vertical modes (`build_coupling_matrix`, `layer_to_mode`, …) |
| Shared | `utils/constants.py` | `GRAVITY`, `R_EARTH`, `OMEGA`, `RHO`, `DEG2M` (parameter defaults only; see contract 6) |

The three operator layers are the ones `docs/stencils.md` and
`docs/masks.md` describe: Layer 1 raw stencils, Layer 2 functional helpers
(metric-scaled, mask-free), Layer 3 class operators (the layer that
carries a mask as a field). The forcing modules number their functional primitives
"Layer 0"; they are the same tier as Layer 2.

`scripts/` holds example models (`swm_linear.py`, `shallow_water.py`,
`qg_1p5_layer.py`, smoke-tested by `tests/test_scripts.py`) and the docs
figure generator (`generate_solver_figures.py`); `scripts/capabilities.py`
builds the capability index.

### Boundaries

- **Upstream: spectraldiffx and gaussx**, both pinned to a release tag in
  `[tool.uv.sources]` plus a `>=` floor in `dependencies` (neither is on
  PyPI). spectraldiffx owns the spectral transforms, the rectangular
  Poisson / Helmholtz solvers, their eigenvalues and the capacitance solver;
  finitevolX re-exports 78 of its names unchanged. gaussx owns structured
  linear algebra; finitevolX re-exports its tridiagonal solves and builds
  `make_nystrom_preconditioner` on `gaussx.NystromPreconditioner`;
  `make_gaussx_multigrid_preconditioner` wraps the symmetrised V-cycle in
  `gaussx.OperatorPreconditioner` for gaussx's CG strategies. Use their
  public APIs only; neither imports finitevolX.
- **Downstream**: ocean models (e.g. somax) build on the operators and
  rebuild them in nondimensional units, which is why operators must stay
  unit-agnostic (contract 6, `tests/test_scale_covariance.py`).
- **Pure `jnp` slicing.** Stencils are written with array slices; do not
  depend on `finitediffx` or `kernex`.
- **NumPy / SciPy only at construction time**, on concrete values: mask
  construction (`Mask2D.from_mask`, `scipy.ndimage`), multigrid level
  building, irregular-boundary indices. Never on a traced value.

## What finitevolX is built on

Each foundation brings a rule. Breaking one runs fine on one example and
fails under a transform, in float32, near a coast, or in someone else's
model.

| Library | finitevolX uses it for | The rule it brings |
|---|---|---|
| **jax** | Arrays, `jit`, `vmap`, `grad`, `lax.fori_loop` (multigrid V-cycles) | Pure functions; `.at[...].set` instead of mutation; no Python control flow on traced values; build outputs with the input's dtype (`interior` uses `jnp.result_type`). |
| **equinox** | Every grid, mask, operator, BC, forcing and solver is an `eqx.Module`; `eqx.filter_vmap` (`multilayer`) | Never a dataclass or plain class for anything holding arrays. Settings that pick a code path are `eqx.field(static=True)` (e.g. `Dirichlet1D.face`). An operator holds its grid, its mask and any precomputed hierarchy as fields, built once in `__init__`. |
| **jaxtyping** | Shape annotations (`Float[Array, "Ny Nx"]`) | Annotate public array arguments and returns; shape strings stay quoted (`UP037` and `F722` are off for that reason). They are documentation: nothing checks them at run time (beartype is a declared dependency but is not imported). |
| **lineax** | `solve_cg` (`lx.CG` on a `FunctionLinearOperator`) | Elliptic operators here are negative (semi)definite: tag them `negative_semidefinite_tag`; a preconditioner approximates `A⁻¹` and `solve_cg` flips its sign for lineax. |
| **diffrax** | Butcher-tableau solvers (`dfx.AbstractERK`), custom `dfx.AbstractSolver`s, `solve_ocean_pde` | A new integrator is a diffrax solver (or a pure step function); forcings live inside the vector field, not as extra diffrax terms. |
| **spectraldiffx** | DCT / DST / FFT, spectral Poisson / Helmholtz solvers, eigenvalues, capacitance solver | Re-export or call; never re-implement a spectral solve or its eigenvalues. |
| **gaussx** | Tridiagonal solves, Nyström preconditioning | Structured solves and low-rank preconditioners come from gaussx, not a hand-written `jnp.linalg` routine. |
| **numpy / scipy** | Construction-time mask and multigrid precomputation | Concrete values only (see Boundaries). |

## Reuse before you write

Before writing a stencil, an operator, a BC, a solver or a helper, find out
whether it exists:

1. **Search the capability index.** [`docs/capabilities.md`](docs/capabilities.md)
   lists every name in `finitevolx.__all__`, grouped by the module that
   defines it (re-exports grouped by library), with a one-line summary; then
   the public API of spectraldiffx and gaussx. It is generated
   (`make capabilities`) and checked by `tests/test_capabilities.py`.
2. **Read the theory page for the area** (see "Documentation" below): most
   operator families document the stencil, the ghost cells they read and the
   mask semantics there.
3. **If it is missing, add it at its layer**, in the subpackage that owns the
   concept: a raw index pattern in `operators/stencils.py`, a metric-scaled
   function next to its family, a mask-aware class over both. A helper two
   modules need goes into the owning module (or `operators/_ghost.py`,
   `forcing/_utils.py`, `operators/_utils.py`), not inline in one caller.
4. **One object, one name.** A finitevolX name never means something
   different from the same name in spectraldiffx or gaussx; the deliberate
   exceptions are listed with their reason in `ALLOWED_SHARED_NAMES` in
   `scripts/capabilities.py`.

| You are about to write… | Use instead |
|---|---|
| `h[1:-1, 2:] - h[1:-1, 1:-1]`, `0.5 * (h[1:-1, 1:-1] + h[2:, 1:-1])` | the raw stencils `diff_x_fwd`, `avg_y_fwd`, … (and their `_1d` / `_3d` forms) |
| `out = jnp.zeros_like(h); out = out.at[1:-1, 1:-1].set(...)` | `interior(values, like)` (`ghost=2` for advection) |
| A gradient, divergence, curl or Laplacian on the C-grid | `Difference1D/2D/3D` (`diff_x_T_to_U`, `divergence`, `curl`, `laplacian`, `grad_perp`), `Divergence2D`, `Vorticity2D`; on the sphere `SphericalDifference2D/3D`, `SphericalDivergence*`, `SphericalLaplacian*`, `SphericalVorticity*` |
| Moving a field between T / U / V / X points | `Interpolation1D/2D/3D` (`T_to_U`, `U_to_T`, `T_to_X`, `U_to_V`, …) |
| A flux-form advective tendency, upwind / WENO / TVD face values | `Advection1D/2D/3D(...)(h, u, v, method=...)`, `Reconstruction1D/2D/3D`, `reconstruct`, `upwind_flux`, the `weno_*` / `linear_*` stencils, the limiters `minmod` / `van_leer` / `superbee` / `mc`, `rusanov_flux` |
| Harmonic / biharmonic diffusion, momentum advection, Coriolis | `Diffusion2D/3D`, `BiharmonicDiffusion2D/3D`, `MomentumAdvection2D/3D`, `Coriolis2D/3D` (and `Spherical*`) |
| Kinetic energy, PV, enstrophy, strain, Okubo–Weiss, an Arakawa Jacobian | `Energetics2D`, `Vorticity2D`, `Strain2D`, `QGPotentialVorticity2D`, `ArakawaJacobian2D`; functional `kinetic_energy`, `arakawa_jacobian`, `okubo_weiss`, … |
| Filling ghost cells | `BoundaryConditionSet` of `Dirichlet1D` / `Neumann1D` / `Periodic1D` / …; `enforce_periodic`, `zero_gradient_boundaries`, `no_slip_boundaries`, `wall_boundaries` |
| Land / ocean masks, coastal stencil fallback | `Mask2D.from_mask` (or `from_center`, `from_u_face`, …), `mask.get_adaptive_masks`, `StencilCapability2D` |
| Area / volume integrals and masked means | `area_sum`, `area_mean`, `volume_mean`, `masked_mean`, `masked_moments` (grid-polymorphic; `cartesian_*` / `spherical_*` explicit) |
| A Poisson / Helmholtz solve | `solve_helmholtz_dst` / `_dct` / `_fft` (spectraldiffx) on a rectangle; `build_capacitance_solver`, `solve_cg` + `make_preconditioner`, or `build_multigrid_solver` on a masked domain; `streamfunction_from_vorticity`, `pressure_from_divergence`, `pv_inversion` as one-call wrappers |
| A DCT / DST | `dct`, `dst`, `dctn`, … (spectraldiffx, scipy's unnormalised convention) |
| A tridiagonal (Thomas) solve | `solve_tridiagonal`, `solve_tridiagonal_batched` (gaussx) |
| A CG loop or a Nyström preconditioner | `solve_cg`, `make_nystrom_preconditioner`, `make_spectral_preconditioner`, `make_multigrid_preconditioner` |
| An RK / multistep / IMEX time loop | `rk3_ssp_step`, `ab2_step`, `imex_ssp2_step`, … or the diffrax solvers with `solve_ocean_pde` |
| Applying a 2-D operator to every layer | `multilayer(fn)`; vertical modes with `decompose_vertical_modes`, `layer_to_mode` |
| Wind stress, bottom drag, sponge damping | `WindStress2D`, `LinearDrag2D`, `QuadraticDrag2D`, `RayleighDamping2D` (and `3D`), composed with `ForcingSum` / `ForcingProduct` |
| `jnp.abs` / `jnp.maximum` / `jnp.clip` on a path you differentiate | `smooth_abs`, `smooth_max`, `smooth_clamp` |
| Golden-output or input fixtures in a test | `tests/fixtures/inputs.py` (`make_grid_2d`, `make_mask_2d`, `make_h_field_2d`, …), `tests/fixtures/_helpers.py` (`assert_matches_golden`) |

## The contracts

The long form, with figures, is in `docs/cgrid_discretization.md`,
`docs/stencils.md`, `docs/masks.md` and `docs/boundary_conditions.md`.

### 1. The C-grid: shapes, staggering and ghost cells

- All arrays have total shape `[Ny, Nx]` (`[Nx]` in 1-D, `[Nz, Ny, Nx]` in
  3-D). The physical interior is `[1:-1, 1:-1]`; one ghost ring on each side
  is reserved for boundary conditions. Grids are built with
  `CartesianGrid2D.from_interior(nx, ny, Lx, Ly)`, which adds the ring.
- Same-index colocation:
  - `T[j, i]` → cell centre `(j, i)`
  - `U[j, i]` → east face `(j, i+1/2)`
  - `V[j, i]` → north face `(j+1/2, i)`
  - `X[j, i]` → NE corner `(j+1/2, i+1/2)`
- In 3-D only the horizontal axes are staggered; `k = 0` and `k = Nz-1` are
  ghost slabs. Multilayer fields `[nl, Ny, Nx]` have **no** ghost layers:
  lift a 2-D operator with `multilayer` rather than using a `3D` operator
  (`docs/multilayer_vs_3d.md`).

All four types share the `[Ny, Nx]` shape, but the meaning of the ghost ring
differs ("BC-owned" cells must be filled by the caller before a chained
operator reads them; "outside domain" cells have no physical meaning):

| Type | West ghost `[:,0]` | East ghost `[:,Nx-1]` | South ghost `[0,:]` | North ghost `[Ny-1,:]` |
|------|--------------------|-----------------------|---------------------|------------------------|
| T    | west BC T-cell     | east BC T-cell        | south BC T-cell     | north BC T-cell        |
| U    | west boundary face (BC-owned) | outside domain | south ghost U-row (BC-owned) | north ghost U-row (BC-owned) |
| V    | west ghost V-col (BC-owned)  | east ghost V-col (BC-owned) | south boundary face (BC-owned) | outside domain |
| X    | west ghost X-col (BC-owned)  | outside domain        | south ghost X-row (BC-owned) | outside domain |

Key asymmetry of the same-index convention:

- **Forward operators** (T→U, T→V, T→X) write to `[1:-1, 1:-1]`; the last
  interior column / row reads the `+`-direction ghost of the source (east
  ghost T for T→U, north ghost T for T→V, the NE corner ghost for T→X).
- **Backward operators** (U→T, V→T, X→T) write to `[1:-1, 1:-1]`; the first
  interior column / row reads the `−`-direction ghost of the source (west
  ghost U-face for U→T, south ghost V-face for V→T). No forward operator
  sets that ghost: the caller must, via a BC.
- **Cross operators** (U→V, V→U, U→X, V→X, X→U, X→V) follow the same
  pattern: the last interior output reads the far-side ghost of the source.

Stencil slices, each the same shape as `[1:-1, 1:-1]`:

| Slice pattern | Meaning (j = row, i = col) |
|---|---|
| `arr[1:-1, 1:-1]` | interior at `(j, i)` |
| `arr[1:-1, 2:]` | one step east `i+1` |
| `arr[1:-1, :-2]` | one step west `i-1` |
| `arr[2:, 1:-1]` | one step north `j+1` |
| `arr[:-2, 1:-1]` | one step south `j-1` |

```
# forward  diff_x_T_to_U:  dh[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx
interior((h[1:-1, 2:] - h[1:-1, 1:-1]) / dx, h)
# forward  diff_y_U_to_X:  du[j+1/2, i+1/2] = (u[j+1, i] - u[j, i]) / dy
interior((u[2:, 1:-1] - u[1:-1, 1:-1]) / dy, u)
# backward diff_x_U_to_T:  du[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx
interior((u[1:-1, 1:-1] - u[1:-1, :-2]) / dx, u)
# backward diff_y_X_to_U:  dq[j, i+1/2] = (q[j+1/2, i+1/2] - q[j-1/2, i+1/2]) / dy
interior((q[1:-1, 1:-1] - q[:-2, 1:-1]) / dy, q)
```

Without BCs, `diff_x_U_to_T` at `i = 1` sees a zero west ghost U-face and
returns `(u[j, 1] - 0) / dx`. That is the **expected** behaviour, not a bug:
apply BCs to intermediate fields before chaining operators.

### 2. Operators: the interior idiom and the three layers

- **Write only the interior.** An operator returns an array of the input's
  shape, written only on `[1:-1, 1:-1]` (via `interior(values, like)`), with
  a zero ghost ring. The caller owns boundary conditions.
- **Advection is the exception.** `Advection1D/2D` write `[2:-2]` /
  `[2:-2, 2:-2]` (`interior(..., ghost=2)`) under the default
  `wall="closed"`, so they never read ghost flux cells; `Advection3D` writes
  `[1:-1, 2:-2, 2:-2]` (z-levels are independent). Never widen the
  closed-wall horizontal write region to `[1:-1, ...]`.
- **Closed walls by default.** Advection writes the tendency only on the
  strict interior and treats the four domain walls as no-flux; diffusion
  leaves the wall-face fluxes at zero. `wall="open"` on `Advection1D/2D` and
  `Diffusion2D/3D` instead computes the wall faces from the caller-filled
  ghost ring (first-order upwind for advection, `κ (h_edge − h_ghost) / dx`
  for diffusion; open advection then writes the full interior), so
  Dirichlet / outflow / periodic BCs drive transport while the flux stays
  single-valued (mass is conserved exactly). It raises
  `NotImplementedError` together with a mask.
- **Layer 1 raw stencils** are pure index arithmetic on the full array,
  returning interior-sized results with no metric and no padding; the
  caller scales by any metric (`dx`, `R cos φ dλ`, …) and pads with
  `interior`.
- **Layer 2 functions** apply the metric and take no `Mask2D` (the
  reductions' optional `mask=` aside).
- **Layer 3 classes** are `eqx.Module`s holding `grid` and `mask` (default
  `None`), with any precomputation (adaptive-stencil hierarchies, an inner
  `Interpolation2D`, …) done once in `__init__`. Cartesian operators are
  annotated with `CartesianGrid*` (coordinate-agnostic ones such as
  `Interpolation*` and `Reconstruction*` accept any grid of their
  dimension); spherical operators take `SphericalGrid*` and are separate
  `Spherical*` classes that share the raw stencils.
- **Comment every stencil with its half-index formula** and track shapes in
  comments (`# [Ny, Nx]`).
- **Write stencils that lower to one fused pass**, forward and VJP: slice
  each offset once (bind the views, or `_neighbours(h)` for the 3x3
  neighbourhood) instead of re-slicing per sub-expression; don't build a
  full-size `zeros().at[...].set()` intermediate only to slice it again
  (compute the faces you need and `pad` the result); `_materialize` an
  intermediate the next stencil reads at several offsets. Keep the
  composed form's arithmetic and name it in a comment so readers see the
  equivalence. `docs/performance.md` has the rules and the measurements;
  benchmark with `scripts/benchmark_operators.py`.

### 3. Masks

- Build masks once, eagerly, from a binary T-point mask:
  `Mask2D.from_mask(h_mask)` derives `h`, `u`, `v`, `xy_corner` (lenient),
  `xy_corner_strict`, the coast classification and the stencil capability
  with numpy / scipy. `True` = wet. The irregular-boundary index arrays have
  a data-dependent shape: do not use them inside `jit`.
- A class operator with a mask zeroes its output on the dry points **of its
  output stagger** (`mask.h` for T, `mask.u` for U, `mask.v` for V,
  `mask.xy_corner_strict` for X). `mask=None` must reproduce the unmasked
  result bit for bit, and an all-ocean mask must equal `mask=None`.
- Mask-dispatchable advection methods (`weno3`, `weno5`, `wenoz5`, the TVD
  limiters) fall back to narrower stencils near the coast through
  `upwind_flux` and the `(2, 4, 6)` adaptive hierarchy; other methods run
  unmasked and multiply the tendency by `mask.h`.
- Where land may hold `NaN` (e.g. `Mask2D.from_center`), zero the dry
  inputs with `jnp.where` before any stencil reads them **and** mask the
  outputs with `jnp.where` (`forcing/_utils.mask_where`): multiplying by a
  float mask keeps `NaN * 0 = NaN`, and masking only the output still
  leaves a `NaN` gradient at the dry points.
- Branch on an optional mask with `if mask is not None:`, never `if mask:`
  (a JAX array raises).

### 4. Boundary conditions

- A BC atom is an `eqx.Module` with a static `face` (`"south" | "north" |
  "west" | "east"`) whose `__call__(field, dx, dy)` returns the field with
  that one ghost row / column set; it is listed in the `BoundaryCondition1D`
  union in `boundary/bc_1d.py`.
- `BoundaryConditionSet(south=, north=, west=, east=)` applies them in
  south, north, west, east order (west / east overwrite the corners);
  `FieldBCSet` maps state-dict keys to sets. Ghost values are placed so the
  condition holds **at the face**, half a cell from the interior point (e.g.
  Dirichlet: `ghost = 2 * value - interior`).
- The elliptic wrappers accept a `BoundaryConditionSet` as `bc=` only when
  all four faces are `Dirichlet1D` (→ DST), zero-valued `Neumann1D` (→ DCT)
  or `Periodic1D` (→ FFT); non-zero Dirichlet values become known values via
  lifting (`solvers/inhomogeneous.py`).

### 5. Solvers

- Rectangular spectral solves come from spectraldiffx; finitevolX re-exports
  them. Masked domains use the capacitance method, preconditioned CG
  (`solve_cg`) or multigrid; `docs/solver_comparison.md` says which when.
- Sign convention: the Helmholtz operator `(∇² − λ)` with `λ ≥ 0` is
  negative (semi)definite. A matvec passed to `solve_cg` must be that
  operator; a preconditioner approximates its (negative) inverse.
- Report convergence, don't assume it: `solve_cg` returns `CGInfo`
  (`iterations`, `residual_norm`, `converged`); check `converged` in tests.
  `solve_cg` returns rather than raises when CG does not converge
  (`throw=False`); the `method="cg"` path of the convenience wrappers
  discards `CGInfo` and raises instead, so call `solve_cg` directly when you
  want to inspect convergence.
- Randomised construction takes a `key` (`make_nystrom_preconditioner`).
- Pre-build what depends only on the geometry (`build_capacitance_solver`,
  `build_multigrid_solver`) and reuse it across right-hand sides.

### 6. Time integration, forcing and units

- A step function is pure and works on any pytree state. The explicit RK
  steps are `step(state, rhs_fn, dt) -> state`; the others take what their
  scheme needs (`rhs_nm1` for `ab2_step`, `state_nm1` for
  `leapfrog_raf_step`, explicit and implicit RHS plus an `implicit_solve`
  for `imex_ssp2_step`) and return the history the next step needs.
- A diffrax solver subclasses `dfx.AbstractERK` with a `ButcherTableau`
  when it is an explicit Runge–Kutta scheme (`RK3SSP`, `SSP_RK104`, …),
  `dfx.AbstractRungeKutta` + `dfx.AbstractImplicitSolver` for IMEX
  (`IMEX_SSP2`), or implements `dfx.AbstractSolver` (`ForwardEulerDfx`,
  `SemiLagrangianSolver`). Schemes that do not fit `diffeqsolve` are
  `eqx.Module`s with their own `step` (plus an `init` that bootstraps the
  history for `AB2Solver` and `LeapfrogRAFSolver`; `SplitExplicitRKSolver`
  substeps the fast mode).
- A forcing subclasses `AbstractForcing`; its `__call__` returns a tendency
  (an array or a tuple such as `(du, dv)`) with its own physics-specific
  arguments. Forcings are time-agnostic and compose with `ForcingSum`,
  `ForcingProduct` and `TimeVaryingForcing` inside the RHS function. The
  core math lives in a functional primitive (`forcing/functional.py`).
- **Operators are unit-agnostic.** Physical constants (`GRAVITY`,
  `R_EARTH`, `OMEGA`, `RHO`, `DEG2M`) appear only as overridable parameter
  defaults, never inside an expression, in `operators/`, `advection/`,
  `diffusion/` and `solvers/`; rescaling the grid by `L` and the fields by
  their amplitudes must rescale the output by the dimensionally correct
  power and nothing else (no absolute epsilons guarding a ratio).

### 7. JAX numerics

- No Python `if` / `bool()` / `float()` on traced values; a mask or option
  that selects a code path is a concrete Python value or a static field.
- Dtypes follow the input: float32 in gives float32 out even with x64 on.
- Divisions and square roots that can hit zero are guarded with
  `jnp.where` on the argument as well as the result
  (`forcing/_utils.safe_speed`, `safe_denominator`), so the gradient stays
  finite too. Spherical divisions by `cos(lat)` go through
  `operators/_utils._safe_div_cos`, which returns `NaN` (not `inf`) at
  pole points, where the metric is degenerate.
- Non-smooth operations on a differentiated path use the smooth surrogates
  (`smooth_abs`, `smooth_max`, `smooth_clamp`).

### The public API

- Export a new public name from `finitevolx/__init__.py` (import plus an
  entry in `__all__`, in its logical group), give it a `::: finitevolx.Name`
  entry on its page in `docs/api/` (no test checks this), and run
  `make capabilities`.
- NumPy-style docstrings: the formula (half-index notation in ASCII or
  Unicode), parameters and returns with shapes, and an `Examples` section for
  new public names. Doctests are **not** collected (no `--doctest-modules`),
  so run an example by hand before committing it.

## What enforces them

There are no test-tier markers: CI runs the whole of `tests/`. Read a test's
docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `tests/test_scale_covariance.py` | Operators are unit-agnostic (length and amplitude rescaling, masked and unmasked, Cartesian and spherical); `TestConstantsAudit` fails on a physical constant used outside a parameter default; the WENO / TVD absolute-epsilon departures are pinned (`xfail`, #243) |
| `tests/test_*_masks.py` + `tests/fixtures/` | Golden-output regression for every mask-aware operator: unmasked and masked outputs match the committed `tests/fixtures/golden/*.npz`, all-ocean equals unmasked, dry points are exactly zero |
| `tests/test_fixture_smoke.py` | The golden-fixture inputs, masks and helpers themselves |
| `tests/test_convergence.py` | Second-order spatial convergence of the difference operators |
| `tests/test_jax_transforms.py`, `tests/test_differentiable.py` | `jit` / `vmap` match eager execution; the smooth surrogates differentiate |
| `tests/test_dtype_behavior.py` | float32 and float64 agree; outputs are finite on extreme inputs |
| `tests/test_open_walls.py`, conservation tests in `test_advection.py`, `test_jacobian.py`, `test_momentum.py`, `test_diagnostics.py` | Mass / energy / enstrophy conservation of the flux-form and Arakawa schemes, including `wall="open"` |
| `tests/test_stencils.py`, `tests/test_ghost.py`, `tests/test_grid.py`, `tests/test_cgrid_mask.py` | Raw stencil slices, the ghost-ring helpers, grid and mask construction |
| `tests/test_stencil_lowering.py` | The performance rewrites equal their composed forms, and their VJPs have one `pad` per distinct offset (a JAX-level check, no timing) |
| `tests/test_elliptic*.py`, `tests/test_multigrid.py`, `tests/test_solver_wrappers.py`, `tests/test_inhomogeneous.py` | Solvers against analytic solutions, preconditioner benefit, BC-set dispatch, known-value lifting |
| `tests/test_timestepping*.py`, `tests/test_diffrax_solvers.py` | Convergence order and accuracy of the step functions and the diffrax solvers; the pure steps match their diffrax twins (`TestPureFunctionalVsDiffrax`) |
| `tests/test_scripts.py` | The example models in `scripts/` run stably on small grids |
| `tests/test_capabilities.py` | `docs/capabilities.md` is current; no name shadows a spectraldiffx / gaussx name with a different object |
| `tests/test_plugin_skill.py` | The downstream plugin's worked example runs and its stated results hold; every `fvx.X` / `finitevolx.X` the plugin names exists |
| `pyproject.toml` `filterwarnings = ["error", ...]`, `xfail_strict = true` | Any warning other than a `DeprecationWarning` fails a test; an `xfail` that starts passing fails too |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them):

| Job | Recipe |
|---|---|
| Add a raw stencil, functional helper or mask-aware class operator (difference, interpolation, vorticity, diffusion, diagnostic, spherical) | `add-stencil-operator` |
| Add a reconstruction scheme, WENO variant, flux limiter or face flux | `add-advection-scheme` |
| Add a boundary-condition atom, BC set or ghost-ring filler | `add-boundary-condition` |
| Add a forcing term (wind, drag, damping, relaxation) | `add-forcing` |
| Add an elliptic solver path, preconditioner or multigrid option | `add-elliptic-solver` |
| Add a time integrator (step function and diffrax solver) | `add-time-integrator` |
| Add or update an example notebook | `add-notebook` |
| Bump spectraldiffx / gaussx | `bump-upstream-pins` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `finitevolx-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `grid-numerics-reviewer.md`) |
| Write a squash commit message | `squash-commit` |

Downstream users get finitevolX's guidance through the Claude Code plugin
in `plugins/finitevolx/` (published by `.claude-plugin/marketplace.json`)
and `docs/llms.txt` (MkDocs copies it to the site root); see
`docs/agents.md`. When the public API or the headline usage changes, update
`plugins/finitevolx/skills/finite-volume-models-with-finitevolx/` too:
`tests/test_plugin_skill.py` runs its worked example and checks that every
`fvx.X` / `finitevolx.X` it and the plugin reviewer name still exists.

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`,
`ls` and other non-Python commands need no `uv run`.

```bash
make install              # uv sync --all-extras + pre-commit hooks
make test                 # uv run pytest tests (every test)
make test-cov             # the same with coverage (reports/coverage.xml)
make uv-lint              # ruff check . + ruff format --check . + ty (mirrors CI)
make uv-format            # ruff format . && ruff check --fix .
make typecheck            # uv run ty check finitevolx
make ci                   # lint, format check, ty, tests
make capabilities         # regenerate docs/capabilities.md
make docs                 # uv run mkdocs build
make docs-serve           # local MkDocs preview
make uv-pre-commit        # every pre-commit hook on every file
```

Run one test from the repo root (the `tests` package imports
`tests.fixtures`): `uv run pytest tests/test_difference.py -k T_to_U -v`.
The suite is large: `uv run pytest tests -n auto` (pytest-xdist, in the
`test` extra) is much faster locally.

### Test tiers

- There are **no markers**: no `slow`, no `integration`. CI (`ci.yml`) runs
  `uv run pytest tests -v` on Ubuntu and macOS, Python 3.12 and 3.13, after
  `uv sync --extra test`. Keep each new test small (grids of 8–32 interior
  cells) so the suite stays fast.
- One test class per operator and dimension, with pytest fixtures for the
  grids; analytic checks (constant → 0, linear → constant, convergence
  order, conservation) plus golden outputs for mask-aware operators.
- There is no `conftest.py`. Most test modules (and `tests/fixtures/`)
  call `jax.config.update("jax_enable_x64", True)` at import, so a full run
  is in float64; a new module that needs float64 must do the same rather
  than rely on another module being collected first.
- `filterwarnings = "error"`: fix a warning rather than silencing it, or
  filter it in the one test that expects it.

### Tests that assert on random draws

- Randomness here is incidental (random input fields, Nyström probes): pin
  it with `jax.random.PRNGKey(n)` or `np.random.default_rng(n)` in the test.
- A tolerance tuned on one draw must hold for a few other seeds; say in a
  comment where a non-obvious tolerance comes from (truncation order, solver
  `rtol`, float32 epsilon).

### Before every commit

All of these must pass, from the repo root:

1. `make test` (or `uv run pytest tests -n auto`): zero failures.
2. `uv run ruff check .`: the **entire** repo, including `tests/` and
   `scripts/` (CI runs `ruff check .`; a subdirectory misses their import
   sorting). `docs/` and `notebooks/` are excluded by the ruff config.
3. `uv run ruff format --check .`
4. `uv run ty check finitevolx` (`make typecheck`): no errors.
5. After changing a public API: `make capabilities`, and the `docs/api` page.
6. After changing an operator's math on purpose: regenerate the goldens
   (`uv run python tests/fixtures/_gen_golden.py`) and explain the diff in
   the PR.
7. After changing a dependency: `uv lock`, and commit `uv.lock`.
8. After changing docs or docstrings: `uv run mkdocs build` (what
   `pages.yml` runs) and check that the log has no new `WARNING` lines.
   `--strict` does not pass yet: the existing docstrings raise about 115
   griffe warnings, so compare against `main` rather than expecting zero.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; push back when a
   simpler approach exists; ask when unsure.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions, no configurability
   nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change;
   mention unrelated dead code rather than deleting it; remove only what
   your change made unused.
4. **Goal-driven.** Turn the task into a check (a failing test, a
   reproduced bug, an analytic solution or convergence order to match) and
   loop until it passes. For a multi-step task, state the plan as steps,
   each with its check.

Also: type hints on every public function, NumPy-style docstrings,
`eqx.Module` (not dataclasses) for anything that holds arrays, and
`from __future__ import annotations` in new modules.

## Git, commits and pull requests

- Never push to or merge into `main` / `master` unless explicitly told to
  ("push to main", "merge to main"). Never run `git push` unless asked;
  commit locally on a feature branch. "Merge the branch" means push the
  feature branch, not merge into `main`. Confirm before anything that
  affects a shared branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/)
  (`<type>[optional scope]: <description>`) with a lowercase subject
  (`feat(advection): add …`); CI validates PR titles. Types: `feat`, `fix`,
  `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`,
  `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** Read it
  first and treat it as the base; only append checklist items or update
  their status. (An agent making a small follow-up change must not replace
  the description with one scoped to its own work.)
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=jejjohnson -f repo=finitevolX -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

MkDocs + Material + mkdocstrings (NumPy docstring style) + mkdocs-jupyter,
configured in `mkdocs.yml`; `pages.yml` runs `mkdocs gh-deploy` on every
push to `main` (no workflow builds the docs on a pull request, so build them
yourself). If `pandoc` is installed locally, mkdocs-jupyter's jupytext probe
of every `.md` page can hang on the `::: finitevolx.Name` blocks; build with
`pandoc` off your `PATH`. The nav pairs a theory page with an API page per topic:

| Topic | Theory / usage | API |
|---|---|---|
| Grid, staggering, ghost cells | `docs/cgrid_discretization.md` | `docs/api/grids.md` |
| Masks | `docs/masks.md` | `docs/api/masks.md` |
| Multilayer vs 3-D | `docs/multilayer_vs_3d.md` | `docs/api/vertical_modes.md` |
| Raw stencils, the three layers | `docs/stencils.md` | `docs/api/stencils.md` |
| Difference / interpolation | `docs/difference.md`, `docs/interpolation.md` | `docs/api/difference.md`, `docs/api/interpolation.md` |
| Advection and reconstruction | `docs/advection.md` | `docs/api/advection.md`, `docs/api/reconstruction.md` |
| Boundary conditions | `docs/boundary_conditions.md` | `docs/api/boundary_conditions.md` |
| Divergence, vorticity, diffusion, Coriolis, forcing, momentum, diagnostics | `docs/spatial_operators.md` | `docs/api/{divergence,vorticity,diffusion,coriolis,forcing,momentum,diagnostics}.md` |
| Time integration | `docs/time_integration.md`, `docs/time_integration_usage.md` | `docs/api/time_integration.md` |
| Elliptic solvers | `docs/elliptic_solvers.md`, `docs/elliptic_solvers_usage.md`, `docs/solver_comparison.md` | `docs/api/{elliptic,spectral_solvers,spectral_transforms,capacitance,iterative}.md` |
| Multigrid | `docs/multigrid_solver.md`, `docs/multigrid_solver_usage.md` | `docs/api/multigrid.md` |
| Every public name | `docs/capabilities.md` (generated) | |

Writing style:

- NumPy-style docstrings for every public function and class.
- Be pedantic and pedagogical: every stencil carries a comment with its
  half-index formula, in ASCII math (`dh_dx[j, i+1/2] = ...`), and array
  shapes are tracked in comments (`[Ny, Nx]`).
- Theory pages use MathJax (`pymdownx.arithmatex`); admonitions with `!!!`.

Example notebooks live in `docs/notebooks/` as **jupytext percent-format
`.py` files**, rendered by mkdocs-jupyter with `execute: false`, so the
rendered page shows no cell outputs. Figures are saved by the notebook
itself (`matplotlib.use("Agg")`, `fig.savefig(IMG_DIR / "name.png")` into
`docs/images/<notebook>/`) and embedded from a markdown cell
(`![...](../../images/<notebook>/name.png)`); regenerate them by running the
script (`uv run python docs/notebooks/<name>.py`) and commit the PNGs. Add
each notebook to the `Examples` section of the `mkdocs.yml` nav. The
double-gyre models in `scripts/` write their GIFs and Zarr stores to
`--output-dir`; `docs/examples.md` shows the committed GIFs.

Research and design notes (boundary-condition design, gap analyses) live in
`docs/design/`, outside the nav.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never
committed); track work in GitHub issues.
