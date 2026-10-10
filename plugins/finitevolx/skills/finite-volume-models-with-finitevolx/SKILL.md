---
name: finite-volume-models-with-finitevolx
description: Write finite-volume ocean and atmosphere models in JAX on finitevolX — Arakawa C-grids with a ghost ring (Cartesian and spherical, 1-D / 2-D / 3-D, multilayer), raw stencils and difference / interpolation / divergence / vorticity operators, WENO / TVD / upwind advection, diffusion, Coriolis and momentum advection, land-sea masks with coastal stencil fallback, boundary conditions, energy / PV / strain diagnostics, wind and drag forcing, spectral / capacitance / CG / multigrid Poisson and Helmholtz solvers, and SSP-RK, multistep, IMEX and diffrax time steppers. Use whenever a task discretises a PDE on a staggered grid, writes a stencil or a ghost-cell fill, advects a tracer, inverts vorticity or PV, or time-steps a shallow-water / QG / primitive-equation model, in a project that uses (or could use) finitevolX.
---

# Finite-volume models on finitevolX

finitevolX gives every piece of a C-grid model as an Equinox module or a
pure JAX function: grids that carry the ghost ring, stencils that know
which face they map to, masks that switch stencils off at the coast,
boundary conditions that fill ghosts at the right half-cell, solvers and
time steppers. Everything runs under `jit`, `vmap` and `grad`. Before
writing a shifted slice, a ghost-cell fill, a WENO reconstruction, a
Poisson solver or an RK loop, look it up:

1. **The capability index** lists every public name with a one-line
   summary, plus the spectraldiffx and gaussx APIs underneath:
   <https://jejjohnson.github.io/finitevolX/capabilities/>. Or list the
   installed version:

   ```python
   import inspect

   import finitevolx

   for name in finitevolx.__all__:
       doc = (inspect.getdoc(getattr(finitevolx, name)) or "").split("\n")[0]
       print(f"finitevolx.{name}: {doc}")
   ```

2. Compose what exists. finitevolX is not on PyPI:
   `uv add "finitevolx @ git+https://github.com/jejjohnson/finitevolX.git"`.
   uv also honours the release tags finitevolX pins for its two git-only
   dependencies, spectraldiffx and gaussx; with pip, install those two from
   GitHub at the same tags first.

## What lives where

| You need… | Use |
|---|---|
| A grid | `fvx.CartesianGrid1D/2D/3D.from_interior(...)`, `fvx.SphericalGrid2D/3D.from_interior(...)`; arrays are `[Ny, Nx]` including one ghost ring, interior `[1:-1, 1:-1]` |
| Raw shifted differences / averages | `fvx.diff_x_fwd`, `fvx.diff_y_bwd`, `fvx.avg_xy_fwd`, … (+ `_1d` / `_3d`); pad back with `fvx.interior(values, like)` |
| Gradients, divergence, curl, Laplacian | `fvx.Difference1D/2D/3D` (`diff_x_T_to_U`, `divergence`, `curl`, `laplacian`, `grad_perp`), `fvx.Divergence2D`, `fvx.Vorticity2D`; on the sphere `fvx.SphericalDifference2D`, `fvx.SphericalDivergence2D`, `fvx.SphericalLaplacian2D`, `fvx.SphericalVorticity2D` |
| Moving fields between T / U / V / X points | `fvx.Interpolation1D/2D/3D` (`T_to_U`, `U_to_T`, `T_to_X`, …) |
| Tracer advection | `fvx.Advection1D/2D/3D(grid, mask)(q, u, v, method="weno5")` (`upwind1/2/3`, `weno3/5/7/9`, `wenoz5`, `minmod`, `van_leer`, `superbee`, `mc`); `fvx.SphericalAdvection2D` |
| Diffusion, momentum advection, Coriolis | `fvx.Diffusion2D/3D`, `fvx.BiharmonicDiffusion2D/3D`, `fvx.MomentumAdvection2D/3D`, `fvx.Coriolis2D/3D` |
| Land and coasts | `fvx.Mask2D.from_mask(wet)` → staggered masks `h` / `u` / `v` / `xy_corner_strict`; pass `mask=` to an operator |
| Ghost cells | `fvx.BoundaryConditionSet(south=fvx.Dirichlet1D("south", 0.0), ...)`, `fvx.enforce_periodic`, `fvx.zero_gradient_boundaries`, `fvx.no_slip_boundaries`, `fvx.wall_boundaries` |
| Diagnostics | `fvx.Energetics2D`, `fvx.Strain2D`, `fvx.QGPotentialVorticity2D`, `fvx.ArakawaJacobian2D`; `fvx.kinetic_energy`, `fvx.okubo_weiss`, `fvx.arakawa_jacobian`, … |
| Integrals and means | `fvx.area_sum`, `fvx.area_mean`, `fvx.volume_mean`, `fvx.masked_mean` |
| Poisson / Helmholtz | `fvx.streamfunction_from_vorticity`, `fvx.pressure_from_divergence`, `fvx.pv_inversion` (`bc="dst"` / `"dct"` / `"fft"`; `method="cg"` / `"capacitance"` on masked domains); `fvx.solve_cg`, `fvx.make_preconditioner`, `fvx.build_capacitance_solver`, `fvx.build_multigrid_solver` |
| Forcing | `fvx.WindStress2D`, `fvx.LinearDrag2D`, `fvx.QuadraticDrag2D`, `fvx.RayleighDamping2D`, combined with `fvx.ForcingSum` / `fvx.ForcingProduct` / `fvx.TimeVaryingForcing` |
| Time stepping | `fvx.rk3_ssp_step(state, rhs_fn, dt)`, `fvx.rk4_step`, `fvx.ab2_step`, `fvx.leapfrog_raf_step`, `fvx.imex_ssp2_step`; diffrax solvers `fvx.RK3SSP`, `fvx.SSP_RK104`, `fvx.IMEX_SSP2` with `fvx.solve_ocean_pde` |
| Several layers | `fvx.multilayer(op.method)` over `[nl, Ny, Nx]`; vertical modes `fvx.decompose_vertical_modes`, `fvx.layer_to_mode` |

Spectral transforms and solvers come from spectraldiffx (finitevolX
re-exports them); structured linear algebra (tridiagonal solves, Nyström
preconditioning) comes from gaussx.

## The rules your code must keep

- **Same-index staggering.** `T[j, i]` is the cell centre, `U[j, i]` its
  east face, `V[j, i]` its north face, `X[j, i]` its NE corner. Use the
  operator whose name says the mapping (`diff_x_T_to_U`, `U_to_T`) instead
  of writing the slice.
- **The caller owns the ghost ring.** Operators write only the interior
  and leave the ghost ring zero; fill ghosts (`enforce_periodic`, a
  `BoundaryConditionSet`, …) on every field before the next operator reads
  it, including inside the RHS of a time stepper. Backward operators (U→T,
  V→T) read the west / south ghost that nothing else fills.
- **Masks are built once and passed to the operator.** `Mask2D.from_mask`
  runs NumPy / SciPy eagerly; give it to the operator as `mask=` so the
  output is zero on land and advection falls back to narrower stencils at
  the coast. Never store `NaN` on land unless the operator documents that
  it sanitises its inputs.
- **Closed walls by default.** `Advection1D/2D` and `Diffusion2D/3D` take
  `wall="closed"` (no flux through the outer walls); use `wall="open"`
  with filled ghosts for periodic or open boundaries (the flux through the
  wall then comes from the ghost ring and mass is conserved exactly).
- **Units are yours.** Physical constants are parameters with SI defaults
  (`gravity` / `g` default to 9.80665, `omega` to Earth's rotation rate,
  `radius` to Earth's radius), never baked into the numerics, so a
  nondimensional model passes its own values and the operators work
  unchanged.
- **JAX rules.** Pure functions, `.at[...].set` instead of mutation, no
  Python control flow on traced values, float32 in gives float32 out; enable
  x64 for solver-accuracy work.

## Worked example

```python
import jax
import jax.numpy as jnp
import numpy as np

import finitevolx as fvx

jax.config.update("jax_enable_x64", True)

# 1. A doubly periodic 64 × 64 C-grid on [0, 1]²: arrays are [Ny, Nx] = [66, 66]
grid = fvx.CartesianGrid2D.from_interior(64, 64, 1.0, 1.0)
xc = (jnp.arange(grid.Nx) - 0.5) * grid.dx  # T-point x, ghost columns included
yc = (jnp.arange(grid.Ny) - 0.5) * grid.dy
X, Y = jnp.meshgrid(xc, yc)  # [Ny, Nx] T-point coordinates

# 2. Streamfunction at T-points → non-divergent velocities on the faces
psi = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)  # [Ny, Nx]
diff = fvx.Difference2D(grid=grid)
u, v = diff.grad_perp(psi)  # u at U-points (east faces), v at V-points (north faces)
u, v = fvx.enforce_periodic(u), fvx.enforce_periodic(v)  # the caller owns the ghosts
div = diff.divergence(u, v)  # [Ny, Nx] at T-points

# 3. Invert the discrete Laplacian back to ψ with the periodic (FFT) solver
lap = diff.laplacian(psi)  # ∇²ψ at T-points, zero ghost ring
psi_rec = fvx.streamfunction_from_vorticity(
    lap[1:-1, 1:-1], grid.dx, grid.dy, bc="fft"
)  # [64, 64], the zero-mean solution
inv_err = jnp.max(jnp.abs(psi_rec - psi[1:-1, 1:-1]))

# 4. Advect a tracer with WENO5 fluxes and SSP-RK3, periodic and conservative
adv = fvx.Advection2D(grid=grid)


def rhs(q):
    q = fvx.enforce_periodic(q)  # fill the ghost ring before every stencil
    return adv(q, u, v, method="weno5", wall="open")  # -∇·(q u) at T-points


q0 = jnp.exp(-((X - 0.3) ** 2 + (Y - 0.5) ** 2) / 0.005)  # [Ny, Nx]
dt = 0.4 * grid.dx / jnp.max(jnp.abs(u))  # CFL number 0.4
step = jax.jit(lambda q: fvx.rk3_ssp_step(q, rhs, dt))
q = q0
for _ in range(100):
    q = step(q)
mass_drift = (fvx.area_sum(q, grid) - fvx.area_sum(q0, grid)) / fvx.area_sum(q0, grid)

# 5. A closed basin with an island: masked operators return exactly zero on land
wet = np.ones((grid.Ny, grid.Nx), dtype=bool)
wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False
wet[28:38, 28:38] = False
mask = fvx.Mask2D.from_mask(wet)
dq = fvx.Advection2D(grid=grid, mask=mask)(q0, u, v, method="weno5")
land_max = jnp.max(jnp.abs(jnp.where(mask.h, 0.0, dq)))
```

With x64 on, the velocities from `grad_perp` have a divergence of order
1e-13 (round-off against velocities of order 6), the FFT solver inverts
the five-point Laplacian to about 1e-15, the periodic WENO5 / SSP-RK3 run
conserves the tracer's mass to about 1e-14 after 100 steps, and the masked
tendency is exactly zero on every land cell.

## Don't write it — use finitevolX

| Don't write… | Use |
|---|---|
| `(h[1:-1, 2:] - h[1:-1, 1:-1]) / dx` into a zeroed array | `fvx.Difference2D(grid=grid).diff_x_T_to_U(h)` or `fvx.interior(fvx.diff_x_fwd(h) / dx, h)` |
| `0.5 * (u[:, 1:] + u[:, :-1])` to move u to cell centres | `fvx.Interpolation2D(grid=grid).U_to_T(u)` |
| An upwind / WENO face reconstruction and flux divergence | `fvx.Advection2D(grid=grid)(q, u, v, method="weno5")` |
| `q[:, 0] = q[:, -2]` (periodic ghosts) and friends | `fvx.enforce_periodic(q)`, `fvx.BoundaryConditionSet.periodic()` |
| A wet-face mask `wet[:, :-1] & wet[:, 1:]` | `fvx.Mask2D.from_mask(wet).u` |
| `scipy.fft.dstn` Poisson solver | `fvx.streamfunction_from_vorticity(zeta, dx, dy, bc="dst")`, `fvx.solve_helmholtz_dst` |
| A conjugate-gradient loop on a masked basin | `fvx.solve_cg(matvec, rhs, preconditioner=fvx.make_preconditioner("spectral", dx=dx, dy=dy))`, or `fvx.build_multigrid_solver` |
| A Thomas algorithm for vertical diffusion | `fvx.solve_tridiagonal` |
| A hand-written RK3 / AB2 / leapfrog loop | `fvx.rk3_ssp_step`, `fvx.ab2_step`, `fvx.leapfrog_raf_step`, or `fvx.RK3SSP` with `fvx.solve_ocean_pde` |
| `jnp.sum(q * dx * dy)` | `fvx.area_sum(q, grid)` (excludes the ghost ring; `mask=` drops land) |
| Kinetic energy / PV / Okubo–Weiss by hand | `fvx.Energetics2D`, `fvx.Vorticity2D`, `fvx.Strain2D` |
| A `vmap` over layers | `fvx.multilayer(diff.laplacian)(psi_layers)` |

## Self-check before you finish

- Every field read by a stencil has its ghost ring filled for the boundary
  condition you mean (periodic, wall, open), at every RHS evaluation.
- Each operator maps the stagger you think: T fields into `*_T_to_*`,
  velocities on U / V faces, vorticity on X corners.
- The total of a conserved tracer (`fvx.area_sum`) is constant over a run
  with periodic or closed boundaries.
- `dt` respects the CFL limit (`dt ≲ dx / max|u|`; on the sphere use
  `grid.min_cell_width`).
- A float32 input stays float32; `jax.jit` and `jax.grad` run through your
  RHS.

If finitevolX lacks what you need, keep your addition small and shaped like
finitevolX (an `eqx.Module` holding `grid` and `mask`, built on the raw
stencils and writing only the interior) and consider proposing it upstream
at <https://github.com/jejjohnson/finitevolX/issues>.
