# pipekit Operators

`gridax_jax.pipekit` wraps the 2-D Cartesian operators as
[pipekit](https://github.com/jejjohnson/pipekit) operators: JSON
configuration, `Sequential` / `Graph` composition, and Equinox pytree leaves
so `jit`, `grad` and `vmap` see through them. Each operator is one Tier-A
call on `[*batch, Ny, Nx]` fields. Requires the `pipekit` extra:
`pip install "gridax-jax[pipekit]"`. Walkthrough:
[Composing finitevolX with pipekit](../pipekit-composition/).

| Tier-B operator | Tier-A counterpart | in → out |
|---|---|---|
| `Difference(axis, src, dst)` | `Difference2D.diff_{axis}_{src}_to_{dst}` | 1 → 1 |
| `Divergence` | `Difference2D.divergence` | (U, V) → T |
| `Curl` | `Difference2D.curl` | (U, V) → X |
| `Laplacian` | `Difference2D.laplacian` | T → T |
| `GradPerp` | `Difference2D.grad_perp` | X → (U, V) |
| `Interp(src, dst)` | `Interpolation2D.{src}_to_{dst}` | 1 → 1 |
| `Coriolis(f0, beta, y0)` | `geophysix.rotation.beta_plane` + `Coriolis2D` | (U, V) → (U, V) |
| `FillGhosts(kind)` | `enforce_periodic`, `zero_boundaries`, …, `wall_boundaries` | 1 → 1 |
| `ApplyMask(mask, stagger)` | `jnp.where(mask.{h,u,v,xy_corner_strict}, x, 0)` | 1 → 1 |
| `RelativeVorticity` | `Vorticity2D.relative_vorticity` | (U, V) → X |
| `PotentialVorticity` | `Vorticity2D.potential_vorticity` | (U, V, T, T) → X |
| `Enstrophy` | `Vorticity2D.enstrophy` | (U, V) → X |
| `QGPotentialVorticity(f0, beta, y0)` | `QGPotentialVorticity2D` | T → T |
| `KineticEnergy` | `Energetics2D.kinetic_energy` | (U, V) → T |
| `AvailablePE(g_prime, H)` | `Energetics2D.available_potential_energy` | T → T |
| `Strain(component)` | `Strain2D.shear` / `.tensor` / `.magnitude_squared` | (U, V) → X / T |
| `OkuboWeiss` | `Strain2D.okubo_weiss` | (U, V) → T |
| `TotalEnergy` | `total_energy` | (T, T) → `[*batch]` |
| `DomainIntegral` / `DomainMean` | `area_sum` / `area_mean` | T → `[*batch]` |
| `ExplicitStep(rhs, dt, scheme)` | `euler_step` / `heun_step` / `rk3_ssp_step` / `rk4_step` | state → state |
| `Pick(index)`, `LinComb(coeffs)` | — (graph plumbing) | |

## Base classes and helpers

::: gridax_jax.pipekit.FVXOperator

::: gridax_jax.pipekit.GridOperator

::: gridax_jax.pipekit.as_grid_spec

::: gridax_jax.pipekit.build_grid

::: gridax_jax.pipekit.over_leading

::: gridax_jax.pipekit.testing.check_operator

## Stencils, interpolation, boundaries and masks

::: gridax_jax.pipekit.Difference

::: gridax_jax.pipekit.Divergence

::: gridax_jax.pipekit.Curl

::: gridax_jax.pipekit.Laplacian

::: gridax_jax.pipekit.GradPerp

::: gridax_jax.pipekit.Interp

::: gridax_jax.pipekit.Coriolis

::: gridax_jax.pipekit.FillGhosts

::: gridax_jax.pipekit.ApplyMask

## Diagnostics and reductions

::: gridax_jax.pipekit.RelativeVorticity

::: gridax_jax.pipekit.PotentialVorticity

::: gridax_jax.pipekit.Enstrophy

::: gridax_jax.pipekit.QGPotentialVorticity

::: gridax_jax.pipekit.KineticEnergy

::: gridax_jax.pipekit.AvailablePE

::: gridax_jax.pipekit.Strain

::: gridax_jax.pipekit.OkuboWeiss

::: gridax_jax.pipekit.TotalEnergy

::: gridax_jax.pipekit.DomainIntegral

::: gridax_jax.pipekit.DomainMean

## Time stepping and graph plumbing

::: gridax_jax.pipekit.ExplicitStep

::: gridax_jax.pipekit.Pick

::: gridax_jax.pipekit.LinComb
