# Time Integration

The generic time steppers live in [integrix](https://github.com/jejjohnson/integrix) (`integrix.march`); the semi-Lagrangian scheme moved there too (#476), and the `solve_ocean_pde` wrapper moved to `sesmax_core.solve_pde`.

## Explicit Runge-Kutta (Pure Functional)

::: integrix.march.euler_step
    options:
      docstring_style: google

::: integrix.march.heun_step
    options:
      docstring_style: google

::: integrix.march.rk3_ssp_step
    options:
      docstring_style: google

::: integrix.march.rk4_step
    options:
      docstring_style: google

## Multistep Methods (Pure Functional)

::: integrix.march.ab2_step
    options:
      docstring_style: google

::: integrix.march.ab3_step
    options:
      docstring_style: google

::: integrix.march.leapfrog_raf_step
    options:
      docstring_style: google

## IMEX (Pure Functional)

::: integrix.march.imex_ssp2_step
    options:
      docstring_style: google

## Split-Explicit (Pure Functional)

::: integrix.march.split_explicit_step
    options:
      docstring_style: google

## Semi-Lagrangian (Pure Functional)

::: integrix.march.departure_points
    options:
      docstring_style: google

::: integrix.march.semi_lagrangian_step
    options:
      docstring_style: google

The T-point coordinates it takes as nodes come from gridax:

::: gridax_jax.t_point_coordinates

## Diffrax Solvers

::: integrix.march.ForwardEulerDfx
    options:
      docstring_style: google

::: integrix.march.RK2Heun
    options:
      docstring_style: google

::: integrix.march.RK3SSP
    options:
      docstring_style: google

::: integrix.march.RK4Classic
    options:
      docstring_style: google

::: integrix.march.SSP_RK2
    options:
      docstring_style: google

::: integrix.march.SSP_RK104
    options:
      docstring_style: google

::: integrix.march.IMEX_SSP2
    options:
      docstring_style: google

::: integrix.march.AB3Solver
    options:
      docstring_style: google

## Steppers Run by `integrix.march.run`

::: integrix.march.run
    options:
      docstring_style: google

::: integrix.march.AbstractStepper
    options:
      docstring_style: google

::: integrix.march.AB2Solver
    options:
      docstring_style: google

::: integrix.march.LeapfrogRAFSolver
    options:
      docstring_style: google

::: integrix.march.SplitExplicitRKSolver
    options:
      docstring_style: google

::: integrix.march.SplitExplicitTerms
    options:
      docstring_style: google

## Semi-Lagrangian Solver

::: integrix.march.SemiLagrangianSolver
    options:
      docstring_style: google

::: integrix.march.SemiLagrangianTerm
    options:
      docstring_style: google

## Convenience Wrapper

`solve_ocean_pde` moved to `sesmax_core.solve_pde`, where `bc_fn=` is `tendency_bc=`, and gridax no longer has it (#475).
See the [`sesmax_core.solve_pde` API](https://jejjohnson.github.io/sesmax/reference/core/).
