# Time Integration

The generic time steppers live in [integrix](https://github.com/jejjohnson/integrix) (`integrix.march`); finitevolX keeps the semi-Lagrangian scheme and the `solve_ocean_pde` wrapper.

!!! warning "Deprecated finitevolX names"
    `finitevolx.RK3SSP`, `finitevolx.rk3_ssp_step` and the other generic steppers below still work for one release and emit a `DeprecationWarning`. The diffrax solvers and the pure step functions are the integrix objects themselves, so import them from `integrix.march`. `finitevolx.AB2Solver`, `finitevolx.LeapfrogRAFSolver` and `finitevolx.SplitExplicitRKSolver` are adapters that keep the old `init` / `step` call shapes; their integrix replacements are `integrix.march.AbstractStepper`s run by `integrix.march.run(stepper, rhs, t0, y0, dt, n_steps)` with `rhs(t, y, args)`.

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

::: finitevolx.semi_lagrangian_step

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

::: finitevolx.SemiLagrangianSolver

::: finitevolx.SemiLagrangianTerm

## Deprecated Adapters

The objects behind `finitevolx.AB2Solver`, `finitevolx.LeapfrogRAFSolver` and `finitevolx.SplitExplicitRKSolver` for one release.

::: finitevolx._src.timestepping._deprecated.AB2Solver

::: finitevolx._src.timestepping._deprecated.LeapfrogRAFSolver

::: finitevolx._src.timestepping._deprecated.SplitExplicitRKSolver

## Convenience Wrapper

::: finitevolx.solve_ocean_pde
