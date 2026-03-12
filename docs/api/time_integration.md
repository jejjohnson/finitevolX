# Time Integration

Pure functional integrators, diffrax-based solvers, and convenience wrappers
for time-stepping PDEs on Arakawa C-grids.

## Explicit Runge-Kutta (Pure Functional)

::: finitevolx.euler_step

::: finitevolx.heun_step

::: finitevolx.rk3_ssp_step

::: finitevolx.rk4_step

## Multistep Methods (Pure Functional)

::: finitevolx.ab2_step

::: finitevolx.ab3_step

::: finitevolx.leapfrog_raf_step

## IMEX (Pure Functional)

::: finitevolx.imex_ssp2_step

## Split-Explicit (Pure Functional)

::: finitevolx.split_explicit_step

## Semi-Lagrangian (Pure Functional)

::: finitevolx.semi_lagrangian_step

## Diffrax Solvers

::: finitevolx.ForwardEulerDfx

::: finitevolx.RK2Heun

::: finitevolx.RK3SSP

::: finitevolx.RK4Classic

::: finitevolx.SSP_RK2

::: finitevolx.SSP_RK104

::: finitevolx.IMEX_SSP2

## Manual Solver Interfaces

::: finitevolx.AB2Solver

::: finitevolx.LeapfrogRAFSolver

::: finitevolx.SplitExplicitRKSolver

::: finitevolx.SemiLagrangianSolver

## Convenience Wrapper

::: finitevolx.solve_ocean_pde
