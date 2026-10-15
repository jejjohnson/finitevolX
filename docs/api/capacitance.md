# Capacitance Matrix Method

Extends fast spectral solvers to irregular / masked domains using the
Sherman-Morrison capacitance correction (Buzbee, Golub & Nielson 1970).
The unknowns are every wet cell of the mask, with ψ = 0 on the coast (the
dry cells next to water): the same problem `masked_laplacian` and
`method="cg"` define.

## Solver

!!! note
    `CapacitanceSolver` is re-exported from spectraldiffx, so its docstring
    describes spectraldiffx's own builder (unknowns: the wet cells off the
    inner ring, ψ = 0 on that ring).  finitevolX's
    [`build_capacitance_solver`][finitevolx.build_capacitance_solver] fills
    the same class with every wet cell as an unknown and ψ = 0 on the coast.

::: finitevolx.CapacitanceSolver

## Factory

::: finitevolx.build_capacitance_solver

## Masked Laplacian

::: finitevolx.masked_laplacian
