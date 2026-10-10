# Building with agents

finitevolX exists so that a staggered-grid model is assembled from parts
that already know their stagger, their ghost cells and their masks. Coding
agents tend to re-implement what they cannot see — a shifted slice divided
by `dx`, a ghost-row copy, a DST Poisson solver, an RK3 loop — and each one
is a place for an off-by-half-a-cell error, a leaking coastline or a
non-conservative flux. finitevolX ships three things that let agents find
the real thing instead.

## The capability index

The [capability index](capabilities.md) lists every public name in
`finitevolx`, grouped by the module that defines it, with a one-line
summary; then the public API of spectraldiffx and gaussx, which finitevolX
builds on (its spectral solvers and transforms are re-exported from
spectraldiffx). It is regenerated from the code and checked in the test
suite, so it never drifts.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/finitevolX
/plugin install finitevolx@finitevolx
```

The plugin adds:

- **`finite-volume-models-with-finitevolx`** (skill) — loads whenever a
  task discretises a PDE on a staggered grid, writes a stencil or a
  ghost-cell fill, advects a tracer, inverts vorticity or PV, or
  time-steps a model: what lives where, the rules (the caller owns the
  ghost ring, use the operator that names the stagger, masks are passed to
  operators, closed walls by default, constants are parameters), a worked
  example, and a "don't write it — use finitevolX" table.
- **`finitevolx-reuse-reviewer`** (subagent) — a read-only check of a diff
  for finite-volume code finitevolX already provides, and for misuse
  (unfilled ghosts, wrong stagger, `NaN` on land).

## llms.txt

For other agents and tools, the docs site serves
[`llms.txt`](https://jejjohnson.github.io/finitevolX/llms.txt): a curated
map of finitevolX and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on
finitevolX:

```markdown
## Finite-volume numerics: build on finitevolX

This project discretises its equations on Arakawa C-grids with finitevolX
(stencils, operators, masks, boundary conditions, advection, elliptic
solvers, time steppers). Before writing a stencil, a ghost-cell fill, an
advection scheme, a Poisson / Helmholtz solver or a time-stepping loop,
search the capability index
(https://jejjohnson.github.io/finitevolX/capabilities/) or
`finitevolx.__all__`, and compose what exists:

- Arrays are `[Ny, Nx]` with one ghost ring; `T[j, i]` is a cell centre,
  `U[j, i]` its east face, `V[j, i]` its north face, `X[j, i]` its NE
  corner. Use the operator that names the mapping
  (`Difference2D.diff_x_T_to_U`, `Interpolation2D.U_to_T`) instead of
  writing the slice.
- Operators write only the interior. Fill every field's ghost ring
  (`enforce_periodic`, a `BoundaryConditionSet`) before the next operator
  reads it, including inside the time stepper's right-hand side.
- Build a `Mask2D.from_mask(wet)` once and pass it as `mask=` to the
  operators; don't multiply by a mask afterwards or keep `NaN` on land.
- Advect with `Advection2D(...)(q, u, v, method="weno5")`; invert with
  `streamfunction_from_vorticity` / `pv_inversion`; step with
  `rk3_ssp_step` or a finitevolX diffrax solver.
- Pass physical constants explicitly; never hard-code them in a model
  meant to run nondimensionally.
```

## Working on finitevolX itself

Contributors (and their agents) follow
[`AGENTS.md`](https://github.com/jejjohnson/finitevolX/blob/main/AGENTS.md)
in the repository: the module map, the boundaries with spectraldiffx and
gaussx, "reuse before you write", the C-grid, operator, mask,
boundary-condition, solver and unit contracts and the tests that enforce
them, and recipe skills for adding stencil operators, advection schemes,
boundary conditions, forcing terms, elliptic solvers, time integrators and
notebooks.
