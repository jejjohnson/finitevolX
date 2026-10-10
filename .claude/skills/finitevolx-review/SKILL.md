---
name: finitevolx-review
description: Review a change or pull request in finitevolX against the repo's own rules — CODE_REVIEW.md, the C-grid / operator / mask / boundary-condition / solver / units contracts in AGENTS.md, reuse of finitevolX, spectraldiffx and gaussx primitives, and stencil and conservation correctness. Use when asked to review a diff, branch or PR in this repo.
---

# Review a finitevolX change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   which areas it touches (grid, masks, stencils, operators, advection,
   diffusion, boundary, forcing, solvers, timestepping, vertical, docs).
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: hand-written
   slices that duplicate a raw stencil or an `Interpolation*` method, a
   spectral solve or DCT / DST, a tridiagonal solve, a CG loop, a ghost-ring
   fill or a diagnostic are the main way finitevolX drifts from itself and
   from spectraldiffx / gaussx.
3. **Grid numerics** — run the `grid-numerics-reviewer` subagent in
   parallel with step 2: half-index slices against their staggering, ghost
   cells read before a BC fills them, write regions, mask semantics and
   `NaN` leakage, conservation, units, dtype, traced control flow, solver
   convergence.
4. **Contracts** — for each area touched, the rules in `AGENTS.md`: the
   interior idiom, the three layers, `mask=None` bit-identical, BC atoms in
   the `BoundaryCondition1D` union, the Helmholtz sign convention, forcings
   as `AbstractForcing`, constants only as defaults.
5. **Tests and fixtures** — new operators have analytic tests, golden
   entries (unmasked and masked), a scale-covariance `Case`; a changed
   golden `.npz` is explained in the PR; tolerances say where they come
   from.
6. **Checklist** — the rest of `CODE_REVIEW.md` (public API, `docs/api`
   entry, capability index, docstrings with the formula), skipping what
   ruff, ty or the tests already enforce.
7. **Verify claims** — for anything you flag as a bug, run it: the operator
   on a linear or trigonometric field against the analytic result, the
   tendency sum for conservation, the masked against the unmasked output,
   a float32 input, `jax.jit` / `jax.grad` around the call.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
