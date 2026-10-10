# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/capabilities.md` (every public finitevolX
  name, plus the spectraldiffx and gaussx APIs underneath) before writing a
  stencil, an operator, a boundary condition, a solver or a helper; the
  "Reuse before you write" table in `AGENTS.md` maps the usual hand-rolled
  code to what already exists.
- **Grid first.** Before touching an operator, re-read contract 1 in
  `AGENTS.md` (same-index staggering and which ghost cells each direction
  reads): most stencil bugs here are an off-by-one-half in the slice.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building: `add-stencil-operator`, `add-advection-scheme`,
    `add-boundary-condition`, `add-forcing`, `add-elliptic-solver`,
    `add-time-integrator`, `add-notebook`, `bump-upstream-pins`;
  - shipping: `pre-pr-check`, `finitevolx-review`, `squash-commit`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `finitevolx-review`; run them on any diff that adds code, before
  committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/capabilities.md` (finitevolX, spectraldiffx, gaussx) or a shared
    private helper?
  - `grid-numerics-reviewer`: stagger and half-index slices, unfilled
    ghost cells, write regions, mask and `NaN` semantics, conservation,
    units, the sphere, CFL and solver signs, dtypes, traced control flow.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
