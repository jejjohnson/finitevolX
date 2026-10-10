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
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
