---
name: squash-commit
description: Write a Conventional-Commits squash commit message for a finitevolX pull request from its commits and description. Use when asked for a squash / merge commit message.
---

Generate a squash commit message for a GitHub PR.

## Instructions

1. If an argument is provided (PR number or URL), fetch the PR details
   (`gh pr view`, or the GitHub MCP tools). Otherwise, detect the current
   branch and find its open PR.
2. Fetch all individual commit messages in the PR (`gh api` or `git log`).
3. Combine them into a single squash commit message following these rules.

### Format

```
<type>(<scope>): <concise summary of the overall change>

<1-3 sentence description combining the key changes from all commits.
Focus on the "why" and overall effect, not per-commit details.>

Co-authored-by: <preserve all unique Co-authored-by lines from the individual commits>
```

### Rules

- Conventional Commits types: `feat`, `fix`, `docs`, `style`, `refactor`,
  `perf`, `test`, `build`, `ci`, `chore`, `revert`; a lowercase subject (CI
  validates PR titles against `^[a-z].*$`). The scope is optional (CI does
  not require one); when used, name the area: `grid`, `masks`, `stencils`,
  `operators`, `advection`, `diffusion`, `boundary`, `forcing`, `solvers`,
  `multigrid`, `timestepping`, `vertical`, `spherical`, `docs`, `deps`,
  `agents`, ….
- release-please (`release-please-config.json`) builds `CHANGELOG.md` from
  `feat` (Features), `fix` (Bug Fixes), `perf` (Performance Improvements)
  and `revert` (Reverts) commits; the other types are hidden, so the type
  decides whether a change is listed. Before 1.0, a `feat` bumps the patch
  version (`bump-patch-for-minor-pre-major`) and a breaking change the
  minor.
- The type/scope reflect the dominant change across all commits; mention the
  rest in the body.
- A breaking change keeps its `!` and `BREAKING CHANGE:` footer.
- Collapse redundant or incremental commits into one coherent description.
- Preserve ALL unique `Co-authored-by` lines.
- Keep the summary line under 72 characters.
- Do NOT list the individual commit messages as bullet points — this is a
  squash, not a merge.

### Output

Print ONLY the final squash commit message in a code block so the user can
copy it directly. Do not add explanation before or after.
