---
name: bump-upstream-pins
description: Bump finitevolX's git-pinned GeoML upstreams — spectraldiffx and gaussx (a release tag in [tool.uv.sources] plus a >= floor in dependencies) — across pyproject.toml, uv.lock and the capability index, then fix what the new versions break (re-exported solver names, the tridiagonal and Nyström delegations). Use when asked to upgrade spectraldiffx or gaussx, after an upstream release, or when a new upstream feature is needed.
---

# Bump spectraldiffx / gaussx

Neither is on PyPI, so each is pinned to a release tag and a bump touches
several places that must agree.

## 1. Where the pins live

| Upstream | Where | Tag format |
|---|---|---|
| spectraldiffx | `pyproject.toml`: the `spectraldiffx>=X.Y.Z` floor in `[project] dependencies` **and** `tag = "X.Y.Z"` in `[tool.uv.sources]` | no `v` prefix since 0.0.12 (`0.0.13`, `0.1.1`) |
| gaussx | `pyproject.toml`: the `gaussx>=X.Y.Z` floor **and** `tag = "vX.Y.Z"` in `[tool.uv.sources]` | `v` prefix (`v0.0.17`) |
| both | `uv.lock`; `docs/capabilities.md` (records the versions it was generated against) | |

List the available tags with
`git ls-remote --tags https://github.com/jejjohnson/spectraldiffx.git` (or
`.../gaussx`). spectraldiffx itself depends on gaussx: bump gaussx to a
version the new spectraldiffx accepts (check its `pyproject.toml` at the
new tag), or the lock fails to resolve.

## 2. Bump

1. Read the upstream changelog between the old and new tags; note
   removals, renames, deprecations and changed signatures. finitevolX
   re-exports 78 spectraldiffx names unchanged (`solvers/spectral.py`,
   `solvers/spectral_transforms.py`, `CapacitanceSolver`) and wraps
   `build_capacitance_solver`; from gaussx it re-exports
   `solve_tridiagonal` / `solve_tridiagonal_batched` and builds
   `make_nystrom_preconditioner` on `gaussx.as_linear_operator` and
   `gaussx.NystromPreconditioner`.
2. Update the floor and the tag together.
3. `uv lock`, then `uv sync --all-extras`.
4. `make capabilities`: the upstream sections and their recorded versions
   change. If `tests/test_capabilities.py` now reports a clash (a new
   upstream name equal to a finitevolX name for a different object),
   rename, or add it to `ALLOWED_SHARED_NAMES` in `scripts/capabilities.py`
   with the reason.

## 3. Fix and verify

```bash
uv run pytest -n auto tests/test_elliptic.py tests/test_elliptic_bcset.py \
  tests/test_elliptic_known_values.py tests/test_solver_wrappers.py \
  tests/test_multigrid.py tests/test_tridiagonal.py tests/test_scripts.py \
  tests/test_capabilities.py
make test
```

- A removed or renamed re-export is a breaking change for finitevolX users:
  keep the old name importable (an alias with a `DeprecationWarning`
  naming the replacement) for at least one release rather than dropping it
  from `__all__`. `filterwarnings = "error"` ignores `DeprecationWarning`,
  so read the test output for new upstream deprecations and replace the
  deprecated calls.
- If the upstream now provides something finitevolX hand-rolls, note it in
  the PR; converting it is a separate change unless it is a one-liner.
- `make typecheck`, lint and format, then the `pre-pr-check` skill.

The PR title is `build(deps): bump spectraldiffx to X.Y.Z` (or `gaussx to
vX.Y.Z`, or both together).
