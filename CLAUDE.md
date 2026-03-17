# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

finitevolX is a JAX-based finite-volume library providing Arakawa C-grid operators for computational fluid dynamics and geophysical modeling. Built on JAX, equinox, jaxtyping, and diffrax. Python 3.12+.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-extras) + pre-commit hooks
make test                 # Run tests: uv run pytest tests
make test-cov             # Tests with coverage
make uv-format            # Auto-fix: ruff format . && ruff check --fix .
make uv-lint              # Lint + format check + typecheck
make ci                   # Full CI: lint → format check → typecheck → tests
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_elliptic.py::TestVmapSpectralSolvers::test_vmap_helmholtz_dst -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest tests -v          # Tests
uv run ruff check .             # Lint — ENTIRE repo, not just finitevolx/
uv run ruff format --check .    # Format — ENTIRE repo
uv run ty check finitevolx      # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `finitevolx/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Architecture

### Package structure

All implementation lives in `finitevolx/_src/` (internal, not directly imported by users). The public API is re-exported through `finitevolx/__init__.py`.

### Core abstractions

- **Grid** (`ArakawaCGrid1D/2D/3D`): Spatial discretization containers. Shape `[Ny, Nx]` with 2-cell ghost ring. Interior: `[1:-1, 1:-1]`. Factory: `ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)`.
- **Operators** (`Difference`, `Interpolation`, `Advection`, `Vorticity`, `Divergence`, `Coriolis`, `Diffusion`): Stateless `equinox.Module` classes, one per dimension (1D/2D/3D). Write only to interior cells.
- **Boundary conditions**: `BoundaryConditionSet` (per-face) and `FieldBCSet` (per-field). 1D BC types: Periodic, Dirichlet, Neumann, Robin, Slip, Sponge, Reflective, Extrapolation, Outflow.
- **Masks** (`ArakawaCGridMask`): Land/ocean masks with automatic stencil dispatch (WENO/TVD auto-downsamples at irregular boundaries).
- **Elliptic solvers**: Spectral (DST/DCT/FFT for Poisson/Helmholtz), capacitance matrix (masked domains), preconditioned CG.
- **Time steppers**: Pure functional (`euler_step`, `heun_step`, `rk4_step`, etc.) and diffrax-based classes. `solve_ocean_pde` for composition.
- **Vertical**: `multilayer` vmap helper applies 2D operators across vertical layers.

### Key design principles

- All operators are pure functions or stateless pytrees (JAX-compatible: JIT, vmap, grad)
- Ghost cells (2-cell ring) are essential for boundary conditions and stencil correctness
- `jaxtyping` shapes (`Float[Array, "Ny Nx"]`) encode array semantics
- No in-place mutations; immutable operations only

## Coding Conventions

- NumPy-style docstrings with half-index stencil formulas (e.g., `dh_dx[j, i+1/2] = ...`)
- `equinox.Module` for dataclasses, `jaxtyping` for array type hints
- Pure JAX operations only (no finitediffx or kernex)
- One test class per operator per dimension; use fixtures for grid sizes
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
