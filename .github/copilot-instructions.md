# Copilot Instructions

## Project Overview

- **Python**: 3.12+
- **Package Manager**: uv
- **Layout**: Package layout (`finitevolx/`)
- **Testing**: pytest
- **Docs**: MkDocs + Material + mkdocstrings + mkdocs-jupyter
- **Scientific Computing**: JAX, equinox, jaxtyping

## Build & Test Commands

```bash
make install     # Install all dependencies (uv sync --all-extras)
make test        # Run tests (uv run pytest tests -v)
make uv-lint     # Lint and type-check code (ruff + ty)
make uv-format   # Format code (ruff format + ruff check --fix)
make uv-pre-commit  # Run pre-commit on all files
make docs-serve  # Serve docs locally
```

## Key Directories

| Path | Purpose |
|------|---------|
| `finitevolx/` | Main package source code |
| `finitevolx/_src/` | Internal implementation details |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Example scripts |

## Behavioral Guidelines

### Do Not Nitpick
- Ignore style issues that linters/formatters catch (formatting, import order, quote style)
- Don't suggest changes to code you weren't asked to modify
- Match existing patterns even if you'd do it differently

### Always Propose Tests
When implementing features or fixing bugs:
1. Write a test that verifies the expected behavior
2. Implement the change
3. Verify the test passes

### Never Suggest Without a Proposal
Bad: "You should add validation here"
Good: "Add validation here. Proposed implementation:"
```python
if value < 0:
    raise ValueError('Value must be non-negative')
```

### Simplicity First
- No abstractions for single-use code
- No speculative features beyond what was asked
- If 200 lines could be 50, propose the simpler version

### Surgical Changes
- Only modify lines directly related to the request
- Don't refactor adjacent code
- Don't add docstrings/comments to code you didn't change
- Remove only imports/functions that YOUR changes made unused

## finitevolX-Specific Guidelines

### Documentation
- Use NumPy-style docstrings
- Include half-index formulas in comments for stencils (e.g., `dh_dx[j, i+1/2] = ...`)
- Track array shapes in comments (e.g., `[Ny, Nx]`)

### JAX & Array Operations
- Use `equinox.Module` for dataclasses
- Use `jaxtyping` for array type hints
- Pure JAX operations only (no `finitediffx` or `kernex`)
- All arrays have shape `[Ny, Nx]` with ghost cells
- Operators write only to `[1:-1, 1:-1]` interior

### Testing
- One test per operator per dimension
- Use fixtures for different grid sizes
- All tests live in `tests/` at repo root

## Code Review

For all code review tasks, follow the guidance in `/CODE_REVIEW.md`.
