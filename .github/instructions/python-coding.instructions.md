---
applyTo: "finitevolx/**/*.py,tests/**/*.py,scripts/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `dataclasses`, `attrs`, or `equinox.Module` for data containers
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Package Preferences

| Purpose | Preferred Package |
|---------|-------------------|
| Logging | `loguru` |
| CLI | `cyclopts` |
| Data containers | `equinox.Module`, `dataclasses` (stdlib), or `attrs` |
| Configuration | `hydra-core` / `omegaconf` |
| Path handling | `pathlib` (stdlib) |
| HTTP | `httpx` |
| Testing | `pytest` |
| JAX computations | `jax`, `jaxtyping`, `equinox`, `beartype` |

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (NumPy style)
- Inline comments explaining *why*, not *what*
- Scientific algorithms should include Unicode equations in docstrings (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- Public classes and functions should include 2–3 example use cases in docstrings

## finitevolX-Specific Standards

### NumPy-Style Docstrings
```python
def diff_x_T_to_U(self, h: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
    """Forward difference in x: T-point -> U-point.

    dh_dx[i+1/2] = (h[i+1] - h[i]) / dx

    Parameters
    ----------
    h : Float[Array, "Nx"]
        Scalar field at T-points.

    Returns
    -------
    Float[Array, "Nx"]
        Forward x-difference at U-points, same shape as input.
    """
```

### JAX Type Hints
- Use `jaxtyping` for array shapes: `Float[Array, "Ny Nx"]`
- Use `beartype` for runtime type checking
- All operators inherit from `equinox.Module`

### Stencil Comments
Every stencil operation must include the half-index formula:
```python
# dh_dx[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx
out = out.at[1:-1, 1:-1].set((h[1:-1, 2:] - h[1:-1, 1:-1]) / self.grid.dx)
```

### Array Shape Tracking
Always document array shapes in comments:
```python
# h: [Ny, Nx] - scalar field at T-points
# u: [Ny, Nx] - velocity at U-points (east faces)
# v: [Ny, Nx] - velocity at V-points (north faces)
```
