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

### Ghost-Cell Rules for All Staggered Types

All four array types (T, U, V, X) share shape `[Ny, Nx]`.  Operators write
only to `[1:-1, 1:-1]` but the ghost cells consumed differ per type:

| Operator direction | Source ghost consumed          | Slice that reads it    |
|--------------------|--------------------------------|------------------------|
| T→U (forward x)    | east ghost T `T[j, Nx-1]`     | `h[1:-1, 2:]` at last col |
| T→V (forward y)    | north ghost T `T[Ny-1, i]`    | `h[2:, 1:-1]` at last row |
| V→X (forward x)    | east ghost V `V[j, Nx-1]`     | `v[1:-1, 2:]` at last col |
| U→X (forward y)    | north ghost U `U[Ny-1, i]`    | `u[2:, 1:-1]` at last row |
| U→T (backward x)   | west ghost U `U[j, 0]`        | `u[1:-1, :-2]` at first col |
| V→T (backward y)   | south ghost V `V[0, i]`       | `v[:-2, 1:-1]` at first row |
| X→V (backward x)   | west ghost X `X[j, 0]`        | `q[1:-1, :-2]` at first col |
| X→U (backward y)   | south ghost X `X[0, i]`       | `q[:-2, 1:-1]` at first row |
| U→V                | north ghost U `U[Ny-1, i]`    | `u[2:, 1:-1]` at last row  |
| V→U                | east ghost V `V[j, Nx-1]`     | `v[1:-1, 2:]` at last col  |

**Forward operators** compute the last interior face using BC-owned ghost source
cells — this is **correct and intentional**.  **Backward operators** read
BC-owned ghost *output-type* cells for the first interior cell — the caller
**must** set these before chaining operators.

### Advection Write Region

Advection operators use `[2:-2]` / `[2:-2, 2:-2]` (not `[1:-1]`) to avoid
reading ghost flux cells.  `Advection3D` uses `[1:-1, 2:-2, 2:-2]` since
the z-axis is independent.  Never change this to `[1:-1, ...]` for the
horizontal dimensions.

### Array Shape Tracking
Always document array shapes in comments:
```python
# h: [Ny, Nx] - scalar field at T-points
# u: [Ny, Nx] - velocity at U-points (east faces)
# v: [Ny, Nx] - velocity at V-points (north faces)
# q: [Ny, Nx] - quantity at X-points (NE corners)
```
