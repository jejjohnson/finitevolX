# AGENTS.md — finitevolX coding conventions

## Documentation
* Use numpy-style docstrings for all functions and classes.
* Be pedantic and pedagogical — every stencil must have a comment showing the half-index formula.
* Use ascii math notation (e.g. `dh_dx[j, i+1/2] = ...`).
* Track array shapes in comments (e.g. `[Ny, Nx]`).

## Packages
* `equinox` for the dataclasses / `eqx.Module` base class.
* `jaxtyping` for the type annotations.
* `jax` / `jax.numpy` for the computations.
* **Do NOT depend on `finitediffx` or `kernex`** — use pure `jnp` slicing.

## Grid conventions
* All arrays have total shape `[Ny, Nx]`.  The physical interior is `(Ny-2) x (Nx-2)`;
  one ghost-cell ring on each side is reserved for boundary conditions.
* Same-index colocation:
  - `T[j, i]`  → cell centre  `(j,     i    )`
  - `U[j, i]`  → east face    `(j,     i+1/2)`
  - `V[j, i]`  → north face   `(j+1/2, i    )`
  - `X[j, i]`  → NE corner    `(j+1/2, i+1/2)`

## Operator idiom
* Every operator writes **only** into `[1:-1, 1:-1]` of the output array.
* Initialise output with `jnp.zeros_like(input)` then use `.at[1:-1, 1:-1].set(...)`.
* The caller is responsible for boundary conditions (pad, enforce_periodic, etc.).

## Testing
* One test per operator per dimension.
* Use fixtures for different grid sizes.
* Tests live in `tests/` at the repo root.
