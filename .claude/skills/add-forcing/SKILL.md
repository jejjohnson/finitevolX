---
name: add-forcing
description: Add a forcing term to finitevolX — a functional primitive in forcing/functional.py plus an AbstractForcing operator (2-D and 3-D) such as wind stress, bottom drag, Rayleigh damping, a sponge or a relaxation, with face interpolation, NaN-safe masking, composition through ForcingSum / ForcingProduct / TimeVaryingForcing, tests and docs. Use when asked to add, port or implement a forcing, drag, damping, relaxation or surface-flux term in finitevolx/_src/forcing.
---

# Add a forcing term

Read contracts 3 (masks) and 6 (forcing and units) in `AGENTS.md` and the
module docstrings of `finitevolx/_src/forcing/_base.py` and
`forcing/functional.py` first.

## 1. Make sure it does not exist yet

Search `docs/capabilities.md`: `wind_stress_tendency`,
`linear_drag_tendency`, `quadratic_drag_tendency`, `rayleigh_tendency`;
`WindStress2D/3D`, `LinearDrag2D/3D`, `QuadraticDrag2D/3D`,
`RayleighDamping2D/3D`; the older tendency helpers `linear_drag` and
`rayleigh_relaxation` (`operators/relaxation.py`). A sponge is a
`ForcingProduct(alpha, RayleighDamping2D(...))`; a seasonal cycle is a
`TimeVaryingForcing`; a sum of processes is a `ForcingSum`. None needs a
new class.

## 2. Write it (`finitevolx/_src/forcing/`)

1. **The primitive** in `functional.py`: pure math on arrays already at the
   right stagger, no grid, no ghost ring, no mask, one velocity component
   per call; physical constants (`rho0`, …) are arguments.
2. **The operator** in its own module (`wind.py`, `drag.py`, … as models):
   subclass `AbstractForcing`; fields `grid`, `mask: Mask2D | None`, and
   any helper operator built in `__init__` (an `Interpolation2D` to move
   T-point coefficients to the faces). `__call__` takes the
   physics-specific inputs and returns a tendency: one array, or a tuple
   `(du, dv)` at U- and V-points.
3. Inside `__call__`, reuse `_utils.py`: `interior_or_scalar` /
   `on_face_interior` (scalar or T-field coefficient → interior face
   values), `safe_speed` and `safe_denominator` (finite values **and**
   gradients at zero speed / dry thickness), `mask_where` (zero dry points
   with `jnp.where`). Zero dry inputs before the math, then mask the
   outputs: NaN-filled land must not leak into the values or the gradient.
4. Pad with `interior(...)`: the ghost ring is zero.
5. 3-D: act on the levels the physics says (bottom drag only on `k = 1`,
   wind stress only on the top interior level); keep the ghost slabs zero.
6. Stay time-agnostic: time dependence goes through `TimeVaryingForcing`
   or the user's `rhs_fn(t, ...)`.

## 3. Export and document

`finitevolx/__init__.py` (import + `__all__` under `# Forcing — Layer 0
functional primitives` and `# Forcing — 2D operators` / `3D operators`),
`::: finitevolx.NewForcing` on `docs/api/forcing.md`, the formula in the
forcing section of `docs/spatial_operators.md`, `make capabilities`.

## 4. Tests

- `tests/test_forcing_functional.py`: the primitive against a hand formula,
  broadcasting of scalar coefficients, `jit` / `grad`.
- `tests/test_forcing_2d.py` / `tests/test_forcing_3d.py`: `isinstance(op,
  AbstractForcing)`, a uniform field against the closed form, coefficients
  interpolated to the faces, zero ghost ring, masked dry faces exactly
  zero, `NaN` on land giving finite values and gradients, and the 3-D level
  placement.
- `tests/test_forcing_base.py` if it adds behaviour to the composition
  wrappers.

## 5. Verify

```bash
uv run pytest -n 2 tests/test_forcing_functional.py tests/test_forcing_2d.py \
  tests/test_forcing_3d.py tests/test_forcing_base.py tests/test_relaxation.py \
  tests/test_capabilities.py
```

Then the `pre-pr-check` skill.
