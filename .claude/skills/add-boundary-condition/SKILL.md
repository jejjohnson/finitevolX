---
name: add-boundary-condition
description: Add a boundary condition to finitevolX — a per-face ghost-ring atom (like Dirichlet1D, Neumann1D, Robin1D, Slip1D, Sponge1D), a BoundaryConditionSet factory, or a functional ghost-ring filler (like wall_boundaries) — with its half-cell placement, the BoundaryCondition1D union, elliptic-solver compatibility, tests and docs. Use when asked to add, port or fix a boundary condition, ghost-cell fill, wall or open-boundary treatment in finitevolx/_src/boundary.
---

# Add a boundary condition

Read contract 1 (which ghost cells are BC-owned for each stagger) and
contract 4 (boundary conditions) in `AGENTS.md`, then
`docs/boundary_conditions.md`.

## 1. Make sure it does not exist yet

Search `docs/capabilities.md` (the `bc_1d`, `bc_set`, `bc_field` and
`boundary` groups): `Dirichlet1D`, `Neumann1D`, `Robin1D` (α·u + β·∂u/∂n =
γ), `Periodic1D`, `Outflow1D`, `Extrapolation1D`, `Reflective1D`,
`Sponge1D`, `Slip1D`; the sets `BoundaryConditionSet.periodic()` /
`.closed()` / `.open()`; the fillers `enforce_periodic`, `pad_interior`,
`zero_boundaries`, `zero_gradient_boundaries`, `no_slip_boundaries`,
`free_slip_boundaries`, `no_flux_boundaries`, `wall_boundaries`,
`fix_boundary_corners`. Many "new" conditions are a `Robin1D` with
particular coefficients, or a set of existing atoms.

## 2. Write it (`finitevolx/_src/boundary/`)

- **An atom** goes in `bc_1d.py`: an `eqx.Module` with
  `face: Face = eqx.field(static=True)` and its parameters as fields;
  validation in `__check_init__` (as `Sponge1D` checks `weight`);
  `__call__(self, field, dx, dy)` returns `field` with **one** ghost row /
  column replaced, using the private helpers there (`_adjacent_interior`,
  `_opposite_interior`, `_normal_spacing`, `_set_face`).
- Place the condition **at the face**, half a cell from the first interior
  value: Dirichlet sets `ghost = 2 * value - interior`, Neumann
  `ghost = interior + value * spacing` on every face (the ghost cell lies
  outward of the interior, so `(ghost - interior) / spacing` is already the
  outward-normal gradient). Write that formula in the docstring and in a
  comment.
- Give the atom `point: Point = eqx.field(static=True, default="T")` and
  pass it to `_adjacent_interior` / `_set_face`: when `_on_wall(face,
  point)` is true (`U` / `X` on west / east, `V` / `X` on south / north)
  the field's node is on the wall itself, so impose the condition on that
  node (Dirichlet writes `value`, a gradient is one-sided) and let
  `_set_face` write the east / north mirror slot.
- Add the class to the `BoundaryCondition1D` union at the bottom of
  `bc_1d.py` (it types `BoundaryConditionSet`'s fields).
- **A set factory** is a `@classmethod` on `BoundaryConditionSet`
  (`bc_set.py`); remember the south, north, west, east application order
  (west / east overwrite the corners).
- **A functional filler** goes in `boundary.py` and should be a thin
  wrapper over a `BoundaryConditionSet` where possible (as
  `enforce_periodic` and `pad_interior` are).
- Elliptic solvers accept a `BoundaryConditionSet` only when all four faces
  are `Dirichlet1D`, zero `Neumann1D` or `Periodic1D` (`_resolve_bc` in
  `solvers/elliptic.py`); a new atom raises there with a clear message.
  Don't extend that mapping unless the spectral solver has a matching
  transform.

## 3. Export and document

`finitevolx/__init__.py` (import + `__all__` under `# Boundary condition
sets and 1D BCs` or `# Boundary helpers`), `::: finitevolx.NewBC` on
`docs/api/boundary_conditions.md`, the formula on
`docs/boundary_conditions.md`, `make capabilities`.

## 4. Tests

In `tests/test_boundary.py` (atoms: `TestBoundaryConditionAtoms` or a
class of its own, as `TestRobin1D` / `TestSlip1D`): the ghost value on each
of the four faces against a hand computation with `dx != dy`, that the
interior and the other faces are untouched, the validation error, and that
the face-centred condition holds (e.g. `0.5 * (ghost + interior) == value`).
`tests/test_boundary_helpers.py` for functional fillers;
`tests/test_elliptic_bcset.py` if it interacts with the elliptic wrappers.

## 5. Verify

```bash
uv run pytest -n 2 tests/test_boundary.py tests/test_boundary_helpers.py \
  tests/test_elliptic_bcset.py tests/test_open_walls.py tests/test_capabilities.py
```

Then the `pre-pr-check` skill.
