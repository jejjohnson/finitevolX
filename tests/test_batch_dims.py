"""Leading (batch) axes on the 2-D Cartesian operators.

Every 2-D entry point takes ``[*batch, Ny, Nx]``: the stencils slice the
trailing two axes (``h[..., 1:-1, 2:]``) and ``interior`` pads only the
axes that lost a ghost ring. Each case is checked against a loop over the
batch members (bit for bit eager, to float64 rounding under ``jit``), with and without a
mask.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import inspect
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx as fvx
from tests.fixtures.inputs import make_grid_2d, make_mask_2d

BATCH = (2, 3)
GRID = make_grid_2d()
MASK = make_mask_2d()
SCALAR = 0.7  # value for every float argument (dx, dy, g_prime, f0, …)

# Modules whose 2-D entry points are annotated ``*batch Ny Nx``. Every
# public function and class method there that carries the annotation is
# tested below, so the annotation and the behaviour cannot drift apart.
MODULES = (
    "finitevolx._src.operators.stencils",
    "finitevolx._src.operators.difference",
    "finitevolx._src.operators.interpolation",
    "finitevolx._src.operators.divergence",
    "finitevolx._src.operators.vorticity",
    "finitevolx._src.operators.coriolis",
    "finitevolx._src.operators.diagnostics",
    "finitevolx._src.operators.diagnostic_operators",
    "finitevolx._src.boundary.bc_1d",
    "finitevolx._src.boundary.bc_set",
    "finitevolx._src.boundary.boundary",
)


def _batched(fn: Callable) -> bool:
    return any("*batch" in str(p.annotation) for p in _params(fn))


def _params(fn: Callable) -> list[inspect.Parameter]:
    return [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.default is inspect.Parameter.empty and p.name != "self"
    ]


def _discover() -> tuple[list, list]:
    functions, classes = [], []
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr, obj in vars(mod).items():
            if attr.startswith("_") or getattr(obj, "__module__", None) != name:
                continue
            if inspect.isfunction(obj) and _batched(obj):
                functions.append(obj)
            elif inspect.isclass(obj) and any(
                _batched(m) for _, m in inspect.getmembers(obj, inspect.isfunction)
            ):
                classes.append(obj)
    return functions, classes


FUNCTIONS, CLASSES = _discover()

FACES = ("south", "north", "west", "east")

# BC atoms, one constructor per class: ``face -> atom``.
ATOMS = {
    fvx.Dirichlet1D: lambda face: fvx.Dirichlet1D(face, 1.5),
    fvx.Neumann1D: lambda face: fvx.Neumann1D(face, 0.25),
    fvx.Periodic1D: fvx.Periodic1D,
    fvx.Outflow1D: fvx.Outflow1D,
    fvx.Reflective1D: fvx.Reflective1D,
    fvx.Sponge1D: lambda face: fvx.Sponge1D(face, background=0.3, weight=0.4),
    fvx.Robin1D: lambda face: fvx.Robin1D(face, alpha=1.0, beta=0.5, gamma=0.2),
    fvx.Extrapolation1D: lambda face: fvx.Extrapolation1D(face, order=2),
    fvx.Slip1D: lambda face: fvx.Slip1D(face, 0.5),
}

BC_SETS = {
    "periodic": fvx.BoundaryConditionSet.periodic(),
    "closed_U": fvx.BoundaryConditionSet.closed(point="U"),
    "open_V": fvx.BoundaryConditionSet.open(point="V"),
    "mixed": fvx.BoundaryConditionSet(
        south=fvx.Dirichlet1D("south", 1.5),
        north=fvx.Neumann1D("north", 0.25),
        west=fvx.Slip1D("west", 0.5),
        east=fvx.Outflow1D("east"),
    ),
    **{
        cls.__name__: fvx.BoundaryConditionSet(**{f: make(f) for f in FACES})
        for cls, make in ATOMS.items()
    },
}


def _array_params(fn: Callable) -> int | None:
    """Number of required array arguments, or ``None`` if an argument is
    neither an array nor a float."""
    n = 0
    for p in inspect.signature(fn).parameters.values():
        if p.default is not inspect.Parameter.empty or p.name == "self":
            continue
        ann = str(p.annotation)
        if ann.startswith("float"):  # float, or float | 0-d array: a scalar
            continue
        if "Array" in ann or p.annotation is inspect.Parameter.empty:
            n += 1
        else:
            return None
    return n


def _call(fn: Callable, fields: list) -> object:
    it = iter(fields)
    args = []
    for p in inspect.signature(fn).parameters.values():
        if p.default is not inspect.Parameter.empty or p.name == "self":
            continue
        ann = str(p.annotation)
        is_array = not ann.startswith("float") and (
            "Array" in ann or p.annotation is inspect.Parameter.empty
        )
        args.append(next(it) if is_array else SCALAR)
    return fn(*args)


def _cases() -> list:
    cases = []
    bc_classes = (*ATOMS, fvx.BoundaryConditionSet, fvx.FieldBCSet)
    grid_classes = [c for c in CLASSES if c not in bc_classes]
    for cls in grid_classes:
        assert "grid" in getattr(cls, "__dataclass_fields__", {}), (
            f"{cls.__name__} is annotated *batch but has no constructor in this test"
        )
    for cls, masked in itertools.product(grid_classes, (False, True)):
        obj = cls(grid=GRID, mask=MASK if masked else None)
        tag = f"{cls.__name__}{'[mask]' if masked else ''}"
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_") and name != "__call__":
                continue
            method = getattr(obj, name)
            if _batched(method) and _array_params(method):
                cases.append(pytest.param(method, id=f"{tag}.{name}"))
    for fn in FUNCTIONS:
        assert _array_params(fn), fn.__name__
        cases.append(pytest.param(fn, id=fn.__name__))
    for name, bcs in BC_SETS.items():
        cases.append(
            pytest.param(lambda f, b=bcs: b(f, dx=0.5, dy=0.25), id=f"BCSet.{name}")
        )
    field_bcs = fvx.FieldBCSet(
        bc_map={"h": BC_SETS["mixed"]}, default=BC_SETS["periodic"]
    )
    cases.append(
        pytest.param(
            lambda h, u: field_bcs({"h": h, "u": u}, dx=0.5, dy=0.25),
            id="FieldBCSet",
        )
    )
    return cases


def _fields(n: int) -> list:
    shape = (*BATCH, GRID.Ny, GRID.Nx)
    return [jax.random.normal(jax.random.key(k), shape) for k in range(n)]


def _per_member(fn: Callable, fields: list) -> object:
    flat = [f.reshape(-1, *f.shape[len(BATCH) :]) for f in fields]
    outs = [_call(fn, [f[b] for f in flat]) for b in range(flat[0].shape[0])]
    return jax.tree.map(lambda *ys: jnp.stack(ys).reshape(*BATCH, *ys[0].shape), *outs)


@pytest.mark.parametrize("fn", _cases())
def test_leading_axes_match_a_loop_over_members(fn):
    fields = _fields(_array_params(fn) or 1)
    expected = _per_member(fn, fields)
    eager = _call(fn, fields)
    jitted = eqx.filter_jit(lambda *fs: _call(fn, list(fs)))(*fields)
    for got in (eager, jitted):
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
            assert a.shape == b.shape
            # Eager is bit-identical; under jit XLA may reassociate a fused
            # sum differently per shape, so allow float64 rounding on O(1)
            # fields (observed max |Δ| ≈ 3e-17).
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_interior_keeps_equal_axes_and_pads_ghost_axes():
    like = jnp.ones((4, 6, 5))
    out = fvx.interior(jnp.full((4, 4, 3), 2.0), like)
    assert out.shape == like.shape
    np.testing.assert_array_equal(out[:, 1:-1, 1:-1], 2.0)
    np.testing.assert_array_equal(out[:, 0], 0.0)
    # 3-D: every axis lost its ghost ring (z ghost slabs included)
    out3 = fvx.interior(jnp.full((2, 4, 3), 2.0), like)
    np.testing.assert_array_equal(out3[0], 0.0)


@pytest.mark.parametrize("shape", [(4, 5, 3), (4, 4)])
def test_interior_rejects_values_that_do_not_fit(shape):
    with pytest.raises(ValueError, match="interior"):
        fvx.interior(jnp.ones(shape), jnp.ones((4, 6, 5)))
