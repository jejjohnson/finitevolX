"""Gate G1 of #435: the golden fixtures hold on NumPy inputs too.

Every golden entry is rebuilt with the canonical input fields as NumPy
arrays instead of JAX arrays and compared with its committed ``.npz``. For
an operator that lives in gridax (array-API code), the result must also stay
NumPy; an operator still in gridax-jax may convert to JAX, and only its
numbers are checked.
"""

from __future__ import annotations

import inspect

import gridax
import numpy as np
import pytest

from tests.fixtures import _gen_golden, inputs
from tests.fixtures._helpers import assert_matches_golden


def _is_field_factory(name: str) -> bool:
    """Factories of input arrays: fields, coordinates, the stretching matrix."""
    return (
        name.startswith("make_")
        and "grid" not in name
        and not name.startswith("make_mask")
    )


FIELD_FACTORIES = [
    name
    for name, _ in inspect.getmembers(inputs, inspect.isfunction)
    if _is_field_factory(name)
]


def _numpy_entries():
    """The golden entries, built while every field factory returns NumPy.

    The generator binds some factories at import and imports others inside
    its entry builders, so both modules are patched.
    """
    patched = []
    for module in (inputs, _gen_golden):
        for name in FIELD_FACTORIES:
            original = getattr(module, name, None)
            if original is None:
                continue
            patched.append((module, name, original))
            setattr(module, name, lambda original=original: np.asarray(original()))
    try:
        return _gen_golden._register_all()
    finally:
        for module, name, original in patched:
            setattr(module, name, original)


ENTRIES = _numpy_entries()


def _in_gridax(operator: str) -> bool:
    obj = getattr(gridax, operator, None)
    return obj is not None and obj.__module__.startswith("gridax.")


def test_fields_are_numpy():
    assert FIELD_FACTORIES
    assert ENTRIES


@pytest.mark.parametrize(
    "entry", ENTRIES, ids=[f"{o}.{m}.{v}" for o, m, v, _ in ENTRIES]
)
def test_golden_on_numpy_inputs(entry):
    operator, method, variant, fn = entry
    out = fn()
    assert_matches_golden(out, operator, method, variant)
    if _in_gridax(operator):
        leaves = out if isinstance(out, tuple) else (out,)
        for leaf in leaves:
            assert isinstance(leaf, np.ndarray), f"{operator}.{method} left NumPy"
