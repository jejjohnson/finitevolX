"""The deprecated finitevolX names of the time steppers that moved to integrix.

The generic steppers live in ``integrix.march``. For one release,
``finitevolx.<name>`` still works and warns: the diffrax solvers and the pure
step functions are the integrix objects themselves, and ``AB2Solver``,
``LeapfrogRAFSolver`` and ``SplitExplicitRKSolver`` are adapters that keep
finitevolX's old call shapes and reproduce the integrix steppers.
"""

from __future__ import annotations

import warnings

import integrix.march as ixm
import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx
from finitevolx._src.timestepping import _deprecated

MOVED = list(_deprecated.MOVED)
ADAPTED = list(_deprecated.ADAPTED)


@pytest.mark.parametrize("name", MOVED)
def test_moved_name_warns_and_is_the_integrix_object(name: str) -> None:
    with pytest.warns(DeprecationWarning, match=rf"integrix\.march\.{name}\b"):
        obj = getattr(finitevolx, name)
    assert obj is getattr(ixm, name)


def test_from_import_warns() -> None:
    with pytest.warns(DeprecationWarning, match=r"integrix\.march\.RK3SSP"):
        from finitevolx import RK3SSP
    assert RK3SSP is ixm.RK3SSP


@pytest.mark.parametrize("name", ADAPTED)
def test_adapted_name_warns_and_names_the_new_interface(name: str) -> None:
    with pytest.warns(DeprecationWarning, match=r"integrix\.march\.run") as record:
        obj = getattr(finitevolx, name)
    assert f"integrix.march.{name}" in str(record[0].message)
    assert obj is getattr(_deprecated, name)


def test_names_that_stay_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in (
            "SemiLagrangianSolver",
            "SemiLagrangianTerm",
            "semi_lagrangian_step",
            "solve_ocean_pde",
        ):
            getattr(finitevolx, name)


def test_unknown_name_raises() -> None:
    with pytest.raises(AttributeError, match="no_such_stepper"):
        finitevolx.no_such_stepper  # noqa: B018


def _march(solver, rhs, y0, dt, n_steps):
    """Run an adapter with its old ``init`` / ``step`` loop."""
    solver, y = solver.init(rhs, 0.0, y0, dt)
    for n in range(1, n_steps):
        y, solver = solver.step(rhs, n * dt, y, dt)
    return y


@pytest.mark.parametrize(
    ("adapter", "stepper"),
    [
        (_deprecated.AB2Solver(), ixm.AB2Solver()),
        (_deprecated.LeapfrogRAFSolver(alpha=0.05), ixm.LeapfrogRAFSolver(alpha=0.05)),
    ],
    ids=["AB2Solver", "LeapfrogRAFSolver"],
)
def test_multistep_adapter_reproduces_integrix(adapter, stepper) -> None:
    # y' = -y over [0, 1]
    y0, dt, n_steps = jnp.array(1.0), 0.01, 100
    y_old = _march(adapter, lambda t, y: -y, y0, dt, n_steps)
    y_new = ixm.run(stepper, lambda t, y, args: -y, 0.0, y0, dt, n_steps)
    np.testing.assert_allclose(y_old, y_new, rtol=1e-12)


def test_split_explicit_adapter_reproduces_integrix() -> None:
    def slow(t, y3, mean_2d):
        return -0.5 * y3 - 0.5 * mean_2d

    def fast(t, y2, y3):
        return -y2 + 0.1 * y3

    def forcing(y3):
        return 0.2 * y3

    def couple(y3, mean_2d):
        return y3 + 0.01 * mean_2d

    y3 = y2 = jnp.array(1.0)
    adapter = _deprecated.SplitExplicitRKSolver(n_substeps=5)
    for k in range(10):
        y3, y2 = adapter.step(
            slow,
            fast,
            0.1 * k,
            y3,
            y2,
            0.1,
            slow_forcing=forcing,
            couple_2d_to_3d=couple,
        )
    terms = ixm.SplitExplicitTerms(
        slow=lambda t, a, m, args: slow(t, a, m),
        fast=lambda t, a, b, args: fast(t, a, b),
        slow_forcing=lambda t, a, args: forcing(a),
        couple=lambda a, m, args: couple(a, m),
    )
    new3, new2 = ixm.run(
        ixm.SplitExplicitRKSolver(n_substeps=5),
        terms,
        0.0,
        (jnp.array(1.0), jnp.array(1.0)),
        0.1,
        10,
    )
    np.testing.assert_allclose(y3, new3, rtol=1e-12)
    np.testing.assert_allclose(y2, new2, rtol=1e-12)
