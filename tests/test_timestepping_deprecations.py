"""The deprecated finitevolX names of the time steppers that moved to integrix.

The generic steppers live in ``integrix.march``. For one release,
``finitevolx.<name>`` still works and warns: the diffrax solvers and the pure
step functions are the integrix objects themselves, and ``AB2Solver``,
``LeapfrogRAFSolver`` and ``SplitExplicitRKSolver`` are adapters that keep
finitevolX's old call shapes and reproduce the integrix steppers.
"""

from __future__ import annotations

import warnings

import gridax_jax
from gridax_jax._src.timestepping import _deprecated
import integrix.march as ixm
import jax.numpy as jnp
import numpy as np
import pytest

MOVED = list(_deprecated.MOVED)
ADAPTED = list(_deprecated.ADAPTED)


@pytest.mark.parametrize("name", MOVED)
def test_moved_name_warns_and_is_the_integrix_object(name: str) -> None:
    with pytest.warns(DeprecationWarning, match=rf"integrix\.march\.{name}\b"):
        obj = getattr(gridax_jax, name)
    assert obj is getattr(ixm, name)


def test_from_import_warns() -> None:
    with pytest.warns(DeprecationWarning, match=r"integrix\.march\.RK3SSP"):
        from gridax_jax import RK3SSP
    assert RK3SSP is ixm.RK3SSP


@pytest.mark.parametrize("name", ADAPTED)
def test_adapted_name_warns_and_names_the_new_interface(name: str) -> None:
    with pytest.warns(DeprecationWarning, match=r"integrix\.march\.run") as record:
        obj = getattr(gridax_jax, name)
    assert f"integrix.march.{name}" in str(record[0].message)
    assert obj is getattr(_deprecated, name)


@pytest.mark.parametrize("name", list(_deprecated.SEMI_LAGRANGIAN))
def test_semi_lagrangian_name_warns_and_names_integrix(name: str) -> None:
    # rewritten in integrix.march (#476); the old object stays for one release
    with pytest.warns(DeprecationWarning, match=r"integrix\.march\.") as record:
        obj = getattr(gridax_jax, name)
    assert f"integrix.march.{name}" in str(record[0].message)
    assert obj is getattr(_deprecated._semi_lagrangian, name)


def test_semi_lagrangian_step_equals_integrix_with_euler_points() -> None:
    """integrix's step, Euler departure points and a linear functax read-out,
    is the old step (periodic) to 1e-13 relative (#476)."""
    from functax.geometry import Axis, Points, RectilinearGrid
    from functax.interp import Linear

    grid = gridax_jax.CartesianGrid2D.from_interior(14, 10, 2.8, 1.0)
    ny, nx = grid.Ny, grid.Nx
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((ny, nx)))
    u = jnp.asarray(0.3 + 0.1 * np.sin(2 * np.pi * cols / nx))
    v = jnp.asarray(-0.2 + 0.05 * np.cos(2 * np.pi * rows / ny))
    dt = 0.37
    with pytest.warns(DeprecationWarning):
        old_step = gridax_jax.semi_lagrangian_step
    expected = old_step(q, u, v, grid.dx, grid.dy, dt, bc="periodic")

    nodes = jnp.asarray(gridax_jax.t_point_coordinates(grid))
    # the old step wraps over the whole array, ghost ring included
    periodic = RectilinearGrid(
        (
            Axis(jnp.arange(ny) * grid.dy, period=ny * grid.dy),
            Axis(jnp.arange(nx) * grid.dx, period=nx * grid.dx),
        )
    )
    speed = jnp.column_stack([v.ravel(), u.ravel()])
    actual = ixm.semi_lagrangian_step(
        q.ravel(),
        nodes,
        lambda t, x: speed,
        lambda x_d: Linear().plan(periodic, Points(x_d)),
        0.0,
        dt,
        iterations=0,
    ).reshape(ny, nx)
    scale = float(jnp.max(jnp.abs(expected)))
    assert float(jnp.max(jnp.abs(actual - expected))) <= 1e-13 * scale


def test_names_that_stay_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gridax_jax.t_point_coordinates  # noqa: B018


def test_unknown_name_raises() -> None:
    with pytest.raises(AttributeError, match="no_such_stepper"):
        gridax_jax.no_such_stepper  # noqa: B018


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
