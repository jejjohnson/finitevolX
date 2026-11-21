"""One axis-generic upwind reconstruction (#464).

Every ``Reconstruction1D/2D/3D`` method (including the masked ones and the four
TVD limiters) and the old ``reconstruct(q, u, dim, method, num_pts)`` call were
recorded before the rewrite in ``data/reconstruction_frozen.npz``; they must
still return the recorded arrays bit for bit.
"""

from __future__ import annotations

import warnings

import _reconstruction_cases as cases
import gridax
import gridax.advection as gadv
import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # the gridax-numpy-only CI job
    jax = jnp = None

FROZEN = np.load(cases.FROZEN)
CALLS = dict(cases.cases())
RECORDED = sorted(name for name in CALLS if name in FROZEN.files)


def test_every_recorded_call_still_exists() -> None:
    assert set(FROZEN.files) <= set(CALLS)
    assert len(FROZEN.files) == 110


@pytest.mark.parametrize("name", RECORDED)
def test_calls_equal_the_record(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        got = np.asarray(CALLS[name]())
    np.testing.assert_array_equal(got, FROZEN[name])


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_reconstruction3d_gains_upwind2_and_upwind3(order: int, axis: str) -> None:
    # new in #464: on every z-level they equal the 2-D methods
    rng = np.random.default_rng(order)
    h, u = rng.standard_normal((2, 5, 10, 12))
    r3 = gridax.Reconstruction3D(
        grid=gridax.CartesianGrid3D.from_interior(10, 8, 3, 1.0, 1.0, 1.0)
    )
    r2 = gridax.Reconstruction2D(
        grid=gridax.CartesianGrid2D.from_interior(10, 8, 1.0, 1.0)
    )
    got = getattr(r3, f"upwind{order}_{axis}")(h, u)
    for level in range(1, 4):
        want = getattr(r2, f"upwind{order}_{axis}")(h[level], u[level])
        np.testing.assert_array_equal(got[level], want)
    np.testing.assert_array_equal(got[[0, -1]], 0.0)


@pytest.mark.parametrize(
    "scheme",
    [
        gadv.Upwind(1),
        gadv.Upwind(3),
        gadv.Centred(),
        gadv.WENO(5),
        gadv.WENO(5, "z"),
        gadv.TVD("van_leer"),
    ],
    ids=repr,
)
@pytest.mark.parametrize("axis", [-1, -2, -3])
def test_any_axis_of_a_3d_array(scheme, axis: int) -> None:
    # along z too, which no class method offers: the same as moving z last
    rng = np.random.default_rng(-axis)
    q, u = rng.standard_normal((2, 8, 9, 10))
    got = gadv.reconstruct(q, u, axis=axis, scheme=scheme)
    moved = gadv.reconstruct(
        np.moveaxis(q, axis, -1), np.moveaxis(u, axis, -1), axis=-1, scheme=scheme
    )
    np.testing.assert_array_equal(got, np.moveaxis(moved, -1, axis))


@pytest.mark.parametrize(
    ("scheme", "order"),
    [
        (gadv.WENO(5), 5),
        (gadv.WENO(5, "z"), 5),
        (gadv.Upwind(3), 3),
        (gadv.Upwind(5), 5),
    ],
    ids=repr,
)
def test_orders_away_from_the_walls(scheme, order: int) -> None:
    # cell averages of sin(2 pi x + 0.7) on [0, 1], positive velocity; faces 6 from a wall
    def error(n: int) -> float:
        dx = 1.0 / (n - 2)
        edges = (np.arange(n + 1) - 1) * dx
        averages = (
            np.cos(2 * np.pi * edges[:-1] + 0.7) - np.cos(2 * np.pi * edges[1:] + 0.7)
        ) / (2 * np.pi * dx)
        flux = gadv.reconstruct(averages, np.ones(n), axis=-1, scheme=scheme)[1:-1]
        exact = np.sin(2 * np.pi * edges[2:-1] + 0.7)
        return np.abs(flux - exact)[6:-6].max()

    assert np.log2(error(66) / error(130)) == pytest.approx(order, abs=0.3)


def test_old_signature_warns_and_matches() -> None:
    q, u = np.random.default_rng(1).standard_normal((2, 10, 12))
    with pytest.warns(DeprecationWarning, match="axis=-1"):
        old = gridax.reconstruct(q, u, 1, "wenoz", 5)
    np.testing.assert_array_equal(
        old, gadv.reconstruct(q, u, axis=-1, scheme=gadv.WENO(5, "z"))
    )


def test_argument_errors() -> None:
    q = np.zeros((6, 6))
    with pytest.raises(TypeError, match="either"):
        gadv.reconstruct(q, q, 1, axis=-1, scheme=gadv.WENO(5))
    with pytest.raises(TypeError, match="axis"):
        gadv.reconstruct(q, q, scheme=gadv.WENO(5))
    with pytest.raises(ValueError, match="WENO"):
        gadv.WENO(7, "z")
    with pytest.raises(ValueError, match="limiter"):
        gadv.TVD("koren")
    with pytest.raises(ValueError, match="order"):
        gadv.Upwind(4)  # type: ignore[arg-type]


def test_tier_sizes_are_the_mask_hierarchy() -> None:
    # the sizes Advection*D uses for get_adaptive_masks: 2, 4, 6, 8, 10
    assert [tier.size for tier in gadv.WENO(9).tiers] == [10, 8, 6, 4, 2]
    assert [tier.size for tier in gadv.Upwind(2).tiers] == [4, 2]
    assert [tier.size for tier in gadv.TVD().tiers] == [4, 2]


@pytest.mark.skipif(jax is None, reason="JAX is not installed")
@pytest.mark.parametrize("scheme", [gadv.WENO(5), gadv.TVD("superbee")], ids=repr)
def test_jit_matches_eager(scheme) -> None:
    rng = np.random.default_rng(2)
    q, u = (jnp.asarray(a, dtype=jnp.float32) for a in rng.standard_normal((2, 10, 12)))

    def flux(q, u):
        return gadv.reconstruct(q, u, axis=-2, scheme=scheme)

    np.testing.assert_allclose(
        np.asarray(jax.jit(flux)(q, u)), np.asarray(flux(q, u)), rtol=1e-5, atol=1e-6
    )
