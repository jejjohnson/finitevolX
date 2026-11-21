"""The raw stencils, ``interior`` and the smooth surrogates run on any backend.

Gate G1 of #435 for this layer: every stencil gives the same numbers on
NumPy and on JAX, the result stays in the input's namespace and dtype, and
the backend hooks keep JAX's own calls on JAX inputs. The JAX half skips when
JAX is not installed (the ``gridax-numpy-only`` CI job).
"""

from __future__ import annotations

import gridax
from gridax._src import backend
from gridax._src.operators._ghost import zero_z_ghosts
from gridax._src.operators._utils import _metric_like, _safe_div_cos
from gridax._src.operators.stencils import _materialize, _neighbours
import numpy as np
import pytest

try:
    import jax
except ImportError:  # the gridax-numpy-only CI job
    jax = None

needs_jax = pytest.mark.skipif(jax is None, reason="needs JAX")

STENCILS = [n for n in gridax.__all__ if n.startswith(("diff_", "avg_"))]
RANK = {n: int(n[-2]) for n in STENCILS}
SPATIAL = {1: (9,), 2: (7, 9), 3: (5, 7, 9)}


@pytest.fixture(autouse=True)
def _x64():
    if jax is None:
        yield
        return
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _field(*shape: int) -> np.ndarray:
    return np.random.default_rng(len(shape)).standard_normal(shape)


def _cases():
    for name in STENCILS:
        yield name, (2, *SPATIAL[RANK[name]])  # one leading batch axis


def test_every_name_follows_the_rule():
    """``<op>_<axis>_<side>_<d>d``: 4 stencils in 1-D, 12 in 2-D, 16 in 3-D."""
    counts = {d: sum(RANK[n] == d for n in STENCILS) for d in (1, 2, 3)}
    assert counts == {1: 4, 2: 12, 3: 16}
    for name in STENCILS:
        assert name.endswith(("_1d", "_2d", "_3d")), name


@pytest.mark.parametrize("name", STENCILS)
def test_trims_every_spatial_axis_and_keeps_batch_axes(name):
    """Every spatial axis loses its ghost ring; leading axes are batch axes."""
    d = RANK[name]
    h = _field(3, 2, *SPATIAL[d])
    out = getattr(gridax, name)(h)
    assert out.shape == (3, 2, *(n - 2 for n in SPATIAL[d]))
    np.testing.assert_array_equal(out[1, 0], getattr(gridax, name)(h[1, 0]))


@pytest.mark.parametrize(("name", "shape"), list(_cases()))
def test_stencil_on_numpy_stays_numpy(name, shape):
    h = _field(*shape)
    out = getattr(gridax, name)(h)
    assert isinstance(out, np.ndarray) and out.dtype == np.float64
    padded = gridax.interior(out, h)
    assert isinstance(padded, np.ndarray) and padded.shape == h.shape


@needs_jax
@pytest.mark.parametrize(("name", "shape"), list(_cases()))
def test_stencil_numpy_equals_jax(name, shape):
    h = _field(*shape)
    fn = getattr(gridax, name)
    want = gridax.interior(fn(h), h)
    got = jax.jit(lambda x: gridax.interior(fn(x), x))(jax.numpy.asarray(h))
    assert isinstance(got, jax.Array)
    np.testing.assert_array_equal(np.asarray(got), want)


def test_interior_keeps_float32_and_writes_only_the_interior():
    h = np.ones((5, 6), dtype=np.float32)
    out = gridax.interior(gridax.diff_x_fwd_2d(h) + 1.0, h)
    assert out.dtype == np.float32
    assert (
        (out[1:-1, 1:-1] == 1.0).all() and out[0].sum() == 0 and out[:, -1].sum() == 0
    )


def test_zero_z_ghosts_on_numpy():
    a = np.ones((4, 3, 3))
    out = zero_z_ghosts(a)
    assert (out[[0, -1]] == 0).all() and (out[1:-1] == 1).all() and (a == 1).all()


def test_smooth_surrogates_on_numpy():
    x = np.linspace(-2.0, 2.0, 9)
    np.testing.assert_allclose(gridax.smooth_abs(x, eps=1e-12), np.abs(x), atol=1e-11)
    np.testing.assert_allclose(
        gridax.smooth_clamp(x, 0.0, sharpness=1e3), np.maximum(x, 0.0), atol=1e-3
    )
    np.testing.assert_allclose(
        gridax.smooth_max(x, 0.5 * x, sharpness=1e3), np.maximum(x, 0.5 * x), atol=1e-3
    )


@needs_jax
def test_smooth_surrogates_numpy_equals_jax_and_differentiate():
    x = np.linspace(-2.0, 2.0, 9)
    jx = jax.numpy.asarray(x)
    for fn in (
        gridax.smooth_abs,
        lambda v: gridax.smooth_clamp(v, 0.1),
        lambda v: gridax.smooth_max(v, 0.3 * v),
    ):
        np.testing.assert_allclose(np.asarray(fn(jx)), fn(x), rtol=1e-12)
        grad = jax.grad(lambda v, fn=fn: fn(v).sum())(jx)
        assert np.isfinite(np.asarray(grad)).all()


def test_spherical_helpers_follow_the_field():
    cos = np.cos(np.linspace(-1.0, 1.0, 5))
    f32 = np.ones(5, dtype=np.float32)
    assert _metric_like(cos, f32).dtype == np.float32
    out = _safe_div_cos(f32, _metric_like(cos, f32), 2.0)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32


def test_hooks_on_numpy_do_not_mutate():
    a = np.zeros(4)
    b = backend.set_at(a, slice(1, 3), 2.0)
    assert (a == 0).all() and (b == [0, 2, 2, 0]).all()
    assert backend.pad(np.ones((2, 2)), ((1, 0), (0, 1))).shape == (3, 3)
    assert backend.optimization_barrier(a) is a and backend.stop_gradient(a) is a
    nb = _neighbours(_field(5, 6))
    assert nb.c.shape == (3, 4)
    assert _materialize(a) is a


@needs_jax
def test_hooks_keep_the_jax_calls():
    """On JAX inputs the hooks lower to JAX's own primitives."""
    x = jax.numpy.ones((4, 4))
    jaxpr = str(
        jax.make_jaxpr(
            lambda v: backend.optimization_barrier(backend.pad(v, ((1, 1), (1, 1))))
        )(x)
    )
    assert "optimization_barrier" in jaxpr and "pad" in jaxpr
    scatter = str(jax.make_jaxpr(lambda v: backend.set_at(v, (slice(1, 3),), 1.0))(x))
    assert "scatter" in scatter or "dynamic_update_slice" in scatter
