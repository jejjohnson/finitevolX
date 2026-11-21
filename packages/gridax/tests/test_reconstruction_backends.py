"""Limiters, reconstructions, upwind fluxes and reductions run on any backend.

Gate G1 of #435 for step 3: each function returns NumPy for NumPy inputs, in
the input's dtype, and equals its JAX result under ``jax.jit``. The JAX half
skips when JAX is not installed.
"""

from __future__ import annotations

import inspect

import gridax
import numpy as np
import pytest

try:
    import jax
except ImportError:  # the gridax-numpy-only CI job
    jax = None

needs_jax = pytest.mark.skipif(jax is None, reason="needs JAX")

POINTWISE = [
    n
    for n in gridax.__all__
    if n.startswith(("linear_", "weno_"))
    or n in ("minmod", "van_leer", "superbee", "mc")
]


@pytest.fixture(autouse=True)
def _x64():
    if jax is None:
        yield
        return
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _fields(n: int, shape=(6, 7), seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape) for _ in range(n)]


def _pointwise_args(name: str) -> list[np.ndarray]:
    n = len(inspect.signature(getattr(gridax, name)).parameters)
    args = _fields(n)
    if name in ("minmod", "van_leer", "superbee", "mc"):
        args = [np.linspace(-3.0, 3.0, 13)]
    return args


def test_pointwise_family_is_covered():
    assert len(POINTWISE) == 4 + 7 + 12


@pytest.mark.parametrize("name", POINTWISE)
def test_pointwise_on_numpy_keeps_dtype(name):
    fn = getattr(gridax, name)
    for dtype in (np.float32, np.float64):
        out = fn(*(a.astype(dtype) for a in _pointwise_args(name)))
        assert isinstance(out, np.ndarray) and out.dtype == dtype, name


@needs_jax
@pytest.mark.parametrize("name", POINTWISE)
def test_pointwise_numpy_equals_jax(name):
    fn = getattr(gridax, name)
    args = _pointwise_args(name)
    want = fn(*args)
    got = jax.jit(fn)(*(jax.numpy.asarray(a) for a in args))
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=1e-15)


RECONSTRUCT_CASES = [
    ("weno", 3),
    ("weno", 5),
    ("wenoz", 5),
    ("linear", 1),
    ("linear", 5),
]


def _reconstruct(q, u, method, num_pts):
    # the old (method, num_pts) cases, as schemes of gridax.advection (#464)
    import gridax.advection as gadv

    if num_pts == 1:
        scheme = gadv.Upwind(1)
    elif method == "linear":
        scheme = gadv.Upwind(num_pts)
    else:
        scheme = gadv.WENO(num_pts, "z" if method == "wenoz" else "js")
    return gridax.reconstruct(q, u, axis=-1, scheme=scheme)


@pytest.mark.parametrize(("method", "num_pts"), RECONSTRUCT_CASES)
def test_reconstruct_on_numpy(method, num_pts):
    q, u = _fields(2, (8, 12))
    out = _reconstruct(q, u, method, num_pts)
    assert isinstance(out, np.ndarray) and out.shape == q.shape


@needs_jax
@pytest.mark.parametrize(("method", "num_pts"), RECONSTRUCT_CASES)
def test_reconstruct_numpy_equals_jax(method, num_pts):
    q, u = _fields(2, (8, 12))
    want = _reconstruct(q, u, method, num_pts)
    got = jax.jit(lambda a, b: _reconstruct(a, b, method, num_pts))(
        jax.numpy.asarray(q), jax.numpy.asarray(u)
    )
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=1e-15)


def _upwind(q, u, mask):
    hierarchy = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
    # full-size face fluxes, one per tier, so the tier selection is what is compared
    rec = {s: (lambda q, u, s=s: s * q * u) for s in (2, 4, 6)}
    return gridax.upwind_flux(q, u, dim=1, rec_funcs=rec, mask_hierarchy=hierarchy)


def _coastal_mask(shape):
    wet = np.ones(shape, dtype=bool)
    wet[3:5, 4:7] = False
    return gridax.Mask2D.from_mask(wet)


def test_upwind_flux_on_numpy():
    q, u = _fields(2, (9, 12))
    out = _upwind(q, u, _coastal_mask(q.shape))
    assert isinstance(out, np.ndarray) and out.shape == q.shape


@needs_jax
def test_upwind_flux_numpy_equals_jax():
    q, u = _fields(2, (9, 12))
    mask = _coastal_mask(q.shape)
    want = _upwind(q, u, mask)
    got = jax.jit(_upwind)(jax.numpy.asarray(q), jax.numpy.asarray(u), mask)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=1e-15)


GRIDS = {
    "cartesian": lambda: gridax.CartesianGrid2D.from_interior(10, 7, 2.0, 1.5),
    "spherical": lambda: gridax.SphericalGrid2D.from_interior(
        10, 7, (0, 40), (-20, 30)
    ),
}


def _reductions(field, grid, mask):
    return (
        gridax.area_sum(field, grid, mask),
        gridax.area_mean(field, grid, mask),
    )


@pytest.mark.parametrize("kind", sorted(GRIDS))
def test_reductions_on_numpy_keep_dtype(kind):
    grid = GRIDS[kind]()
    mask = _coastal_mask((grid.Ny, grid.Nx))
    for dtype in (np.float32, np.float64):
        field = _fields(1, (grid.Ny, grid.Nx))[0].astype(dtype)
        for out in _reductions(field, grid, mask):
            assert np.asarray(out).dtype == dtype
            assert not hasattr(out, "devices")  # NumPy in, NumPy out
    weights = gridax.area_weights(grid)
    assert isinstance(weights, np.ndarray)


@needs_jax
@pytest.mark.parametrize("kind", sorted(GRIDS))
def test_reductions_numpy_equals_jax(kind):
    grid = GRIDS[kind]()
    mask = _coastal_mask((grid.Ny, grid.Nx))
    field = _fields(1, (grid.Ny, grid.Nx))[0]
    want = _reductions(field, grid, mask)
    got = jax.jit(_reductions)(jax.numpy.asarray(field), grid, mask)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(float(g), float(w), rtol=1e-13)
    samples = np.stack(_fields(4, (grid.Ny, grid.Nx), seed=2))
    for fn in (gridax.masked_mean, gridax.masked_std):
        np.testing.assert_allclose(
            np.asarray(fn(jax.numpy.asarray(samples), mask)),
            fn(samples, mask),
            rtol=1e-12,
        )
