"""gridax.analysis differences: the geoops kernel tests, on NumPy and JAX.

Ported from geotoolz-ops (``tests/test_gradient.py``, ``tests/test_calculus.py``),
where they ran against the kernels moved here (#423); the carrier and
Operator tests stay there. finitediffx's numbers come from
``data/finitediffx_reference.npz``, written once with finitediffx 0.1.0
(the generator script is quoted in that file's test below), so gridax does
not depend on it. Tests that need JAX skip without it: the
``gridax-numpy-only`` CI job runs the rest.
"""

from __future__ import annotations

import functools
from pathlib import Path

from gridax.analysis import (
    adaptive_curl_2d,
    adaptive_curl_3d,
    adaptive_difference,
    adaptive_divergence_1d,
    adaptive_divergence_2d,
    adaptive_divergence_3d,
    adaptive_gradient_1d,
    adaptive_gradient_2d,
    adaptive_gradient_3d,
    adaptive_laplacian_1d,
    adaptive_laplacian_2d,
    adaptive_laplacian_3d,
    curl_2d,
    curl_3d,
    difference,
    difference_coeffs,
    difference_offsets,
    divergence_1d,
    divergence_2d,
    divergence_3d,
    gradient_1d,
    gradient_2d,
    gradient_3d,
    laplacian_1d,
    laplacian_2d,
    laplacian_3d,
    nonuniform_coeffs,
)
import numpy as np
import pytest

try:
    import jax
except ImportError:  # the gridax-numpy-only CI job
    jax = None

HAVE_JAX = jax is not None
needs_jax = pytest.mark.skipif(not HAVE_JAX, reason="needs JAX")
REFERENCE = Path(__file__).parent / "data" / "finitediffx_reference.npz"

CASES = [
    (derivative, accuracy, method)
    for derivative in (1, 2)
    for accuracy in (1, 2, 3, 4)
    for method in ("central", "forward", "backward")
]
DEPTHS = np.array([0.0, 5.0, 15.0, 30.0, 50.0, 80.0])
LEVELS = np.array([0.0, 5.0, 15.0, 30.0, 50.0, 75.0, 105.0, 140.0, 180.0])


def _jax():
    if not HAVE_JAX:
        pytest.skip("needs JAX")
    return jax


@pytest.fixture(autouse=True)
def _x64():
    if not HAVE_JAX:
        yield
        return
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@functools.cache
def _reference() -> dict[str, np.ndarray]:
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


# finitediffx reference values (geotoolz-ops' finitediffx tests)


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_difference_equals_finitediffx(
    derivative: int, accuracy: int, method: str
) -> None:
    ref = _reference()
    key = f"difference/{derivative}/{accuracy}/{method}"
    f = ref[key + "/f"]
    for axis, step in ((0, 0.3), (1, -2.0)):
        got = difference(
            f,
            axis=axis,
            step=step,
            derivative=derivative,
            accuracy=accuracy,
            method=method,
        )
        np.testing.assert_allclose(got, ref[f"{key}/axis{axis}"], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("accuracy", [1, 2, 3, 4])
def test_2d_primitives_equal_finitediffx(accuracy: int) -> None:
    ref = _reference()
    k = f"calculus2d/{accuracy}"
    u, v = ref[k + "/u"], ref[k + "/v"]
    dy, dx = 0.3, -2.0
    kw = {"accuracy": accuracy}
    np.testing.assert_allclose(
        divergence_2d(u, v, dy, dx, **kw), ref[k + "/div"], atol=1e-9
    )
    np.testing.assert_allclose(curl_2d(u, v, dy, dx, **kw), ref[k + "/curl"], atol=1e-9)
    np.testing.assert_allclose(
        laplacian_2d(u, dy, dx, **kw), ref[k + "/lap"], atol=1e-9
    )


@pytest.mark.parametrize("accuracy", [1, 2])
def test_3d_primitives_equal_finitediffx(accuracy: int) -> None:
    ref = _reference()
    k = f"calculus3d/{accuracy}"
    u, v, w = ref[k + "/u"], ref[k + "/v"], ref[k + "/w"]
    dz, dy, dx = 0.7, 0.3, -2.0
    kw = {"accuracy": accuracy}
    np.testing.assert_allclose(
        divergence_3d(u, v, w, dz, dy, dx, **kw), ref[k + "/div"], atol=1e-9
    )
    np.testing.assert_allclose(
        laplacian_3d(u, dz, dy, dx, **kw), ref[k + "/lap"], atol=1e-9
    )
    np.testing.assert_allclose(
        curl_3d(u, v, w, dz, dy, dx, **kw), ref[k + "/curl"], atol=1e-9
    )


# dtypes (#421): the caller decides


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_output_keeps_the_input_dtype(dtype) -> None:
    y, x = np.mgrid[0:6, 0:8].astype(dtype)
    assert laplacian_2d(x**2 + y**2, 0.5, 2.0).dtype == dtype
    assert adaptive_curl_2d(-y, x).dtype == dtype
    assert difference(x[0], coords=np.arange(8.0)).dtype == dtype


@needs_jax
def test_jax_output_keeps_the_input_dtype() -> None:
    jnp = jax.numpy
    y, x = np.mgrid[0:6, 0:8]
    f32 = jnp.asarray(x**2 + y**2, dtype=jnp.float32)
    f64 = jnp.asarray(x**2 + y**2, dtype=jnp.float64)
    assert laplacian_2d(f32, 0.5, 2.0).dtype == jnp.float32
    assert laplacian_2d(f64, 0.5, 2.0).dtype == jnp.float64
    assert jax.jit(adaptive_laplacian_2d)(f64).dtype == jnp.float64


# ported from geotoolz-ops tests/test_gradient.py


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_kernel_is_exact_on_a_linear_field(backend: str) -> None:
    rows, cols = np.mgrid[0:5, 0:6].astype(float)
    f = 2.0 * rows - 0.5 * cols
    if backend == "jax":
        f = _jax().numpy.asarray(f)
    d_y, d_x = gradient_2d(f, 4.0, -2.0)
    np.testing.assert_allclose(np.asarray(d_y), 0.5)  # every row, edges too
    np.testing.assert_allclose(np.asarray(d_x), 0.25)


def test_kernel_converges_at_second_order_inside() -> None:
    """f = sin(kx) cos(ly): halving the step quarters the interior error."""
    k, ell = 1.3, 0.7

    def interior_error(n: int) -> float:
        h = 2.0 / n
        y, x = np.mgrid[0 : n + 1, 0 : n + 1] * h
        f = np.sin(k * x) * np.cos(ell * y)
        d_y, d_x = gradient_2d(f, h, h)
        want_x = k * np.cos(k * x) * np.cos(ell * y)
        want_y = -ell * np.sin(k * x) * np.sin(ell * y)
        inner = (slice(1, -1), slice(1, -1))
        return max(np.abs(d_x - want_x)[inner].max(), np.abs(d_y - want_y)[inner].max())

    ratio = interior_error(32) / interior_error(64)
    assert 3.8 < ratio < 4.2


def test_kernel_propagates_nan_to_neighbours() -> None:
    f = np.zeros((5, 5))
    f[2, 2] = np.nan
    d_y, d_x = gradient_2d(f, 1.0, 1.0)
    assert np.isnan(d_y[[1, 3], 2]).all() and np.isfinite(d_y[0, 0])
    assert np.isnan(d_x[2, [1, 3]]).all()


def test_stencils_are_the_textbook_ones() -> None:
    assert difference_offsets(derivative=1, accuracy=1, method="central") == (-1, 0, 1)
    assert difference_coeffs((-1, 0, 1), 1) == (-0.5, 0.0, 0.5)
    assert difference_coeffs((-2, -1, 0, 1, 2), 1) == pytest.approx(
        (1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12)
    )
    assert difference_coeffs((-1, 0, 1), 2) == (1.0, -2.0, 1.0)
    with pytest.raises(ValueError, match="cannot resolve"):
        difference_coeffs((0, 1), 2)


def test_periodic_difference_reads_across_the_seam() -> None:
    lon = np.arange(36) * 10.0
    f = np.sin(np.deg2rad(lon))
    got = difference(f, step=np.deg2rad(10.0), accuracy=3, periodic=True)
    np.testing.assert_allclose(got, np.cos(np.deg2rad(lon)), atol=1e-4)
    assert abs(got[0] - 1.0) < 1e-4  # the seam is a centred stencil too


@pytest.mark.parametrize("accuracy", [1, 2, 3])
def test_adaptive_is_exact_on_a_linear_field_up_to_the_mask(accuracy: int) -> None:
    rows, cols = np.mgrid[0:12, 0:15].astype(float)
    f = 2.0 * rows - 0.5 * cols
    land = (rows - 6) ** 2 + (cols - 7) ** 2 < 6  # an island
    land[:, 12] = True  # and a wall; every valid run is at least two cells
    for backend in ("numpy", "jax") if HAVE_JAX else ("numpy",):
        field = _jax().numpy.asarray(f) if backend == "jax" else f
        d_y, d_x = adaptive_gradient_2d(field, mask=land, accuracy=accuracy)
        for component, want in ((d_y, 2.0), (d_x, -0.5)):
            got = np.asarray(component)
            assert np.isnan(got[land]).all()
            np.testing.assert_allclose(got[~land], want, rtol=1e-9)


def test_propagate_spreads_one_halo_and_adaptive_none() -> None:
    f = np.arange(9.0)
    f[4] = np.nan
    plain = difference(f, accuracy=2)
    assert np.isnan(plain).sum() == 3  # the cell and both neighbours
    recovered = adaptive_difference(f, accuracy=2)
    assert np.isnan(recovered).sum() == 1
    np.testing.assert_allclose(recovered[[3, 5]], 1.0)


def test_adaptive_descends_accuracy_for_a_narrow_channel() -> None:
    """A two-cell channel supports a first-order stencil and nothing wider."""
    f = np.array([np.nan, np.nan, 3.0, 4.0, np.nan, np.nan])
    got = adaptive_difference(f, accuracy=3)
    np.testing.assert_allclose(got[2:4], 1.0)
    assert np.isnan(got[[0, 1, 4, 5]]).all()


@needs_jax
def test_adaptive_jits_and_traces_once() -> None:
    traces = []

    @jax.jit
    def step(f, mask):
        traces.append(1)
        return adaptive_difference(f, mask=mask, accuracy=2)

    f = jax.numpy.arange(10.0)
    step(f, jax.numpy.zeros(10, bool))
    out = step(f, jax.numpy.arange(10) == 4)
    assert len(traces) == 1
    np.testing.assert_allclose(np.asarray(out)[[3, 5]], 1.0)


def test_coastline_measurement() -> None:
    """xrgrad's comparison: sin(2λ)cos(φ) on 121 x 91 with a synthetic coast.

    Propagate loses the coastal strip; adaptive loses none of it and stays
    within a small multiple of the interior error there.
    """
    lam = np.deg2rad(np.linspace(0.0, 120.0, 121))
    phi = np.deg2rad(np.linspace(-45.0, 45.0, 91))
    lam2, phi2 = np.meshgrid(lam, phi)
    f = np.sin(2 * lam2) * np.cos(phi2)
    want = 2 * np.cos(2 * lam2) * np.cos(phi2)
    cols = np.arange(121)[None, :]
    rows = np.arange(91)[:, None]
    land = (cols < 10) | ((cols - 70) ** 2 + (rows - 45) ** 2 < 15**2)
    step = lam[1] - lam[0]
    padded = np.pad(land, ((0, 0), (1, 1)))
    coast = ~land & (padded[:, :-2] | padded[:, 2:])  # water next to land along x
    plain = difference(np.where(land, np.nan, f), step=step, accuracy=2)
    adaptive = adaptive_difference(f, mask=land, step=step, accuracy=2)
    assert np.isnan(plain[coast]).all()
    assert np.isfinite(adaptive[~land]).all()
    scale = np.abs(want[~land]).mean()
    interior = ~land & ~coast
    assert np.abs(adaptive - want)[coast].max() / scale < 1e-2
    assert np.abs(adaptive - want)[interior].max() / scale < 1e-3


@pytest.mark.parametrize(
    ("size", "accuracy", "method"),
    [(3, 2, "forward"), (3, 2, "backward"), (5, 4, "central"), (2, 1, "central")],
)
def test_a_short_axis_raises_instead_of_reading_outside_it(
    size: int, accuracy: int, method: str
) -> None:
    with pytest.raises(ValueError, match="short"):
        difference(np.arange(float(size)), accuracy=accuracy, method=method)


def test_adaptive_descends_on_a_short_axis() -> None:
    got = adaptive_difference(np.arange(3.0), accuracy=4)
    np.testing.assert_allclose(got, 1.0)


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_nonuniform_coeffs_reduce_to_the_uniform_stencil(
    derivative: int, accuracy: int, method: str
) -> None:
    offsets = difference_offsets(
        derivative=derivative, accuracy=accuracy, method=method
    )
    got = nonuniform_coeffs(2.5 * np.asarray(offsets, dtype=float), derivative)
    want = np.asarray(difference_coeffs(offsets, derivative)) / 2.5**derivative
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(("derivative", "accuracy", "method"), CASES)
def test_evenly_spaced_coords_equal_the_step(
    derivative: int, accuracy: int, method: str
) -> None:
    f = np.random.default_rng(accuracy).normal(size=(4, 17))
    coords = -1.5 * np.arange(17.0)
    got = difference(
        f, coords=coords, derivative=derivative, accuracy=accuracy, method=method
    )
    want = difference(
        f, step=-1.5, derivative=derivative, accuracy=accuracy, method=method
    )
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("accuracy", [1, 2, 3])
def test_uneven_levels_are_exact_for_the_stencils_degree(accuracy: int) -> None:
    """A K-point stencil on uneven levels is exact for polynomials of degree K - 1."""
    for derivative in (1, 2):
        degree = derivative + accuracy - 1  # the one-sided stencils' degree
        f = LEVELS**degree
        want = np.polyval(np.polyder(np.eye(degree + 1)[0], derivative), LEVELS)
        got = difference(f, coords=LEVELS, derivative=derivative, accuracy=accuracy)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


@needs_jax
def test_uneven_levels_run_under_jit_with_traced_coords() -> None:
    want = difference(LEVELS**2, coords=LEVELS)
    jitted = jax.jit(lambda f, z: difference(f, coords=z))
    got = jitted(jax.numpy.asarray(LEVELS**2), jax.numpy.asarray(LEVELS))
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12)


def test_adaptive_difference_on_uneven_levels_keeps_the_cells_next_to_a_gap() -> None:
    f = LEVELS.copy()  # f = z: every stencil is exact
    f[4] = np.nan
    got = adaptive_difference(f, coords=LEVELS)
    assert np.isnan(got[4]) and np.isfinite(np.delete(got, 4)).all()
    np.testing.assert_allclose(np.delete(got, 4), 1.0, rtol=1e-12)


def test_coords_must_fit_the_axis_and_not_wrap() -> None:
    with pytest.raises(ValueError, match="entries"):
        difference(np.zeros((3, 9)), coords=LEVELS[:5])
    with pytest.raises(ValueError, match="periodic"):
        difference(np.zeros((3, 9)), coords=LEVELS, periodic=True)


# ported from geotoolz-ops tests/test_calculus.py


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_primitives_keep_the_invariants_up_to_the_edges(backend: str) -> None:
    rows, cols = np.mgrid[0:6, 0:8].astype(float)
    xp = np if backend == "numpy" else _jax().numpy
    # coordinates with steps dy = 0.5, dx = 2
    y, x = xp.asarray(0.5 * rows), xp.asarray(2.0 * cols)
    for fn in (curl_2d, adaptive_curl_2d):
        np.testing.assert_allclose(np.asarray(fn(-y, x, 0.5, 2.0)), 2.0, rtol=1e-12)
    for fn in (divergence_2d, adaptive_divergence_2d):
        np.testing.assert_allclose(np.asarray(fn(-y, x, 0.5, 2.0)), 0.0, atol=1e-12)
    for fn in (laplacian_2d, adaptive_laplacian_2d):
        out = np.asarray(fn(x**2 + y**2, 0.5, 2.0))
        np.testing.assert_allclose(out, 4.0, rtol=1e-12)
    if HAVE_JAX:
        x, y = xp.asarray(cols), xp.asarray(rows)  # index coordinates, scaled by steps
        jitted = jax.jit(lambda f: laplacian_2d(f, 0.5, 2.0))
        want = 2 / 4 + 2 / 0.25
        np.testing.assert_allclose(jitted(jax.numpy.asarray(x**2 + y**2)), want)


@pytest.mark.parametrize(
    "mask_kind",
    [pytest.param("island", id="island"), pytest.param("wall", id="one-column-wall")],
)
def test_adaptive_keeps_the_invariants_next_to_a_mask(mask_kind: str) -> None:
    y, x = np.mgrid[0:9, 0:11].astype(float)
    mask = (y == 4) & (x == 5) if mask_kind == "island" else x == 5
    zeta = adaptive_curl_2d(-y, x, mask=mask)
    div = adaptive_divergence_2d(-y, x, mask=mask)
    lap = adaptive_laplacian_2d(x**2 + y**2, mask=mask)
    for out in (zeta, div, lap):
        np.testing.assert_array_equal(np.isnan(out), mask)  # only the mask is lost
    np.testing.assert_allclose(zeta[~mask], 2.0, rtol=1e-12)
    np.testing.assert_allclose(div[~mask], 0.0, atol=1e-12)
    np.testing.assert_allclose(lap[~mask], 4.0, rtol=1e-12)


def test_propagate_spreads_nan_one_stencil_halo() -> None:
    y, x = np.mgrid[0:9, 0:11].astype(float)
    f = x**2 + y**2
    f[4, 5] = np.nan
    lost = np.isnan(laplacian_2d(f))
    want = np.zeros_like(lost)
    want[3:6, 5] = True  # the 3-point stencils along rows
    want[4, 4:7] = True  # and along columns
    np.testing.assert_array_equal(lost, want)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_3d_primitives_are_exact_on_uneven_levels(backend: str) -> None:
    xp = np if backend == "numpy" else _jax().numpy
    z, y, x = np.mgrid[0:6, 0:5, 0:7].astype(float)
    z = DEPTHS[:, None, None] + 0.0 * z
    z, y, x, levels = (xp.asarray(a) for a in (z, y, x, DEPTHS))
    lap = laplacian_3d(x**2 + y**2 + z**2, z=levels)
    np.testing.assert_allclose(np.asarray(lap), 6.0, rtol=1e-9)
    grad = np.asarray(gradient_3d(3.0 * z + 2.0 * y + x, z=levels))
    for component, want in zip(grad, (3.0, 2.0, 1.0), strict=True):
        np.testing.assert_allclose(component, want, rtol=1e-9)
    omega = np.asarray(curl_3d(z, x, 0.0 * z, z=levels))  # u = z: shear; v = x: spin
    np.testing.assert_allclose(omega[0], 1.0, rtol=1e-9)  # ω_z = ∂v/∂x
    np.testing.assert_allclose(omega[1], 1.0, rtol=1e-9)  # ω_y = ∂u/∂z
    np.testing.assert_allclose(omega[2], 0.0, atol=1e-12)
    div = divergence_3d(x, y, -2.0 * z, z=levels)
    np.testing.assert_allclose(np.asarray(div), 0.0, atol=1e-9)
    if HAVE_JAX:
        jitted = jax.jit(lambda f, zz: laplacian_3d(f, z=zz))
        got = jitted(xp.asarray(x**2 + z**2), levels)
        np.testing.assert_allclose(got, 4.0, rtol=1e-9)


def test_3d_adaptive_primitives_keep_the_invariants_next_to_a_mask() -> None:
    z, y, x = np.mgrid[0:7, 0:7, 0:8].astype(float)
    rock = (z == 3) & (y == 3) & (x == 4)
    lap = adaptive_laplacian_3d(x**2 + y**2 + z**2, mask=rock)
    div = adaptive_divergence_3d(x, y, z, mask=rock)
    omega = adaptive_curl_3d(-y, x, 0.0 * z, mask=rock)
    grad = adaptive_gradient_3d(z + y + x, mask=rock)
    for out in (lap, div, *omega, *grad):
        np.testing.assert_array_equal(np.isnan(out), rock)
    np.testing.assert_allclose(lap[~rock], 6.0, rtol=1e-12)
    np.testing.assert_allclose(div[~rock], 3.0, rtol=1e-12)
    np.testing.assert_allclose(omega[0][~rock], 2.0, rtol=1e-12)


def test_1d_primitives_on_even_and_uneven_axes() -> None:
    s = np.arange(6.0)
    np.testing.assert_allclose(gradient_1d(3.0 * s, 0.5), 6.0)
    np.testing.assert_allclose(divergence_1d(3.0 * s, coords=2.0 * s), 1.5)
    np.testing.assert_allclose(laplacian_1d(DEPTHS**2, coords=DEPTHS), 2.0, rtol=1e-9)
    gappy = DEPTHS.copy()
    gappy[2] = np.nan
    for fn in (adaptive_gradient_1d, adaptive_divergence_1d):
        got = fn(gappy, coords=DEPTHS)
        np.testing.assert_allclose(np.delete(got, 2), 1.0, rtol=1e-9)
    lap = adaptive_laplacian_1d(gappy**2, coords=DEPTHS)
    assert np.isnan(lap[2])
