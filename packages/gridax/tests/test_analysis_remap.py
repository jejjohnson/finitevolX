"""gridax.analysis remapping: the xrops kernel tests, plus backends and dtypes.

The kernel tests are ported unchanged from xrtoolz-ops
(``tests/interpolate/test_coord_remap.py`` and
``test_coord_remap_columns.py``), where they ran against the NumPy kernels
these replace (#423). The new tests check that JAX inputs give the same
numbers, run under ``jax.jit``, and that the output keeps the input's dtype.
"""

from __future__ import annotations

from gridax.analysis import remap_axis, remap_axis_columns
import numpy as np
import pytest


class _Kernels:
    remap_axis = staticmethod(remap_axis)
    remap_axis_columns = staticmethod(remap_axis_columns)


ia = _Kernels()


def test_array_remap_identity_when_target_equals_source():
    src = np.linspace(0.0, 10.0, 11)
    values = np.sin(src)
    out = ia.remap_axis(values, axis=-1, source_coords=src, target_coords=src)
    np.testing.assert_allclose(out, values)


def test_array_remap_linear_recovers_known_function():
    src = np.linspace(0.0, 1.0, 21)
    values = 2.0 * src + 3.0  # exactly representable by linear interp
    tgt = np.linspace(0.0, 1.0, 7)
    out = ia.remap_axis(values, axis=-1, source_coords=src, target_coords=tgt)
    np.testing.assert_allclose(out, 2.0 * tgt + 3.0, rtol=1e-12)


def test_array_remap_2d_along_first_axis():
    src = np.linspace(0.0, 1.0, 11)
    field = np.outer(src, np.arange(4, dtype=float))  # shape (11, 4)
    tgt = np.array([0.25, 0.5, 0.75])
    out = ia.remap_axis(field, axis=0, source_coords=src, target_coords=tgt)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, np.outer(tgt, np.arange(4, dtype=float)))


def test_array_remap_handles_descending_source():
    src = np.linspace(10.0, 0.0, 11)  # descending
    values = src.copy()  # f(x)=x
    tgt = np.array([2.0, 5.0, 8.0])
    out = ia.remap_axis(values, axis=-1, source_coords=src, target_coords=tgt)
    np.testing.assert_allclose(out, tgt, rtol=1e-12)


def test_array_remap_outside_range_returns_nan():
    src = np.linspace(0.0, 1.0, 11)
    values = np.ones_like(src)
    tgt = np.array([-0.5, 0.5, 1.5])
    out = ia.remap_axis(values, axis=-1, source_coords=src, target_coords=tgt)
    assert np.isnan(out[0])
    assert out[1] == 1.0
    assert np.isnan(out[2])


def test_array_remap_nearest_method():
    src = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    tgt = np.array([0.4, 1.6, 2.9])
    out = ia.remap_axis(
        values, axis=-1, source_coords=src, target_coords=tgt, method="nearest"
    )
    # 0.4 → 0.0; 1.6 → 2.0; 2.9 → 3.0
    np.testing.assert_array_equal(out, np.array([10.0, 30.0, 40.0]))


def test_array_remap_invalid_method_raises():
    src = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        ia.remap_axis(
            np.array([1.0, 2.0]),
            axis=-1,
            source_coords=src,
            target_coords=src,
            method="bogus",
        )


def test_array_remap_non_monotonic_source_raises():
    with pytest.raises(ValueError):
        ia.remap_axis(
            np.array([1.0, 2.0, 3.0]),
            axis=-1,
            source_coords=np.array([0.0, 2.0, 1.0]),
            target_coords=np.array([0.5]),
        )


def test_array_remap_preserves_complex_dtype():
    """Complex inputs must not silently lose their imaginary component."""
    src = np.linspace(0.0, 1.0, 11)
    z = src + 1j * src**2
    tgt = np.array([0.25, 0.5, 0.75])
    out = ia.remap_axis(z, axis=-1, source_coords=src, target_coords=tgt)
    assert np.iscomplexobj(out)
    np.testing.assert_allclose(out.real, tgt, atol=1e-12)
    # Linear interpolation of x^2 incurs O(h^2) discretization error.
    np.testing.assert_allclose(out.imag, tgt**2, atol=5e-3)


# ---------------------------------------------------------------------------
# Kernel — remap_axis_columns
# ---------------------------------------------------------------------------


@pytest.fixture
def field_1d_levels() -> tuple[np.ndarray, np.ndarray]:
    """(values, 1-D source levels) with the remap axis in the middle."""
    rng = np.random.default_rng(0)
    src = np.linspace(0.0, 10.0, 11)
    values = rng.standard_normal((3, src.size, 4))
    return values, src


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_columns_kernel_matches_1d_kernel_exactly(field_1d_levels, method) -> None:
    """Invariant (i): a broadcast 1-D coordinate reproduces the old kernel."""
    values, src = field_1d_levels
    tgt = np.array([-1.0, 0.0, 0.3, 4.999, 5.0, 7.25, 10.0, 11.0, np.nan])
    expected = ia.remap_axis(
        values, axis=1, source_coords=src, target_coords=tgt, method=method
    )
    out = ia.remap_axis_columns(
        values,
        axis=1,
        source_coords=np.broadcast_to(src[None, :, None], values.shape),
        target_coords=tgt,
        method=method,
    )
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_columns_kernel_matches_1d_kernel_descending(field_1d_levels, method) -> None:
    values, src = field_1d_levels
    values, src = values[:, ::-1], src[::-1]
    tgt = np.array([-0.5, 2.0, 5.0, 9.75, 10.5])
    expected = ia.remap_axis(
        values, axis=1, source_coords=src, target_coords=tgt, method=method
    )
    out = ia.remap_axis_columns(
        values,
        axis=1,
        source_coords=np.broadcast_to(src[None, :, None], values.shape),
        target_coords=tgt,
        method=method,
    )
    np.testing.assert_array_equal(out, expected)


def test_columns_kernel_recovers_linear_field_per_column() -> None:
    """Invariant (ii): a field linear in z is exact for any in-range target,
    even when columns have different level spacing (different brackets)."""
    n_lev, n_col = 6, 5
    rng = np.random.default_rng(1)
    # Each column: strictly increasing levels with its own random spacing.
    z = np.cumsum(rng.uniform(0.5, 2.0, size=(n_col, n_lev)), axis=1)
    z -= z[:, :1]  # every column starts at 0 but ends at a different height
    a = rng.standard_normal(n_col)
    b = rng.standard_normal(n_col)
    f = a[:, None] * z + b[:, None]
    tgt = np.array([0.0, 0.7, 1.9, 2.5])  # within every column's range
    out = ia.remap_axis_columns(f, axis=1, source_coords=z, target_coords=tgt)
    np.testing.assert_allclose(out, a[:, None] * tgt[None, :] + b[:, None], rtol=1e-12)
    # Different spacing means different bracketing: the level index bracketing
    # the same target differs between columns.
    hi = (z <= 1.9).sum(axis=1)
    assert len(set(hi.tolist())) > 1


def test_columns_kernel_nearest_extrapolation_holds_end_values() -> None:
    """Invariant (iii): out-of-range targets equal the column's end values."""
    z = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
    f = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    tgt = np.array([0.0, 12.0, 33.0, 100.0])
    out = ia.remap_axis_columns(
        f, axis=1, source_coords=z, target_coords=tgt, extrapolate="nearest"
    )
    # Column 0 range [10, 30]: 0 -> f0, 12 -> interp, 33 -> f[-1], 100 -> f[-1].
    np.testing.assert_allclose(out[0], [1.0, 1.2, 3.0, 3.0])
    # Column 1 range [15, 35]: 0 & 12 -> f0, 33 -> interp, 100 -> f[-1].
    np.testing.assert_allclose(out[1], [-1.0, -1.0, -2.8, -3.0])


def test_columns_kernel_nan_extrapolation_uses_each_columns_own_range() -> None:
    z = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
    f = np.ones_like(z)
    tgt = np.array([12.0, 33.0])
    out = ia.remap_axis_columns(
        f, axis=1, source_coords=z, target_coords=tgt, extrapolate="nan"
    )
    # 12 is inside column 0 but below column 1; 33 the other way round.
    assert out[0, 0] == 1.0 and np.isnan(out[1, 0])
    assert np.isnan(out[0, 1]) and out[1, 1] == 1.0


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_columns_kernel_nearest_method_extrapolation_modes(method) -> None:
    z = np.array([[0.0, 1.0, 2.0]])
    f = np.array([[10.0, 20.0, 30.0]])
    tgt = np.array([-5.0, 0.4, 1.6, 7.0])
    out_nan = ia.remap_axis_columns(
        f, axis=1, source_coords=z, target_coords=tgt, method=method
    )
    out_near = ia.remap_axis_columns(
        f,
        axis=1,
        source_coords=z,
        target_coords=tgt,
        method=method,
        extrapolate="nearest",
    )
    assert np.isnan(out_nan[0, 0]) and np.isnan(out_nan[0, -1])
    assert out_near[0, 0] == 10.0 and out_near[0, -1] == 30.0
    np.testing.assert_array_equal(out_nan[0, 1:3], out_near[0, 1:3])


def test_columns_kernel_descending_and_mixed_columns() -> None:
    """Invariant (iv): a descending column equals its ascending flip, and
    ascending / descending columns may be mixed in one call."""
    rng = np.random.default_rng(2)
    z_asc = np.sort(rng.uniform(0.0, 10.0, size=(4, 7)), axis=1)
    f_asc = rng.standard_normal((4, 7))
    tgt = np.array([-1.0, 1.0, 4.5, 8.0, 12.0])
    ref = ia.remap_axis_columns(
        f_asc, axis=1, source_coords=z_asc, target_coords=tgt, extrapolate="nearest"
    )
    # All descending.
    out_desc = ia.remap_axis_columns(
        f_asc[:, ::-1],
        axis=1,
        source_coords=z_asc[:, ::-1],
        target_coords=tgt,
        extrapolate="nearest",
    )
    np.testing.assert_array_equal(out_desc, ref)
    # Mixed: flip odd columns only.
    z_mix = z_asc.copy()
    f_mix = f_asc.copy()
    z_mix[1::2] = z_asc[1::2, ::-1]
    f_mix[1::2] = f_asc[1::2, ::-1]
    out_mix = ia.remap_axis_columns(
        f_mix, axis=1, source_coords=z_mix, target_coords=tgt, extrapolate="nearest"
    )
    np.testing.assert_array_equal(out_mix, ref)


def test_columns_kernel_non_monotone_column_raises() -> None:
    z = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]])
    f = np.ones_like(z)
    with pytest.raises(ValueError, match="monotone"):
        ia.remap_axis_columns(f, axis=1, source_coords=z, target_coords=np.array([1.0]))


def test_columns_kernel_nan_target_and_bad_args() -> None:
    z = np.array([[0.0, 1.0, 2.0]])
    f = np.array([[1.0, 2.0, 3.0]])
    out = ia.remap_axis_columns(
        f,
        axis=1,
        source_coords=z,
        target_coords=np.array([np.nan, 1.0]),
        extrapolate="nearest",
    )
    assert np.isnan(out[0, 0]) and out[0, 1] == 2.0
    with pytest.raises(ValueError, match="shape"):
        ia.remap_axis_columns(
            f, axis=1, source_coords=z[:, :2], target_coords=np.array([1.0])
        )
    with pytest.raises(ValueError, match="extrapolate"):
        ia.remap_axis_columns(
            f,
            axis=1,
            source_coords=z,
            target_coords=np.array([1.0]),
            extrapolate="bogus",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="method"):
        ia.remap_axis_columns(
            f, axis=1, source_coords=z, target_coords=np.array([1.0]), method="bogus"
        )


def test_columns_kernel_preserves_complex_dtype() -> None:
    z = np.linspace(0.0, 1.0, 11)[None, :]
    f = z + 1j * (2.0 * z)
    tgt = np.array([0.25, 0.5, 2.0])
    out = ia.remap_axis_columns(
        f, axis=1, source_coords=z, target_coords=tgt, extrapolate="nearest"
    )
    assert np.iscomplexobj(out)
    np.testing.assert_allclose(out[0], [0.25 + 0.5j, 0.5 + 1.0j, 1.0 + 2.0j])


# ---------------------------------------------------------------------------
# Layer 0 — remap_axis wrapper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# New: backends and dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64])
def test_output_keeps_the_input_dtype(dtype) -> None:
    """#421: the caller decides the precision; coordinates follow the values."""
    src = np.linspace(0.0, 1.0, 6)
    values = np.ones((2, 6), dtype=dtype)
    tgt = np.array([0.1, 0.5])
    assert remap_axis(values, source_coords=src, target_coords=tgt).dtype == dtype
    z = np.broadcast_to(src, values.shape)
    out = remap_axis_columns(values, axis=-1, source_coords=z, target_coords=tgt)
    assert out.dtype == dtype


def test_integer_values_become_float() -> None:
    src = np.arange(4.0)
    out = remap_axis(np.arange(4), source_coords=src, target_coords=np.array([1.5]))
    np.testing.assert_allclose(out, [1.5])


@pytest.fixture
def jax_x64():
    """JAX with x64 on for one test, restored afterwards."""
    jax = pytest.importorskip("jax")
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_jax_inputs_give_the_numpy_result_and_jit(method, jax_x64) -> None:
    jax = jax_x64
    jnp = jax.numpy
    rng = np.random.default_rng(3)
    src = np.sort(rng.uniform(0.0, 10.0, 9))
    values = rng.standard_normal((3, 9, 4))
    tgt = np.array([-1.0, 0.0, 2.5, 9.9, 11.0, np.nan])
    want = remap_axis(
        values, axis=1, source_coords=src, target_coords=tgt, method=method
    )
    got = remap_axis(
        jnp.asarray(values),
        axis=1,
        source_coords=src,
        target_coords=tgt,
        method=method,
    )
    assert isinstance(got, jax.Array)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, equal_nan=True)
    z = rng.uniform(0.5, 2.0, size=(3, 9, 4)).cumsum(axis=1)
    z[1] = z[1, ::-1]
    want_c = remap_axis_columns(
        values, axis=1, source_coords=z, target_coords=tgt, method=method
    )

    @jax.jit
    def columns(v, zz):
        return remap_axis_columns(
            v, axis=1, source_coords=zz, target_coords=tgt, method=method
        )

    got_c = columns(jnp.asarray(values), jnp.asarray(z))
    np.testing.assert_allclose(np.asarray(got_c), want_c, rtol=1e-12, equal_nan=True)
