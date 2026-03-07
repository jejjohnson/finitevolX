"""Tests for spectral_transforms: DCT/DST types I–IV, multi-axis helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.fft as sf

from finitevolx._src.spectral_transforms import (
    dct,
    dctn,
    dst,
    dstn,
    idct,
    idctn,
    idst,
    idstn,
)

jax.config.update("jax_enable_x64", True)

_X4 = np.array([1.0, 2.0, 3.0, 4.0])
_X6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# DCT types I–IV: agreement with scipy
# ---------------------------------------------------------------------------


class TestDCTvsScipyType1:
    def test_1d(self):
        y = np.array(dct(_X4.copy(), type=1))
        np.testing.assert_allclose(y, sf.dct(_X4, type=1), rtol=1e-6)

    def test_1d_longer(self):
        y = np.array(dct(_X6.copy(), type=1))
        np.testing.assert_allclose(y, sf.dct(_X6, type=1), rtol=1e-6)


class TestDCTvsScipyType2:
    def test_1d(self):
        y = np.array(dct(_X4.copy(), type=2))
        np.testing.assert_allclose(y, sf.dct(_X4, type=2), rtol=1e-6)

    def test_matches_jax_scipy(self):
        import jax.scipy.fft as jf

        x = jnp.array(_X6)
        np.testing.assert_allclose(
            np.array(dct(x, type=2)), np.array(jf.dct(x, type=2)), atol=1e-6
        )


class TestDCTvsScipyType3:
    def test_1d(self):
        y = np.array(dct(_X4.copy(), type=3))
        np.testing.assert_allclose(y, sf.dct(_X4, type=3), rtol=1e-6)


class TestDCTvsScipyType4:
    def test_1d(self):
        y = np.array(dct(_X4.copy(), type=4))
        np.testing.assert_allclose(y, sf.dct(_X4, type=4), rtol=1e-6)


# ---------------------------------------------------------------------------
# DST types I–IV: agreement with scipy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_dst_vs_scipy(t):
    y = np.array(dst(jnp.array(_X4), type=t))
    np.testing.assert_allclose(y, sf.dst(_X4, type=t), rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Inverse round-trips for all types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_idct_roundtrip(t):
    x = jnp.array(_X4)
    np.testing.assert_allclose(
        np.array(idct(dct(x, type=t), type=t)), _X4, atol=1e-10
    )


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_idst_roundtrip(t):
    x = jnp.array(_X4)
    np.testing.assert_allclose(
        np.array(idst(dst(x, type=t), type=t)), _X4, atol=1e-10
    )


# ---------------------------------------------------------------------------
# Axis argument
# ---------------------------------------------------------------------------


class TestAxisArg:
    def test_dct2_axis0(self):
        x2d = np.arange(1.0, 13.0).reshape(3, 4)
        jax_r = np.array(dct(jnp.array(x2d), type=2, axis=0))
        sci_r = sf.dct(x2d, type=2, axis=0)
        np.testing.assert_allclose(jax_r, sci_r, rtol=1e-6, atol=1e-10)

    def test_dct2_axis1(self):
        x2d = np.arange(1.0, 13.0).reshape(3, 4)
        jax_r = np.array(dct(jnp.array(x2d), type=2, axis=1))
        sci_r = sf.dct(x2d, type=2, axis=1)
        np.testing.assert_allclose(jax_r, sci_r, rtol=1e-6)

    def test_dst1_negative_axis(self):
        x2d = np.arange(1.0, 13.0).reshape(3, 4)
        jax_r = np.array(dst(jnp.array(x2d), type=1, axis=-1))
        sci_r = sf.dst(x2d, type=1, axis=-1)
        np.testing.assert_allclose(jax_r, sci_r, rtol=1e-6)

    def test_output_shape_preserved(self):
        x = jnp.ones((5, 7, 3))
        assert dct(x, type=2, axis=1).shape == (5, 7, 3)
        assert dst(x, type=1, axis=2).shape == (5, 7, 3)


# ---------------------------------------------------------------------------
# Multi-axis helpers
# ---------------------------------------------------------------------------


class TestMultiAxis:
    @pytest.mark.parametrize("t", [1, 2])
    def test_dctn_vs_scipy(self, t):
        x2d = np.arange(1.0, 17.0).reshape(4, 4)
        jax_r = np.array(dctn(jnp.array(x2d), type=t, axes=[0, 1]))
        sci_r = sf.dctn(x2d, type=t, axes=[0, 1])
        np.testing.assert_allclose(jax_r, sci_r, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("t", [1, 2])
    def test_idctn_roundtrip(self, t):
        x2d = jnp.arange(1.0, 17.0).reshape(4, 4)
        rt = np.array(idctn(dctn(x2d, type=t), type=t))
        np.testing.assert_allclose(rt, np.arange(1.0, 17.0).reshape(4, 4), atol=1e-8)

    def test_dstn_roundtrip(self):
        x2d = jnp.arange(1.0, 17.0).reshape(4, 4)
        rt = np.array(idstn(dstn(x2d, type=1), type=1))
        np.testing.assert_allclose(rt, np.arange(1.0, 17.0).reshape(4, 4), atol=1e-8)

    def test_dctn_none_axes_uses_all(self):
        x = jnp.ones((3, 4))
        y_all = dctn(x, type=2, axes=None)
        y_explicit = dctn(x, type=2, axes=[0, 1])
        np.testing.assert_allclose(np.array(y_all), np.array(y_explicit), atol=1e-10)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_dct_invalid_type():
    with pytest.raises(ValueError, match="DCT type must be"):
        dct(jnp.ones(4), type=5)


def test_dst_invalid_type():
    with pytest.raises(ValueError, match="DST type must be"):
        dst(jnp.ones(4), type=0)
