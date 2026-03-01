import itertools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.reconstructions.reconstruct import (
    reconstruct,
    reconstruct_1pt,
    reconstruct_3pt,
    reconstruct_5pt,
)

# The legacy reconstruct_* functions require u.shape[dim] == q.shape[dim] - 1.
# U_ONES/V_ONES are sliced accordingly for dim=0 and dim=1.
Q_ONES = jnp.ones((6, 6))
U_ONES = jnp.ones((5, 6))   # shape[0] = q.shape[0] - 1 for dim=0
V_ONES = jnp.ones((6, 5))   # shape[1] = q.shape[1] - 1 for dim=1

METHODS = ["linear", "weno", "wenoz"]
NUM_PTS = [1, 3, 5]
METHODS_NUMPTS = list(itertools.product(METHODS, NUM_PTS))

class _DummyMask(tp.NamedTuple):
    """Minimal duck-typed mask with distbound* attributes for testing."""

    distbound1: jnp.ndarray
    distbound2: jnp.ndarray
    distbound2plus: jnp.ndarray
    distbound3plus: jnp.ndarray


def _make_all_distbound1_mask(shape):
    """All cells are in distbound1 tier: forces 1pt flux everywhere.

    Using only the 1pt tier avoids zero-padded boundary artefacts from the
    higher-order stencils while still exercising the masked code path.
    """
    ones = jnp.ones(shape)
    zeros = jnp.zeros(shape)
    return _DummyMask(
        distbound1=ones,
        distbound2=zeros,
        distbound2plus=zeros,
        distbound3plus=zeros,
    )


# ── reconstruct_1pt ───────────────────────────────────────────────────────────


def test_reconstruct_1pt_nomask():
    flux = reconstruct_1pt(q=Q_ONES, u=U_ONES, dim=0, u_mask=None)
    assert flux.shape == U_ONES.shape
    np.testing.assert_array_almost_equal(flux, np.ones_like(flux))


def test_reconstruct_1pt_mask():
    # distbound1=ones → flux *= 1, so result equals the no-mask flux
    mask = _make_all_distbound1_mask(U_ONES.shape)
    flux = reconstruct_1pt(q=Q_ONES, u=U_ONES, dim=0, u_mask=mask)
    assert flux.shape == U_ONES.shape
    np.testing.assert_array_almost_equal(flux, np.ones_like(flux))


# ── reconstruct_3pt ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_reconstruct_3pt_nomask_dim0(method):
    flux = reconstruct_3pt(q=Q_ONES, u=U_ONES, u_mask=None, dim=0, method=method)
    assert flux.shape == U_ONES.shape
    np.testing.assert_allclose(flux, np.ones_like(np.asarray(flux)), atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_reconstruct_3pt_nomask_dim1(method):
    flux = reconstruct_3pt(q=Q_ONES, u=V_ONES, u_mask=None, dim=1, method=method)
    assert flux.shape == V_ONES.shape
    np.testing.assert_allclose(flux, np.ones_like(np.asarray(flux)), atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_reconstruct_3pt_mask(method):
    # Masked branch: distbound1=0, distbound2plus=1 → flux comes from 3pt term
    mask = _make_all_distbound1_mask(U_ONES.shape)
    flux = reconstruct_3pt(q=Q_ONES, u=U_ONES, u_mask=mask, dim=0, method=method)
    assert flux.shape == U_ONES.shape
    np.testing.assert_allclose(flux, np.ones_like(np.asarray(flux)), atol=1e-5)


# ── reconstruct_5pt ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["linear", "weno", "wenoz"])
def test_reconstruct_5pt_nomask_dim0(method):
    flux = reconstruct_5pt(q=Q_ONES, u=U_ONES, u_mask=None, dim=0, method=method)
    assert flux.shape == U_ONES.shape
    np.testing.assert_allclose(flux, np.ones_like(np.asarray(flux)), atol=1e-5)


@pytest.mark.parametrize("method", ["linear", "weno", "wenoz"])
def test_reconstruct_5pt_nomask_dim1(method):
    flux = reconstruct_5pt(q=Q_ONES, u=V_ONES, u_mask=None, dim=1, method=method)
    assert flux.shape == V_ONES.shape
    np.testing.assert_allclose(flux, np.ones_like(np.asarray(flux)), atol=1e-5)


@pytest.mark.parametrize("method", ["linear", "weno", "wenoz"])
def test_reconstruct_5pt_mask(method):
    # Masked branch: all cells in the distbound3plus tier → uses 5pt stencil
    mask = _make_all_distbound1_mask(U_ONES.shape)
    flux = reconstruct_5pt(q=Q_ONES, u=U_ONES, u_mask=mask, dim=0, method=method)
    assert flux.shape == U_ONES.shape
    np.testing.assert_allclose(flux, np.ones_like(np.asarray(flux)), atol=1e-5)


# ── reconstruct (dispatcher) ──────────────────────────────────────────────────


@pytest.mark.parametrize("method,num_pts", METHODS_NUMPTS)
def test_reconstruct_nomask_u(method, num_pts):
    u = jax.lax.slice_in_dim(jnp.ones((6, 6)), axis=0, start_index=0, limit_index=5)
    flux = reconstruct(q=Q_ONES, u=u, u_mask=None, dim=0, method=method, num_pts=num_pts)
    assert flux.shape == u.shape, f"Shape: {flux.shape} | {u.shape}"
    np.testing.assert_allclose(flux, np.ones_like(np.asarray(flux)), atol=1e-5)


@pytest.mark.parametrize("method,num_pts", METHODS_NUMPTS)
def test_reconstruct_nomask_v(method, num_pts):
    v = jax.lax.slice_in_dim(jnp.ones((6, 6)), axis=1, start_index=0, limit_index=5)
    flux = reconstruct(q=Q_ONES, u=v, u_mask=None, dim=1, method=method, num_pts=num_pts)
    assert flux.shape == v.shape, f"Shape: {flux.shape} | {v.shape}"
    np.testing.assert_allclose(flux, np.ones_like(np.asarray(flux)), atol=1e-5)
