import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.reconstructions.reconstruct import (
    reconstruct,
    reconstruct_1pt,
    reconstruct_3pt,
    reconstruct_5pt,
)

# Fixed 6x6 all-ones arrays (no old MaskGrid dependency)
Q_ONES = jnp.ones((6, 6))
U_ONES = jnp.ones((6, 6))
V_ONES = jnp.ones((6, 6))

METHODS = ["linear", "weno", "wenoz"]
NUM_PTS = [1, 3, 5]
METHODS_NUMPTS = list(itertools.product(METHODS, NUM_PTS))


def test_reconstruct_1pt_nomask():
    u = U_ONES[1:-1]
    flux = reconstruct_1pt(q=Q_ONES, u=u, dim=0, u_mask=None)
    assert flux.shape == u.shape
    np.testing.assert_array_almost_equal(flux, np.ones_like(flux))


@pytest.mark.parametrize("method", METHODS)
def test_reconstruct_3pt_nomask(method):
    u = U_ONES[1:-1]
    flux = reconstruct_3pt(q=Q_ONES, u=u, u_mask=None, dim=0, method=method)
    assert flux.shape == u.shape
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method", ["linear", "weno", "wenoz"])
def test_reconstruct_5pt_nomask(method):
    u = U_ONES[1:-1]
    flux = reconstruct_5pt(q=Q_ONES, u=u, u_mask=None, dim=0, method=method)
    assert flux.shape == u.shape
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method,num_pts", METHODS_NUMPTS)
def test_reconstruct_nomask_u(method, num_pts):
    import jax
    u = jax.lax.slice_in_dim(U_ONES, axis=0, start_index=1, limit_index=-1)
    flux = reconstruct(q=Q_ONES, u=u, u_mask=None, dim=0, method=method, num_pts=num_pts)
    assert flux.shape == u.shape, f"Shape: {flux.shape} | {u.shape}"
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))


@pytest.mark.parametrize("method,num_pts", METHODS_NUMPTS)
def test_reconstruct_nomask_v(method, num_pts):
    import jax
    v = jax.lax.slice_in_dim(V_ONES, axis=1, start_index=1, limit_index=-1)
    flux = reconstruct(q=Q_ONES, u=v, u_mask=None, dim=1, method=method, num_pts=num_pts)
    assert flux.shape == v.shape, f"Shape: {flux.shape} | {v.shape}"
    np.testing.assert_array_equal(flux, jnp.ones_like(flux))
