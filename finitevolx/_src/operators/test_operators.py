import pytest
import numpy as np
import jax.numpy as jnp
from finitevolx._src.operators.operators import difference, laplacian
import jax
jax.config.update("jax_enable_x64", True)

rng = np.random.RandomState(123)


@pytest.fixture()
def u_1d_ones():
    return jnp.ones(100)


@pytest.fixture()
def u_2d_ones():
    return jnp.ones((100, 50))


@pytest.fixture()
def u_1d_randn():
    return rng.randn(100)


@pytest.fixture()
def u_2d_randn():
    return rng.randn(100, 50)


def test_difference_1d_order1_ones(u_1d_ones):
    step_size = 0.1

    # classic difference
    du_dx_np = jnp.diff(u_1d_ones, n=1, axis=0) / step_size
    du_dx = difference(u_1d_ones, axis=0, step_size=step_size)

    np.testing.assert_array_equal(du_dx_np, du_dx)


def test_difference_1d_order1_random(u_1d_randn):
    step_size = 0.1

    # classic difference
    du_dx_np = jnp.diff(u_1d_randn, n=1, axis=0) / step_size
    du_dx = difference(u_1d_randn, axis=0, step_size=step_size)

    np.testing.assert_array_equal(du_dx_np, du_dx)


def test_difference_2d_order1_ones(u_2d_ones):
    step_size = 0.1

    # classic difference
    du_dx_np = jnp.diff(u_2d_ones, n=1, axis=0) / step_size
    du_dx = difference(u_2d_ones, axis=0, step_size=step_size)

    np.testing.assert_array_equal(du_dx_np, du_dx)

    # classic difference
    du_dy_np = jnp.diff(u_2d_ones, n=1, axis=1) / step_size
    du_dy = difference(u_2d_ones, axis=1, step_size=step_size)

    np.testing.assert_array_equal(du_dy_np, du_dy)


def test_difference_2d_order1_random(u_2d_randn):
    step_size = 0.1

    # classic difference
    du_dx_np = jnp.diff(u_2d_randn, n=1, axis=0) / step_size
    du_dx = difference(u_2d_randn, axis=0, step_size=step_size)

    np.testing.assert_array_equal(du_dx_np, du_dx)

    du_dy_np = jnp.diff(u_2d_randn, n=1, axis=1) / step_size
    du_dy = difference(u_2d_randn, axis=1, step_size=step_size)

    np.testing.assert_array_equal(du_dy_np, du_dy)


def test_difference_1d_order2_ones(u_1d_ones):
    step_size = 0.1
    derivative = 2

    # classic difference
    du_dx_np = jnp.diff(u_1d_ones, n=2, axis=0) / step_size**derivative
    du_dx = difference(u_1d_ones, axis=0, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dx_np, du_dx)


def test_difference_1d_order2_random(u_1d_randn):
    step_size = 0.1
    derivative = 2

    # classic difference
    du_dx_np = jnp.diff(u_1d_randn, n=2, axis=0) / step_size**derivative
    du_dx = difference(u_1d_randn, axis=0, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dx_np, du_dx)


def test_difference_2d_order2_ones(u_2d_ones):
    step_size = 0.1
    derivative = 2

    # classic difference
    du_dx_np = jnp.diff(u_2d_ones, n=derivative, axis=0) / step_size**derivative
    du_dx = difference(u_2d_ones, axis=0, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dx_np, du_dx)

    # classic difference
    du_dy_np = jnp.diff(u_2d_ones, n=derivative, axis=1) / step_size**derivative
    du_dy = difference(u_2d_ones, axis=1, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dy_np, du_dy)


def test_difference_2d_order2_random(u_2d_randn):
    step_size = 0.1
    derivative = 2
    # classic difference
    du_dx_np = jnp.diff(u_2d_randn, n=derivative, axis=0) / step_size**derivative
    du_dx = difference(u_2d_randn, axis=0, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dx_np, du_dx)

    du_dy_np = jnp.diff(u_2d_randn, n=derivative, axis=1) / step_size**derivative
    du_dy = difference(u_2d_randn, axis=1, step_size=step_size, derivative=derivative)

    np.testing.assert_array_almost_equal(du_dy_np, du_dy)


def test_lap_ones(u_2d_ones):
    step_size = 0.1
    # classic difference
    d2u_dx2 = jnp.diff(u_2d_ones, n=2, axis=0) / step_size**2
    d2u_dy2 = jnp.diff(u_2d_ones, n=2, axis=1) / step_size**2
    lap_u_np = d2u_dx2[:, 1:-1] + d2u_dy2[1:-1, :]

    # wrapper function
    lap_u = laplacian(u_2d_ones, step_size=step_size)

    np.testing.assert_array_almost_equal(lap_u_np, lap_u)


def test_lap_random(u_2d_randn):
    step_size = 0.1

    # classic difference
    d2u_dx2 = jnp.diff(u_2d_randn, n=2, axis=0) / step_size**2
    d2u_dy2 = jnp.diff(u_2d_randn, n=2, axis=1) / step_size**2
    lap_u_np = d2u_dx2[:, 1:-1] + d2u_dy2[1:-1, :]

    # wrapper function
    lap_u = laplacian(u_2d_randn, step_size=step_size)

    np.testing.assert_array_almost_equal(lap_u_np, lap_u)

