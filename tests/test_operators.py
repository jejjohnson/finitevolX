import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.constants import GRAVITY
from finitevolx._src.operators.operators import (
    bernoulli_potential,
    kinetic_energy,
)

jax.config.update("jax_enable_x64", True)

# sizes for all of the Arrays
Nx_center, Ny_center = 100, 50
Nx_face_u, Ny_face_u = Nx_center + 1, Ny_center
Nx_face_v, Ny_face_v = Nx_center, Ny_center + 1


@pytest.fixture()
def u_2d_ones():
    return jnp.ones((Nx_face_u, Ny_face_u))


@pytest.fixture()
def v_2d_ones():
    return jnp.ones((Nx_face_v, Ny_face_v))


@pytest.fixture()
def center_2d_ones():
    return jnp.ones((Nx_center, Ny_center))


def test_kinetic_energy_2d_ones(u_2d_ones, v_2d_ones, center_2d_ones):
    u = u_2d_ones
    v = v_2d_ones
    h = center_2d_ones

    # Test that kinetic_energy returns the correct shape
    ke = kinetic_energy(u=u, v=v)

    # For constant velocity u=v=1, ke should be 0.5 * (1² + 1²) = 1.0 everywhere
    expected_ke = 1.0

    # kinetic_energy should return shape [Nx, Ny]
    assert ke.shape == h.shape
    np.testing.assert_array_almost_equal(ke, expected_ke)


def test_kinetic_energy_2d_nonconst():
    """Test kinetic energy with non-constant fields (linear ramps).

    This test catches axis mix-ups (e.g., averaging u along y instead of x)
    and off-by-one indexing bugs that constant-field tests would miss.
    """
    # Create linear ramp in x for u: u[i, j] = i
    # u is on x-faces, shape [Nx+1, Ny] = [101, 50]
    u = jnp.arange(Nx_face_u)[:, None] * jnp.ones((Nx_face_u, Ny_face_u))

    # Create linear ramp in y for v: v[i, j] = j
    # v is on y-faces, shape [Nx, Ny+1] = [100, 51]
    v = jnp.arange(Ny_face_v)[None, :] * jnp.ones((Nx_face_v, Ny_face_v))

    # Compute expected kinetic energy via explicit slice math
    # u²_on_center[i, j] = 0.5 * (u²[i, j] + u²[i+1, j])
    # For u[i, j] = i, this gives: 0.5 * (i² + (i+1)²)
    u2 = u**2
    u2_on_center_expected = 0.5 * (u2[:-1, :] + u2[1:, :])

    # v²_on_center[i, j] = 0.5 * (v²[i, j] + v²[i, j+1])
    # For v[i, j] = j, this gives: 0.5 * (j² + (j+1)²)
    v2 = v**2
    v2_on_center_expected = 0.5 * (v2[:, :-1] + v2[:, 1:])

    # ke = 0.5 * (u²_on_center + v²_on_center)
    expected_ke = 0.5 * (u2_on_center_expected + v2_on_center_expected)

    # Compute actual kinetic energy
    ke = kinetic_energy(u=u, v=v)

    # Verify shape and values
    assert ke.shape == (Nx_center, Ny_center)
    np.testing.assert_allclose(ke, expected_ke)


def test_bernoulli_potential_2d_ones(u_2d_ones, v_2d_ones, center_2d_ones):
    u = u_2d_ones
    v = v_2d_ones
    h = center_2d_ones

    # Test that bernoulli_potential returns the correct shape
    p = bernoulli_potential(h=h, u=u, v=v)

    # For constant u=v=1 and h=1, p should be ke + g*h
    # ke = 1.0 (from test above), h = 1.0, so p = 1.0 + 9.81*1.0 = 10.81
    expected_p = 1.0 + GRAVITY * 1.0

    # bernoulli_potential should return shape [Nx, Ny]
    assert p.shape == h.shape
    np.testing.assert_array_almost_equal(p, expected_p)


def test_bernoulli_potential_2d_nonconst():
    """Test Bernoulli potential with spatially varying u, v, h.

    This test asserts p == ke(u, v) + g*h elementwise, catching any
    centering or indexing mistakes hidden by constant inputs.
    """
    # Create spatially varying fields
    # u: linear ramp in x, shape [Nx+1, Ny] = [101, 50]
    u = jnp.arange(Nx_face_u)[:, None] * jnp.ones((Nx_face_u, Ny_face_u))

    # v: linear ramp in y, shape [Nx, Ny+1] = [100, 51]
    v = jnp.arange(Ny_face_v)[None, :] * jnp.ones((Nx_face_v, Ny_face_v))

    # h: sequential values reshaped to [Nx, Ny] = [100, 50]
    h = jnp.arange(Nx_center * Ny_center).reshape(Nx_center, Ny_center) * 0.1

    # Compute Bernoulli potential
    p = bernoulli_potential(h=h, u=u, v=v)

    # Compute expected: p = ke(u, v) + g*h
    ke = kinetic_energy(u=u, v=v)
    expected_p = ke + GRAVITY * h

    # Verify shape and elementwise equality
    assert p.shape == (Nx_center, Ny_center)
    np.testing.assert_allclose(p, expected_p)
