"""Tests for QG/SWM convenience utilities (Issue #153).

Covers: coriolis_param, beta_param, coriolis_fn,
        streamfn_to_ssh, ssh_to_streamfn,
        potential_vorticity_multilayer.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.operators.diagnostics import (
    beta_param,
    coriolis_fn,
    coriolis_param,
    potential_vorticity_multilayer,
    qg_potential_vorticity,
    relative_vorticity_cgrid,
    ssh_to_streamfn,
    streamfn_to_ssh,
    stretching_term,
    sw_potential_vorticity,
    sw_potential_vorticity_multilayer,
)
from finitevolx._src.utils.constants import GRAVITY, OMEGA, R_EARTH
from finitevolx._src.vertical.multilayer import multilayer


@pytest.fixture
def grid2d():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


# ======================================================================
# Coriolis / beta-plane constructors
# ======================================================================


class TestCoriolisParam:
    def test_equator(self):
        """f0 = 0 at the equator."""
        result = coriolis_param(0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_north_pole(self):
        """f0 = 2*Omega at the north pole."""
        result = coriolis_param(90.0)
        np.testing.assert_allclose(result, 2.0 * OMEGA, rtol=1e-10)

    def test_45_degrees(self):
        """f0 = 2*Omega*sin(45°) at 45°N."""
        result = coriolis_param(45.0)
        expected = 2.0 * OMEGA * jnp.sin(jnp.deg2rad(45.0))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_array_input_uses_mean(self):
        """Array input uses the mean latitude."""
        lats = jnp.array([40.0, 50.0])
        result = coriolis_param(lats)
        expected = coriolis_param(45.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_custom_omega(self):
        result = coriolis_param(30.0, omega=1.0)
        expected = 2.0 * 1.0 * jnp.sin(jnp.deg2rad(30.0))
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestBetaParam:
    def test_equator(self):
        """beta is maximal at the equator (cos(0) = 1)."""
        result = beta_param(0.0)
        expected = 2.0 * OMEGA / R_EARTH
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_north_pole(self):
        """beta = 0 at the pole (cos(90°) = 0)."""
        result = beta_param(90.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-20)

    def test_45_degrees(self):
        result = beta_param(45.0)
        expected = (2.0 * OMEGA / R_EARTH) * jnp.cos(jnp.deg2rad(45.0))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_array_input_uses_mean(self):
        lats = jnp.array([40.0, 50.0])
        result = beta_param(lats)
        expected = beta_param(45.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestCoriolisFn:
    def test_f_plane(self):
        """With beta=0, f(y) = f0 everywhere."""
        Y = jnp.linspace(0, 1e6, 10)
        result = coriolis_fn(Y, f0=1e-4, beta=0.0)
        np.testing.assert_allclose(result, 1e-4, atol=1e-15)

    def test_beta_plane(self):
        """f(y) = f0 + beta*(y - y0)."""
        f0, beta = 1e-4, 2e-11
        Y = jnp.array([0.0, 1e6, 2e6])
        y0 = 1e6
        result = coriolis_fn(Y, f0=f0, beta=beta, y0=y0)
        expected = f0 + beta * (Y - y0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_y0_defaults_to_mean(self):
        """When y0 is None, uses mean(Y)."""
        f0, beta = 1e-4, 2e-11
        Y = jnp.array([0.0, 1e6, 2e6])
        result = coriolis_fn(Y, f0=f0, beta=beta)
        expected = f0 + beta * (Y - jnp.mean(Y))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_2d_array(self):
        """Works with 2D Y arrays (e.g., meshgrid output)."""
        y1d = jnp.linspace(0, 1e6, 5)
        Y = jnp.broadcast_to(y1d[:, None], (5, 5))
        f0, beta, y0 = 1e-4, 2e-11, 5e5
        result = coriolis_fn(Y, f0=f0, beta=beta, y0=y0)
        assert result.shape == (5, 5)
        expected = f0 + beta * (Y - y0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ======================================================================
# Streamfunction ↔ SSH conversion
# ======================================================================


class TestStreamfnToSsh:
    def test_roundtrip(self):
        """ssh_to_streamfn(streamfn_to_ssh(psi)) == psi."""
        psi = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        f0 = 1e-4
        ssh = streamfn_to_ssh(psi, f0=f0)
        recovered = ssh_to_streamfn(ssh, f0=f0)
        np.testing.assert_allclose(recovered, psi, rtol=1e-10)

    def test_formula_streamfn_to_ssh(self):
        """eta = (f0/g) * psi."""
        psi = jnp.ones((3, 3))
        f0, g = 1e-4, 10.0
        result = streamfn_to_ssh(psi, f0=f0, g=g)
        expected = (f0 / g) * psi
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_formula_ssh_to_streamfn(self):
        """psi = (g/f0) * eta."""
        ssh = jnp.ones((3, 3)) * 0.5
        f0, g = 1e-4, 10.0
        result = ssh_to_streamfn(ssh, f0=f0, g=g)
        expected = (g / f0) * ssh
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_default_gravity(self):
        """Default g uses GRAVITY constant."""
        psi = jnp.ones((2, 2))
        f0 = 1e-4
        result = streamfn_to_ssh(psi, f0=f0)
        expected = (f0 / GRAVITY) * psi
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_shape_preserved(self):
        psi = jnp.ones((5, 10))
        result = streamfn_to_ssh(psi, f0=1e-4)
        assert result.shape == psi.shape


# ======================================================================
# Multilayer QG potential vorticity
# ======================================================================


class TestPotentialVorticityMultilayer:
    def test_output_shape(self, grid2d):
        nl = 3
        psi = jnp.zeros((nl, grid2d.Ny, grid2d.Nx))
        A = 0.01 * jnp.eye(nl)
        y = jnp.broadcast_to(
            jnp.arange(grid2d.Ny, dtype=float)[:, None] * grid2d.dy,
            (grid2d.Ny, grid2d.Nx),
        )
        result = potential_vorticity_multilayer(
            psi, A, f0=1e-4, beta=1e-11, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0
        )
        assert result.shape == (nl, grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_zero(self, grid2d):
        nl = 2
        psi = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        A = 0.01 * jnp.eye(nl)
        y = jnp.zeros((grid2d.Ny, grid2d.Nx))
        result = potential_vorticity_multilayer(
            psi, A, f0=1.0, beta=0.0, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0
        )
        for layer in range(nl):
            np.testing.assert_allclose(result[layer, 0, :], 0.0, atol=1e-10)
            np.testing.assert_allclose(result[layer, -1, :], 0.0, atol=1e-10)
            np.testing.assert_allclose(result[layer, :, 0], 0.0, atol=1e-10)
            np.testing.assert_allclose(result[layer, :, -1], 0.0, atol=1e-10)

    def test_matches_manual_composition(self, grid2d):
        """Must match multilayer(qg_pv)(psi) - stretching_term(A, psi)."""
        nl = 2
        f0, beta, y0 = 1.0, 1e-11, 0.5
        y = jnp.broadcast_to(
            jnp.arange(grid2d.Ny, dtype=float)[:, None] * grid2d.dy,
            (grid2d.Ny, grid2d.Nx),
        )
        key = jax.random.PRNGKey(42)
        psi = jax.random.normal(key, (nl, grid2d.Ny, grid2d.Nx))
        A = jnp.array([[0.02, -0.01], [-0.01, 0.02]])

        result = potential_vorticity_multilayer(
            psi, A, f0=f0, beta=beta, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=y0
        )

        # Manual composition
        qg_pv = multilayer(
            lambda p: qg_potential_vorticity(p, f0, beta, grid2d.dx, grid2d.dy, y, y0)
        )
        expected = qg_pv(psi) - stretching_term(A, psi)

        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_single_layer(self, grid2d):
        """Single-layer case: A is 1x1, should match qg_pv - A*psi."""
        f0, beta, y0 = 1.0, 0.0, 0.0
        y = jnp.zeros((grid2d.Ny, grid2d.Nx))
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y_arr = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        X, Y = jnp.meshgrid(x, y_arr)
        psi = 0.5 * (X**2 + Y**2)
        psi = psi[None, :, :]  # [1, Ny, Nx]
        a_val = 0.05
        A = jnp.array([[a_val]])
        result = potential_vorticity_multilayer(
            psi, A, f0=f0, beta=beta, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=y0
        )
        # laplacian of 0.5*(x^2+y^2) = 2.0, so qg_pv interior = 2.0/f0 = 2.0
        # stretching interior = a_val * psi
        expected_qg = 2.0
        expected_stretch = a_val * psi[0, 1:-1, 1:-1]
        np.testing.assert_allclose(
            result[0, 1:-1, 1:-1], expected_qg - expected_stretch, rtol=1e-5
        )

    def test_zero_coupling(self, grid2d):
        """With A=0, result equals vmapped qg_pv."""
        nl = 2
        f0, beta, y0 = 1.0, 2e-11, 0.0
        y = jnp.broadcast_to(
            jnp.arange(grid2d.Ny, dtype=float)[:, None] * grid2d.dy,
            (grid2d.Ny, grid2d.Nx),
        )
        psi = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        A = jnp.zeros((nl, nl))
        result = potential_vorticity_multilayer(
            psi, A, f0=f0, beta=beta, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=y0
        )
        qg_pv = multilayer(
            lambda p: qg_potential_vorticity(p, f0, beta, grid2d.dx, grid2d.dy, y, y0)
        )
        expected = qg_pv(psi)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ======================================================================
# Shallow-water potential vorticity
# ======================================================================


class TestSWPotentialVorticity:
    def test_output_shape(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        h = 10.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = sw_potential_vorticity(u, v, h, f, grid2d.dx, grid2d.dy)
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_zero(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        h = 10.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = sw_potential_vorticity(u, v, h, f, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-15)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-15)
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(result[:, -1], 0.0, atol=1e-15)

    def test_irrotational_uniform_h(self, grid2d):
        """Uniform u,v (no vorticity) + uniform f,h => PV = f/h at interior X-points."""
        u = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        h = 10.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = sw_potential_vorticity(u, v, h, f, grid2d.dx, grid2d.dy)
        expected = 1e-4 / 10.0
        np.testing.assert_allclose(result[1:-1, 1:-1], expected, rtol=1e-10)

    def test_matches_manual_composition(self, grid2d):
        """Must match relative_vorticity + interp + pointwise PV."""
        c = 1.5
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-c * y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        h = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))

        result = sw_potential_vorticity(u, v, h, f, grid2d.dx, grid2d.dy)

        # Vorticity of solid body rotation = 2c at interior X-points
        omega = relative_vorticity_cgrid(u, v, grid2d.dx, grid2d.dy)
        # f and h are uniform so interpolation doesn't change values
        expected_pv = (omega[1:-1, 1:-1] + 1e-4) / 5.0
        np.testing.assert_allclose(result[1:-1, 1:-1], expected_pv, rtol=1e-10)


class TestSWPotentialVorticityMultilayer:
    def test_output_shape(self, grid2d):
        nl = 3
        u = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        v = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        h = 10.0 * jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = sw_potential_vorticity_multilayer(u, v, h, f, grid2d.dx, grid2d.dy)
        assert result.shape == (nl, grid2d.Ny, grid2d.Nx)

    def test_matches_single_layer(self, grid2d):
        """Each layer should match sw_potential_vorticity called independently."""
        nl = 2
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        u = jax.random.normal(k1, (nl, grid2d.Ny, grid2d.Nx))
        v = jax.random.normal(k2, (nl, grid2d.Ny, grid2d.Nx))
        h = 5.0 + jax.random.normal(k3, (nl, grid2d.Ny, grid2d.Nx)) * 0.1
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))

        result = sw_potential_vorticity_multilayer(u, v, h, f, grid2d.dx, grid2d.dy)

        for k in range(nl):
            expected = sw_potential_vorticity(u[k], v[k], h[k], f, grid2d.dx, grid2d.dy)
            np.testing.assert_allclose(result[k], expected, atol=1e-12)

    def test_ghost_ring_zero(self, grid2d):
        nl = 2
        u = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        v = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        h = 10.0 * jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = sw_potential_vorticity_multilayer(u, v, h, f, grid2d.dx, grid2d.dy)
        for k in range(nl):
            np.testing.assert_allclose(result[k, 0, :], 0.0, atol=1e-15)
            np.testing.assert_allclose(result[k, -1, :], 0.0, atol=1e-15)
            np.testing.assert_allclose(result[k, :, 0], 0.0, atol=1e-15)
            np.testing.assert_allclose(result[k, :, -1], 0.0, atol=1e-15)


# ======================================================================
# JIT compatibility
# ======================================================================


class TestJITCompatibility:
    def test_coriolis_fn_jit(self):
        Y = jnp.linspace(0, 1e6, 10)
        result = coriolis_fn(Y, f0=1e-4, beta=2e-11, y0=5e5)
        result_jit = jax.jit(lambda y: coriolis_fn(y, f0=1e-4, beta=2e-11, y0=5e5))(Y)
        np.testing.assert_allclose(result_jit, result, atol=1e-15)

    def test_streamfn_to_ssh_jit(self):
        psi = jnp.ones((5, 5))
        fn = lambda p: streamfn_to_ssh(p, f0=1e-4)
        np.testing.assert_allclose(jax.jit(fn)(psi), fn(psi), atol=1e-15)

    def test_ssh_to_streamfn_jit(self):
        ssh = jnp.ones((5, 5)) * 0.1
        fn = lambda s: ssh_to_streamfn(s, f0=1e-4)
        np.testing.assert_allclose(jax.jit(fn)(ssh), fn(ssh), atol=1e-15)

    def test_potential_vorticity_multilayer_jit(self, grid2d):
        nl = 2
        psi = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        A = 0.01 * jnp.eye(nl)
        y = jnp.zeros((grid2d.Ny, grid2d.Nx))
        fn = lambda p: potential_vorticity_multilayer(
            p, A, f0=1.0, beta=0.0, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0
        )
        result = fn(psi)
        result_jit = jax.jit(fn)(psi)
        np.testing.assert_allclose(result_jit, result, atol=1e-12)

    def test_sw_potential_vorticity_jit(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        h = 10.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))
        fn = lambda u, v: sw_potential_vorticity(u, v, h, f, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(jax.jit(fn)(u, v), fn(u, v), atol=1e-15)

    def test_sw_potential_vorticity_multilayer_jit(self, grid2d):
        nl = 2
        u = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        v = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        h = 10.0 * jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        f = 1e-4 * jnp.ones((grid2d.Ny, grid2d.Nx))
        fn = lambda u, v: sw_potential_vorticity_multilayer(
            u, v, h, f, grid2d.dx, grid2d.dy
        )
        np.testing.assert_allclose(jax.jit(fn)(u, v), fn(u, v), atol=1e-15)
