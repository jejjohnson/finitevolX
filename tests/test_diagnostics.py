"""Tests for diagnostic operators: strain, enstrophy, energy, Okubo-Weiss, QG PV, vertical velocity."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.grid import ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.operators.diagnostics import (
    available_potential_energy,
    bernoulli_potential,
    enstrophy,
    kinetic_energy,
    okubo_weiss,
    potential_enstrophy,
    potential_vorticity,
    qg_potential_vorticity,
    relative_vorticity_cgrid,
    shear_strain,
    strain_magnitude_squared,
    stretching_term,
    tensor_strain,
    total_energy,
    total_enstrophy,
    vertical_velocity,
)
from finitevolx._src.operators.jacobian import arakawa_jacobian
from finitevolx._src.vertical.multilayer import multilayer


@pytest.fixture
def grid2d():
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return ArakawaCGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


# ======================================================================
# Kinetic energy (existing)
# ======================================================================


class TestKineticEnergy:
    def test_output_shape(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert kinetic_energy(u, v).shape == (grid2d.Ny, grid2d.Nx)

    def test_zero_velocity(self, grid2d):
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        result = kinetic_energy(u, v)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_ghost_ring_zero(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = kinetic_energy(u, v)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[:, -1], 0.0, atol=1e-10)

    def test_uniform_velocity(self, grid2d):
        c = 3.0
        u = c * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = c * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = kinetic_energy(u, v)
        # KE = 0.5 * (u^2 + v^2) = 0.5 * (9 + 9) = 9
        np.testing.assert_allclose(result[1:-1, 1:-1], c**2, rtol=1e-5)


# ======================================================================
# Bernoulli potential (existing)
# ======================================================================


class TestBernoulliPotential:
    def test_output_shape(self, grid2d):
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        assert bernoulli_potential(h, u, v).shape == (grid2d.Ny, grid2d.Nx)

    def test_zero_velocity(self, grid2d):
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        g = 10.0
        result = bernoulli_potential(h, u, v, gravity=g)
        np.testing.assert_allclose(result[1:-1, 1:-1], g * 2.0, rtol=1e-5)


# ======================================================================
# Strain operators (Issue #2)
# ======================================================================


class TestShearStrain:
    def test_output_shape(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = shear_strain(u, v, grid2d.dx, grid2d.dy)
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_uniform_field_zero(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = shear_strain(u, v, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_fields(self, grid2d):
        """u = a*y, v = b*x => Ss = b + a at X-points."""
        a, b = 2.0, 3.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(a * y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(b * x, (grid2d.Ny, grid2d.Nx))
        result = shear_strain(u, v, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], a + b, rtol=1e-5)

    def test_ghost_ring_zero(self, grid2d):
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))
        result = shear_strain(u, v, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[:, -1], 0.0, atol=1e-10)


class TestTensorStrain:
    def test_output_shape(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = tensor_strain(u, v, grid2d.dx, grid2d.dy)
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_uniform_field_zero(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = tensor_strain(u, v, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_fields(self, grid2d):
        """u = a*x, v = b*y => Sn = a - b at T-points."""
        a, b = 5.0, 2.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(a * x, (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(b * y[:, None], (grid2d.Ny, grid2d.Nx))
        result = tensor_strain(u, v, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], a - b, rtol=1e-5)


class TestStrainMagnitudeSquared:
    def test_zero_strain(self):
        sn = jnp.zeros((10, 10))
        ss = jnp.zeros((10, 10))
        result = strain_magnitude_squared(sn, ss)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_known_values(self):
        sn = 3.0 * jnp.ones((10, 10))
        ss = 4.0 * jnp.ones((10, 10))
        result = strain_magnitude_squared(sn, ss)
        np.testing.assert_allclose(result, 25.0, rtol=1e-5)


class TestOkuboWeiss:
    def test_strain_dominated(self):
        sn = 3.0 * jnp.ones((10, 10))
        ss = 4.0 * jnp.ones((10, 10))
        omega = 1.0 * jnp.ones((10, 10))
        result = okubo_weiss(sn, ss, omega)
        # OW = 9 + 16 - 1 = 24 > 0 (strain dominated)
        np.testing.assert_allclose(result, 24.0, rtol=1e-5)

    def test_vorticity_dominated(self):
        sn = 1.0 * jnp.ones((10, 10))
        ss = 1.0 * jnp.ones((10, 10))
        omega = 5.0 * jnp.ones((10, 10))
        result = okubo_weiss(sn, ss, omega)
        # OW = 1 + 1 - 25 = -23 < 0 (vorticity dominated)
        np.testing.assert_allclose(result, -23.0, rtol=1e-5)

    def test_zero_all(self):
        z = jnp.zeros((10, 10))
        result = okubo_weiss(z, z, z)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ======================================================================
# Enstrophy (Issues #2, #73)
# ======================================================================


class TestEnstrophy:
    def test_zero_vorticity(self):
        omega = jnp.zeros((10, 10))
        result = enstrophy(omega)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_known_value(self):
        omega = 4.0 * jnp.ones((10, 10))
        result = enstrophy(omega)
        # 0.5 * 16 = 8
        np.testing.assert_allclose(result, 8.0, rtol=1e-5)

    def test_negative_vorticity(self):
        omega = -3.0 * jnp.ones((10, 10))
        result = enstrophy(omega)
        np.testing.assert_allclose(result, 4.5, rtol=1e-5)


class TestPotentialEnstrophy:
    def test_known_value(self):
        q = 2.0 * jnp.ones((10, 10))
        h = 3.0 * jnp.ones((10, 10))
        result = potential_enstrophy(q, h)
        # 0.5 * 4 * 3 = 6
        np.testing.assert_allclose(result, 6.0, rtol=1e-5)

    def test_zero_q(self):
        q = jnp.zeros((10, 10))
        h = jnp.ones((10, 10))
        result = potential_enstrophy(q, h)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ======================================================================
# Available potential energy (Issue #73)
# ======================================================================


class TestAvailablePotentialEnergy:
    def test_zero_perturbation(self):
        h = 5.0 * jnp.ones((10, 10))
        H = 5.0 * jnp.ones((10, 10))
        result = available_potential_energy(h, H, g_prime=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_known_value(self):
        h = 6.0 * jnp.ones((10, 10))
        H = 4.0 * jnp.ones((10, 10))
        g_prime = 2.0
        result = available_potential_energy(h, H, g_prime)
        # 0.5 * 2 * (6-4)^2 = 0.5 * 2 * 4 = 4
        np.testing.assert_allclose(result, 4.0, rtol=1e-5)


# ======================================================================
# QG potential vorticity (Issue #73)
# ======================================================================


class TestQGPotentialVorticity:
    def test_output_shape(self, grid2d):
        psi = jnp.zeros((grid2d.Ny, grid2d.Nx))
        y = jnp.broadcast_to(
            jnp.arange(grid2d.Ny, dtype=float)[:, None] * grid2d.dy,
            (grid2d.Ny, grid2d.Nx),
        )
        result = qg_potential_vorticity(psi, f0=1e-4, beta=1e-11, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0)
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_zero(self, grid2d):
        psi = jnp.ones((grid2d.Ny, grid2d.Nx))
        y = jnp.broadcast_to(
            jnp.arange(grid2d.Ny, dtype=float)[:, None] * grid2d.dy,
            (grid2d.Ny, grid2d.Nx),
        )
        result = qg_potential_vorticity(psi, f0=1e-4, beta=0.0, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)

    def test_beta_plane_only(self, grid2d):
        """Zero psi => q = beta*(y - y0)/f0."""
        psi = jnp.zeros((grid2d.Ny, grid2d.Nx))
        f0, beta, y0 = 1e-4, 2e-11, 0.5
        y = jnp.broadcast_to(
            jnp.arange(grid2d.Ny, dtype=float)[:, None] * grid2d.dy,
            (grid2d.Ny, grid2d.Nx),
        )
        result = qg_potential_vorticity(psi, f0=f0, beta=beta, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=y0)
        expected = beta * (y[1:-1, 1:-1] - y0) / f0
        np.testing.assert_allclose(result[1:-1, 1:-1], expected, rtol=1e-5)

    def test_quadratic_psi(self, grid2d):
        """psi = 0.5*c*(x^2 + y^2) => laplacian = 2*c, q = 2*c/f0 + beta*(y-y0)/f0."""
        c = 1.0
        f0, beta, y0 = 1.0, 0.0, 0.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y_arr = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        X, Y = jnp.meshgrid(x, y_arr)
        psi = 0.5 * c * (X**2 + Y**2)
        result = qg_potential_vorticity(psi, f0=f0, beta=beta, dx=grid2d.dx, dy=grid2d.dy, y=Y, y0=y0)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c / f0, rtol=1e-5)

    def test_multilayer_output_shape(self, grid2d):
        """Multi-layer psi [nl, Ny, Nx] via multilayer() returns [nl, Ny, Nx]."""
        nl = 3
        psi = jnp.zeros((nl, grid2d.Ny, grid2d.Nx))
        y = jnp.broadcast_to(
            jnp.arange(grid2d.Ny, dtype=float)[:, None] * grid2d.dy,
            (grid2d.Ny, grid2d.Nx),
        )
        qg_pv = multilayer(
            lambda p: qg_potential_vorticity(p, f0=1e-4, beta=1e-11, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0)
        )
        A = jnp.eye(nl) * 0.01
        result = qg_pv(psi) - stretching_term(A, psi)
        assert result.shape == (nl, grid2d.Ny, grid2d.Nx)

    def test_multilayer_stretching_term(self, grid2d):
        """With zero beta and uniform psi, q = -A.psi interior values."""
        nl = 2
        f0 = 1.0
        # Uniform psi = c across both layers
        c = 5.0
        psi = c * jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        y = jnp.zeros((grid2d.Ny, grid2d.Nx))
        # A is diagonal: A @ psi gives a * psi per layer
        a_val = 0.1
        A = a_val * jnp.eye(nl)
        qg_pv = multilayer(
            lambda p: qg_potential_vorticity(p, f0=f0, beta=0.0, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0)
        )
        result = qg_pv(psi) - stretching_term(A, psi)
        # Laplacian of constant = 0, beta term = 0, so q = 0 - a_val*c
        expected = -a_val * c
        np.testing.assert_allclose(result[:, 1:-1, 1:-1], expected, rtol=1e-5)

    def test_multilayer_ghost_ring_zero(self, grid2d):
        """Ghost ring must be zero for all layers."""
        nl = 2
        psi = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        y = jnp.zeros((grid2d.Ny, grid2d.Nx))
        A = 0.01 * jnp.eye(nl)
        qg_pv = multilayer(
            lambda p: qg_potential_vorticity(p, f0=1.0, beta=0.0, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0)
        )
        result = qg_pv(psi) - stretching_term(A, psi)
        for layer in range(nl):
            np.testing.assert_allclose(result[layer, 0, :], 0.0, atol=1e-10)
            np.testing.assert_allclose(result[layer, -1, :], 0.0, atol=1e-10)
            np.testing.assert_allclose(result[layer, :, 0], 0.0, atol=1e-10)
            np.testing.assert_allclose(result[layer, :, -1], 0.0, atol=1e-10)

    def test_multilayer_laplacian_per_layer(self, grid2d):
        """Each layer gets its own Laplacian via multilayer(); no stretching."""
        f0, beta, y0 = 1.0, 0.0, 0.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y_arr = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        X, Y = jnp.meshgrid(x, y_arr)
        # Layer 0: psi = x^2 => lap = 2; Layer 1: psi = y^2 => lap = 2
        psi_0 = 0.5 * X**2
        psi_1 = 0.5 * Y**2
        psi = jnp.stack([psi_0, psi_1], axis=0)
        qg_pv = multilayer(
            lambda p: qg_potential_vorticity(p, f0=f0, beta=beta, dx=grid2d.dx, dy=grid2d.dy, y=Y, y0=y0)
        )
        result = qg_pv(psi)
        # Both layers: laplacian = 1.0, q = 1.0/f0 = 1.0
        np.testing.assert_allclose(result[0, 1:-1, 1:-1], 1.0, rtol=1e-5)
        np.testing.assert_allclose(result[1, 1:-1, 1:-1], 1.0, rtol=1e-5)


# ======================================================================
# Domain-integrated diagnostics (Issue #73)
# ======================================================================


class TestTotalEnergy:
    def test_known_value(self, grid2d):
        ke = jnp.ones((grid2d.Ny, grid2d.Nx))
        ape = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = total_energy(ke, ape, grid2d.dx, grid2d.dy)
        # Interior is 8x8, each cell = 2 * dx * dy = 2 * 0.125 * 0.125
        n_interior = 8 * 8
        expected = 2.0 * n_interior * grid2d.dx * grid2d.dy
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_zero_fields(self, grid2d):
        z = jnp.zeros((grid2d.Ny, grid2d.Nx))
        result = total_energy(z, z, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


class TestTotalEnstrophy:
    def test_known_value(self, grid2d):
        ens = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = total_enstrophy(ens, grid2d.dx, grid2d.dy)
        n_interior = 8 * 8
        expected = 3.0 * n_interior * grid2d.dx * grid2d.dy
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ======================================================================
# Vertical velocity (Issue #82)
# ======================================================================


class TestVerticalVelocity:
    def test_output_shape(self, grid3d):
        u = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        w = vertical_velocity(u, v, grid3d.dx, grid3d.dy, grid3d.dz)
        assert w.shape == (grid3d.Nz + 1, grid3d.Ny, grid3d.Nx)

    def test_zero_divergence(self, grid3d):
        """Non-divergent flow => w = 0 everywhere."""
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        w = vertical_velocity(u, v, grid3d.dx, grid3d.dy, grid3d.dz)
        np.testing.assert_allclose(w, 0.0, atol=1e-10)

    def test_bottom_boundary_zero(self, grid3d):
        """Bottom boundary w[0] should always be zero."""
        # Use a divergent field
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        u = jnp.broadcast_to(x, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        w = vertical_velocity(u, v, grid3d.dx, grid3d.dy, grid3d.dz)
        np.testing.assert_allclose(w[0], 0.0, atol=1e-10)

    def test_known_w_uniform_divergence(self, grid3d):
        """Uniform du/dx = c at every level => w[k] = -c * k * dz."""
        # u = c * x on U-points => du/dx = c at T-points
        c = 2.0
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        u = jnp.broadcast_to(c * x, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        w = vertical_velocity(u, v, grid3d.dx, grid3d.dy, grid3d.dz)
        # At interior T-point: du/dx = c, div = c
        # w[k+1] = -cumsum(div[0..k]) * dz = -(k+1)*c*dz
        for k in range(grid3d.Nz):
            expected_w = -(k + 1) * c * grid3d.dz
            np.testing.assert_allclose(
                w[k + 1, 1:-1, 1:-1], expected_w, rtol=1e-5,
                err_msg=f"w mismatch at level {k+1}",
            )

    def test_mask_zeroes_divergence(self, grid3d):
        """Masking out all cells => w = 0."""
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        u = jnp.broadcast_to(x, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        mask = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        w = vertical_velocity(u, v, grid3d.dx, grid3d.dy, grid3d.dz, mask=mask)
        np.testing.assert_allclose(w, 0.0, atol=1e-10)


# ======================================================================
# Standalone vorticity functions (Issue #73)
# ======================================================================


class TestRelativeVorticityCgrid:
    def test_output_shape(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = relative_vorticity_cgrid(u, v, grid2d.dx, grid2d.dy)
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_irrotational_field(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = relative_vorticity_cgrid(u, v, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_solid_body_rotation(self, grid2d):
        """u = -c*y, v = c*x => zeta = 2c."""
        c = 1.5
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-c * y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = relative_vorticity_cgrid(u, v, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c, rtol=1e-5)

    def test_ghost_ring_zero(self, grid2d):
        c = 1.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-c * y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = relative_vorticity_cgrid(u, v, grid2d.dx, grid2d.dy)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)


class TestPotentialVorticity:
    def test_known_value(self):
        omega = 2.0 * jnp.ones((10, 10))
        f = 1.0 * jnp.ones((10, 10))
        h = 3.0 * jnp.ones((10, 10))
        result = potential_vorticity(omega, f, h)
        # (2 + 1) / 3 = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_zero_h_gives_nan(self):
        omega = jnp.ones((5, 5))
        f = jnp.ones((5, 5))
        h = jnp.zeros((5, 5))
        result = potential_vorticity(omega, f, h)
        assert jnp.all(jnp.isnan(result))

    def test_zero_vorticity(self):
        omega = jnp.zeros((5, 5))
        f = 2.0 * jnp.ones((5, 5))
        h = 4.0 * jnp.ones((5, 5))
        result = potential_vorticity(omega, f, h)
        np.testing.assert_allclose(result, 0.5, rtol=1e-5)


# ======================================================================
# Mask support tests
# ======================================================================


class TestMaskSupport:
    def test_kinetic_energy_mask(self, grid2d):
        u = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        mask = jnp.zeros((grid2d.Ny, grid2d.Nx))
        result = kinetic_energy(u, v, mask=mask)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_kinetic_energy_mask_partial(self, grid2d):
        u = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        mask = jnp.ones((grid2d.Ny, grid2d.Nx))
        result_nomask = kinetic_energy(u, v)
        result_mask = kinetic_energy(u, v, mask=mask)
        np.testing.assert_allclose(result_mask, result_nomask, atol=1e-10)

    def test_enstrophy_mask(self):
        omega = 4.0 * jnp.ones((10, 10))
        mask = jnp.zeros((10, 10))
        result = enstrophy(omega, mask=mask)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_potential_enstrophy_mask(self):
        q = 2.0 * jnp.ones((10, 10))
        h = 3.0 * jnp.ones((10, 10))
        mask = jnp.zeros((10, 10))
        result = potential_enstrophy(q, h, mask=mask)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


# ======================================================================
# JIT compatibility tests
# ======================================================================


class TestJITCompatibility:
    def test_kinetic_energy_jit(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result_eager = kinetic_energy(u, v)
        result_jit = jax.jit(kinetic_energy)(u, v)
        np.testing.assert_allclose(result_jit, result_eager, atol=1e-10)

    def test_relative_vorticity_jit(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        fn = lambda u, v: relative_vorticity_cgrid(u, v, grid2d.dx, grid2d.dy)
        result_eager = fn(u, v)
        result_jit = jax.jit(fn)(u, v)
        np.testing.assert_allclose(result_jit, result_eager, atol=1e-10)

    def test_enstrophy_jit(self):
        omega = jnp.ones((10, 10))
        result_eager = enstrophy(omega)
        result_jit = jax.jit(enstrophy)(omega)
        np.testing.assert_allclose(result_jit, result_eager, atol=1e-10)

    def test_qg_pv_jit(self, grid2d):
        psi = jnp.ones((grid2d.Ny, grid2d.Nx))
        y = jnp.zeros((grid2d.Ny, grid2d.Nx))
        fn = lambda p: qg_potential_vorticity(p, f0=1.0, beta=0.0, dx=grid2d.dx, dy=grid2d.dy, y=y, y0=0.0)
        result_eager = fn(psi)
        result_jit = jax.jit(fn)(psi)
        np.testing.assert_allclose(result_jit, result_eager, atol=1e-10)

    def test_vertical_velocity_jit(self, grid3d):
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        fn = lambda u, v: vertical_velocity(u, v, grid3d.dx, grid3d.dy, grid3d.dz)
        result_eager = fn(u, v)
        result_jit = jax.jit(fn)(u, v)
        np.testing.assert_allclose(result_jit, result_eager, atol=1e-10)


# ======================================================================
# Conservation tests (Issue #73)
# ======================================================================


class TestConservation:
    def test_enstrophy_conservation_arakawa_jacobian(self):
        """Arakawa Jacobian conserves enstrophy: sum(q * J(psi, q)) = 0.

        The Arakawa Jacobian has the property that the sum of
        q * J(psi, q) over the domain vanishes (enstrophy conservation).
        Uses float64 to verify the property at near-machine-epsilon precision.
        """
        N = 16
        dx = dy = 1.0 / N
        Ny = Nx = N + 2  # including ghost ring
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        # Random smooth fields with zero ghost ring (float64 for precision)
        psi = jnp.zeros((Ny, Nx), dtype=jnp.float64)
        q = jnp.zeros((Ny, Nx), dtype=jnp.float64)
        psi = psi.at[1:-1, 1:-1].set(jax.random.normal(k1, (N, N), dtype=jnp.float64))
        q = q.at[1:-1, 1:-1].set(jax.random.normal(k2, (N, N), dtype=jnp.float64))
        # Arakawa Jacobian: output shape is (Ny-2, Nx-2) = (N, N)
        J = arakawa_jacobian(psi, q, dx, dy)
        # Enstrophy conservation: sum(q_interior * J) ~ 0
        # J has shape (N, N), matching q[1:-1, 1:-1]
        q_int = q[1:-1, 1:-1]
        assert J.shape == q_int.shape, f"{J.shape} != {q_int.shape}"
        ens_tendency = jnp.sum(q_int * J) * dx * dy
        np.testing.assert_allclose(ens_tendency, 0.0, atol=1e-10)

    def test_energy_conservation_arakawa_jacobian(self):
        """Arakawa Jacobian conserves energy: sum(psi * J(psi, q)) = 0."""
        N = 16
        dx = dy = 1.0 / N
        Ny = Nx = N + 2
        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)
        psi = jnp.zeros((Ny, Nx), dtype=jnp.float64)
        q = jnp.zeros((Ny, Nx), dtype=jnp.float64)
        psi = psi.at[1:-1, 1:-1].set(jax.random.normal(k1, (N, N), dtype=jnp.float64))
        q = q.at[1:-1, 1:-1].set(jax.random.normal(k2, (N, N), dtype=jnp.float64))
        J = arakawa_jacobian(psi, q, dx, dy)
        psi_int = psi[1:-1, 1:-1]
        energy_tendency = jnp.sum(psi_int * J) * dx * dy
        np.testing.assert_allclose(energy_tendency, 0.0, atol=1e-10)
