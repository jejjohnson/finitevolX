"""Tests for spherical diffusion (harmonic + biharmonic)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion2D, Diffusion2D
from finitevolx._src.diffusion.spherical_diffusion import (
    SphericalBiharmonicDiffusion2D,
    SphericalBiharmonicDiffusion3D,
    SphericalDiffusion2D,
    SphericalDiffusion3D,
)
from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D

R = 6.371e6
NX_INT, NY_INT = 16, 10


@pytest.fixture
def grid():
    return SphericalGrid2D.from_interior(
        nx_interior=NX_INT,
        ny_interior=NY_INT,
        lon_range=(0.0, 360.0),
        lat_range=(-40.0, 40.0),
        R=R,
    )


@pytest.fixture
def grid3d():
    return SphericalGrid3D.from_interior(
        nx_interior=NX_INT,
        ny_interior=NY_INT,
        nz_interior=3,
        lon_range=(0.0, 360.0),
        lat_range=(-40.0, 40.0),
        Lz=100.0,
        R=R,
    )


@pytest.fixture
def diff_op(grid):
    return SphericalDiffusion2D(grid=grid)


@pytest.fixture
def diff_op3d(grid3d):
    return SphericalDiffusion3D(grid=grid3d)


# ======================================================================
# SphericalDiffusion2D
# ======================================================================


class TestSphericalDiffusion2D:
    @pytest.mark.slow
    def test_output_shape(self, diff_op, grid):
        h = jnp.zeros((grid.Ny, grid.Nx))
        out = diff_op(h, kappa=1.0)
        assert out.shape == (grid.Ny, grid.Nx)

    def test_ghost_ring_is_zero(self, diff_op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        out = diff_op(h, kappa=1.0)
        np.testing.assert_allclose(out[0, :], 0.0)
        np.testing.assert_allclose(out[-1, :], 0.0)
        np.testing.assert_allclose(out[:, 0], 0.0)
        np.testing.assert_allclose(out[:, -1], 0.0)

    def test_constant_tracer_zero_tendency(self, diff_op, grid):
        """Constant h → zero tendency everywhere (including non-constant cos)."""
        h = 3.5 * jnp.ones((grid.Ny, grid.Nx))
        out = diff_op(h, kappa=1.0)
        np.testing.assert_allclose(out[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_kappa_scales_linearly(self, diff_op, grid):
        """Doubling κ doubles the tendency."""
        key = jax.random.PRNGKey(0)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        t1 = diff_op(h, kappa=1.0)
        t2 = diff_op(h, kappa=2.0)
        np.testing.assert_allclose(t2[1:-1, 1:-1], 2.0 * t1[1:-1, 1:-1], rtol=1e-10)

    def test_zero_tendency_for_zero_kappa(self, diff_op, grid):
        key = jax.random.PRNGKey(1)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = diff_op(h, kappa=0.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    @pytest.mark.slow
    def test_matches_cartesian_at_narrow_equatorial_band(self):
        """At the equator with a very thin lat band, cos(lat) ≈ 1 and the
        spherical operator reduces to the Cartesian flux-form diffusion
        with dx = R·dlon, dy = R·dlat."""
        nx, ny = 20, 6
        lon_range = (0.0, 20.0)  # small so dlon is small
        lat_range = (-0.5, 0.5)  # very narrow band around equator
        sphere = SphericalGrid2D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            lon_range=lon_range,
            lat_range=lat_range,
            R=R,
        )
        cart = CartesianGrid2D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            Lx=sphere.Lx,
            Ly=sphere.Ly,
        )
        key = jax.random.PRNGKey(3)
        h = jax.random.normal(key, (sphere.Ny, sphere.Nx))

        sph_op = SphericalDiffusion2D(grid=sphere)
        cart_op = Diffusion2D(grid=cart)

        t_sph = sph_op(h, kappa=1.0)
        t_cart = cart_op(h, kappa=1.0)

        # The two should agree to a few tolerance digits because cos(lat)
        # deviates from 1 by O(1e-4) across a 1-degree lat band.
        np.testing.assert_allclose(
            t_sph[1:-1, 1:-1],
            t_cart[1:-1, 1:-1],
            rtol=5e-4,
            atol=1e-4,
        )

    def test_fluxes_shape(self, diff_op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        assert fx.shape == (grid.Ny, grid.Nx)
        assert fy.shape == (grid.Ny, grid.Nx)

    def test_fluxes_constant_zero(self, diff_op, grid):
        h = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        # Constant tracer → zero flux everywhere.
        np.testing.assert_allclose(fx, 0.0, atol=1e-12)
        np.testing.assert_allclose(fy, 0.0, atol=1e-12)

    def test_fluxes_boundary_faces_zero(self, diff_op, grid):
        """East and north domain-wall faces are not written."""
        key = jax.random.PRNGKey(4)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        np.testing.assert_allclose(fx[:, -1], 0.0)  # east wall
        np.testing.assert_allclose(fx[:, -2], 0.0)  # east-wall face (not written)
        np.testing.assert_allclose(fx[0, :], 0.0)
        np.testing.assert_allclose(fx[-1, :], 0.0)
        np.testing.assert_allclose(fy[-1, :], 0.0)
        np.testing.assert_allclose(fy[-2, :], 0.0)

    def test_jit(self, diff_op, grid):
        key = jax.random.PRNGKey(5)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = jax.jit(diff_op.__call__)(h, 1.0)
        assert out.shape == (grid.Ny, grid.Nx)

    @pytest.mark.slow
    def test_grad(self, diff_op, grid):
        key = jax.random.PRNGKey(6)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        g = jax.grad(lambda x: diff_op(x, kappa=1.0).sum())(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))


# ======================================================================
# Masking
# ======================================================================


class TestSphericalDiffusion2DMasked:
    def test_all_dry_mask_zeros_tendency(self, grid):
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        op = SphericalDiffusion2D(grid=grid, mask=all_dry)
        key = jax.random.PRNGKey(7)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = op(h, kappa=1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_all_wet_mask_matches_unmasked(self, grid):
        all_wet = Mask2D.from_mask(np.ones((grid.Ny, grid.Nx), dtype=bool))
        op_masked = SphericalDiffusion2D(grid=grid, mask=all_wet)
        op_unmask = SphericalDiffusion2D(grid=grid)
        key = jax.random.PRNGKey(8)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(
            op_masked(h, kappa=1.0),
            op_unmask(h, kappa=1.0),
            atol=1e-12,
        )

    def test_dry_cell_zero(self, grid):
        """A dry T-cell has zero tendency in the masked tendency."""
        mask = np.ones((grid.Ny, grid.Nx), dtype=bool)
        mask[3, 5] = False  # make an interior cell dry
        op = SphericalDiffusion2D(grid=grid, mask=Mask2D.from_mask(mask))
        key = jax.random.PRNGKey(9)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = op(h, kappa=1.0)
        assert float(out[3, 5]) == 0.0


# ======================================================================
# SphericalDiffusion3D
# ======================================================================


class TestSphericalDiffusion3D:
    @pytest.mark.slow
    def test_output_shape(self, diff_op3d, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = diff_op3d(h, kappa=1.0)
        assert out.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_constant_tracer_zero(self, diff_op3d, grid3d):
        h = 2.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = diff_op3d(h, kappa=1.0)
        np.testing.assert_allclose(out[:, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_z_ghost_slices_zero(self, diff_op3d, grid3d):
        key = jax.random.PRNGKey(10)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = diff_op3d(h, kappa=1.0)
        np.testing.assert_allclose(out[0, :, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(out[-1, :, :], 0.0, atol=1e-10)

    def test_matches_2d_per_level(self, diff_op3d, grid3d):
        """Applying SphericalDiffusion3D on a z-broadcast field should
        equal the 2-D operator at each interior level."""
        op2d = SphericalDiffusion2D(grid=grid3d.horizontal_grid())
        key = jax.random.PRNGKey(11)
        h_2d = jax.random.normal(key, (grid3d.Ny, grid3d.Nx))
        h_3d = jnp.broadcast_to(h_2d, (grid3d.Nz, grid3d.Ny, grid3d.Nx))

        t_2d = op2d(h_2d, kappa=1.0)
        t_3d = diff_op3d(h_3d, kappa=1.0)

        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(t_3d[k], t_2d, atol=1e-12)

    def test_masked_all_dry_zero(self, grid3d):
        all_dry = Mask3D.from_mask(
            np.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        op = SphericalDiffusion3D(grid=grid3d, mask=all_dry)
        key = jax.random.PRNGKey(12)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = op(h, kappa=1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_masked_all_wet_matches_unmasked(self, grid3d):
        all_wet = Mask3D.from_mask(
            np.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        op_m = SphericalDiffusion3D(grid=grid3d, mask=all_wet)
        op_u = SphericalDiffusion3D(grid=grid3d)
        key = jax.random.PRNGKey(13)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        np.testing.assert_allclose(op_m(h, 1.0), op_u(h, 1.0), atol=1e-12)

    def test_fluxes_shape(self, diff_op3d, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        fx, fy = diff_op3d.fluxes(h, kappa=1.0)
        assert fx.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert fy.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    @pytest.mark.slow
    def test_jit_grad(self, diff_op3d, grid3d):
        key = jax.random.PRNGKey(14)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        jitted = jax.jit(diff_op3d.__call__)
        out = jitted(h, 1.0)
        assert out.shape == h.shape
        g = jax.grad(lambda x: diff_op3d(x, 1.0).sum())(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))


# ======================================================================
# SphericalBiharmonicDiffusion2D
# ======================================================================


class TestSphericalBiharmonicDiffusion2D:
    @pytest.fixture
    def op(self, grid):
        return SphericalBiharmonicDiffusion2D(grid=grid)

    def test_output_shape(self, op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        assert op(h, kappa=1.0).shape == (grid.Ny, grid.Nx)

    def test_ghost_ring_zero(self, op, grid):
        key = jax.random.PRNGKey(40)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = op(h, kappa=1.0)
        np.testing.assert_allclose(out[0, :], 0.0)
        np.testing.assert_allclose(out[-1, :], 0.0)
        np.testing.assert_allclose(out[:, 0], 0.0)
        np.testing.assert_allclose(out[:, -1], 0.0)

    def test_constant_field_zero(self, op, grid):
        h = 4.2 * jnp.ones((grid.Ny, grid.Nx))
        np.testing.assert_allclose(op(h, kappa=1.0)[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_kappa_scales_linearly(self, op, grid):
        key = jax.random.PRNGKey(41)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        t1 = op(h, kappa=1.0)
        t2 = op(h, kappa=2.5)
        np.testing.assert_allclose(t2[1:-1, 1:-1], 2.5 * t1[1:-1, 1:-1], rtol=1e-10)

    def test_zero_kappa(self, op, grid):
        key = jax.random.PRNGKey(42)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(op(h, kappa=0.0), 0.0, atol=1e-12)

    @pytest.mark.slow
    def test_sign_is_dissipative_for_small_scales(self, grid):
        """For a short-wave mode, positive kappa should dissipate.

        Evaluates -kappa * nabla^4 h on a sinusoidal field and checks
        that the sign of the tendency opposes the field at interior
        cells — the defining property of dissipative biharmonic mixing.
        """
        op = SphericalBiharmonicDiffusion2D(grid=grid)
        # Use grid indices to get an oscillating pattern.
        jj, ii = jnp.indices((grid.Ny, grid.Nx))
        h = jnp.cos(jnp.pi * ii / 3.0) * jnp.cos(jnp.pi * jj / 3.0)
        tend = op(h, kappa=1.0)
        # In the deep interior, sign(tend) * sign(h) should be <= 0
        # for a dissipative operator.
        deep_h = h[3:-3, 3:-3]
        deep_t = tend[3:-3, 3:-3]
        # Amplitudes where |h| is large — should be clearly opposite-signed.
        large = jnp.abs(deep_h) > 0.5
        assert jnp.all(jnp.where(large, deep_t * deep_h <= 0, True))

    def test_matches_cartesian_at_narrow_equatorial_band(self):
        nx, ny = 20, 6
        sphere = SphericalGrid2D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            lon_range=(0.0, 20.0),
            lat_range=(-0.5, 0.5),
            R=R,
        )
        cart = CartesianGrid2D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            Lx=sphere.Lx,
            Ly=sphere.Ly,
        )
        key = jax.random.PRNGKey(43)
        h = jax.random.normal(key, (sphere.Ny, sphere.Nx))
        t_s = SphericalBiharmonicDiffusion2D(grid=sphere)(h, kappa=1.0)
        t_c = BiharmonicDiffusion2D(grid=cart)(h, kappa=1.0)
        np.testing.assert_allclose(
            t_s[2:-2, 2:-2],
            t_c[2:-2, 2:-2],
            rtol=5e-3,
            atol=5e-3,
        )

    def test_jit_grad(self, op, grid):
        key = jax.random.PRNGKey(44)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = jax.jit(op.__call__)(h, 1.0)
        assert out.shape == h.shape
        g = jax.grad(lambda x: op(x, 1.0).sum())(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))

    def test_masked_all_dry_zero(self, grid):
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        op = SphericalBiharmonicDiffusion2D(grid=grid, mask=all_dry)
        key = jax.random.PRNGKey(45)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(op(h, kappa=1.0), 0.0, atol=1e-12)

    def test_masked_all_wet_matches_unmasked(self, grid):
        all_wet = Mask2D.from_mask(np.ones((grid.Ny, grid.Nx), dtype=bool))
        op_m = SphericalBiharmonicDiffusion2D(grid=grid, mask=all_wet)
        op_u = SphericalBiharmonicDiffusion2D(grid=grid)
        key = jax.random.PRNGKey(46)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(op_m(h, 1.0), op_u(h, 1.0), atol=1e-12)

    def test_dry_cell_zero(self, grid):
        m = np.ones((grid.Ny, grid.Nx), dtype=bool)
        m[4, 7] = False
        op = SphericalBiharmonicDiffusion2D(grid=grid, mask=Mask2D.from_mask(m))
        key = jax.random.PRNGKey(47)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = op(h, kappa=1.0)
        assert float(out[4, 7]) == 0.0


# ======================================================================
# SphericalBiharmonicDiffusion3D
# ======================================================================


class TestSphericalBiharmonicDiffusion3D:
    @pytest.fixture
    def op(self, grid3d):
        return SphericalBiharmonicDiffusion3D(grid=grid3d)

    def test_output_shape(self, op, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert op(h, kappa=1.0).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_constant_zero(self, op, grid3d):
        h = 1.7 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        np.testing.assert_allclose(op(h, kappa=1.0)[:, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_z_ghost_slices_zero(self, op, grid3d):
        key = jax.random.PRNGKey(48)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = op(h, kappa=1.0)
        np.testing.assert_allclose(out[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(out[-1], 0.0, atol=1e-10)

    def test_matches_2d_per_level(self, op, grid3d):
        op2d = SphericalBiharmonicDiffusion2D(grid=grid3d.horizontal_grid())
        key = jax.random.PRNGKey(49)
        h2 = jax.random.normal(key, (grid3d.Ny, grid3d.Nx))
        h3 = jnp.broadcast_to(h2, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        t2 = op2d(h2, kappa=1.0)
        t3 = op(h3, kappa=1.0)
        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(t3[k], t2, atol=1e-12)

    def test_masked_all_dry_zero(self, grid3d):
        all_dry = Mask3D.from_mask(
            np.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        op = SphericalBiharmonicDiffusion3D(grid=grid3d, mask=all_dry)
        key = jax.random.PRNGKey(50)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        np.testing.assert_allclose(op(h, kappa=1.0), 0.0, atol=1e-12)

    def test_jit_grad(self, op, grid3d):
        key = jax.random.PRNGKey(51)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = jax.jit(op.__call__)(h, 1.0)
        assert out.shape == h.shape
        g = jax.grad(lambda x: op(x, 1.0).sum())(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))


# ======================================================================
# #295: variable kappa averaged to the faces (second order)
# ======================================================================


def _sph_kappa_q(lam, phi, walls):
    """κ and q on the box between the closed walls; ∂q/∂n = 0 there."""
    lw, le, ps, pn = walls
    a = jnp.pi * (lam - lw) / (le - lw)
    b = jnp.pi * (phi - ps) / (pn - ps)
    return 1.0 + 0.5 * jnp.sin(a) * jnp.cos(0.5 * b), jnp.cos(a) * jnp.cos(b)


def _sph_exact(lam, phi, walls, R):
    """∇·(κ∇q) = [∂λ(κ/cosφ ∂λq) + ∂φ(κ cosφ ∂φq)] / (R² cosφ)."""

    def f_lam(a, b):
        kap, _ = _sph_kappa_q(a, b, walls)
        return kap / jnp.cos(b) * jax.grad(lambda s: _sph_kappa_q(s, b, walls)[1])(a)

    def f_phi(a, b):
        kap, _ = _sph_kappa_q(a, b, walls)
        return kap * jnp.cos(b) * jax.grad(lambda s: _sph_kappa_q(a, s, walls)[1])(b)

    def div(a, b):
        return (jax.grad(f_lam, 0)(a, b) + jax.grad(f_phi, 1)(a, b)) / (
            R**2 * jnp.cos(b)
        )

    return jax.vmap(jax.vmap(div))(lam, phi)


def _sph_variable_kappa_error(n: int) -> float:
    g = SphericalGrid2D.from_interior(
        n, n, lon_range=(0.0, 60.0), lat_range=(-30.0, 30.0), R=1.0
    )
    lam, phi = g.lon_T, g.lat_T  # radians, [Ny, Nx]
    # closed walls: the faces half a cell outside the first / last interior T
    lw = float(lam[1, 1]) - 0.5 * g.dlon
    ps = float(phi[1, 1]) - 0.5 * g.dlat
    walls = (lw, lw + n * g.dlon, ps, ps + n * g.dlat)
    kap, q = _sph_kappa_q(lam, phi, walls)
    exact = jax.jit(_sph_exact, static_argnums=(2, 3))(lam, phi, walls, 1.0)
    out = SphericalDiffusion2D(grid=g)(q, kappa=kap)
    return float(jnp.abs(out - exact)[1:-1, 1:-1].max())


class TestSphericalVariableKappa:
    @pytest.mark.slow
    def test_convergence_second_order(self):
        """Manufactured solution, κ varying in lon and lat: rate ≥ 1.9
        (first order, rate 1, with the pre-#295 source-cell κ)."""
        errs = [_sph_variable_kappa_error(n) for n in (8, 16, 32)]
        rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(rates >= 1.9), (errs, rates)

    def test_constant_field_bit_identical_to_scalar(self, grid):
        """Eager: under jit XLA may fold a scalar κ into the metric
        constants differently from a field (last-ulp reassociation)."""
        rng = np.random.default_rng(8)
        h = jnp.asarray(rng.normal(size=(grid.Ny, grid.Nx)))
        k = jnp.full((grid.Ny, grid.Nx), 0.37)
        op = SphericalDiffusion2D(grid=grid)
        np.testing.assert_array_equal(op(h, k), op(h, 0.37))
        for a, b in zip(op.fluxes(h, k), op.fluxes(h, 0.37), strict=True):
            np.testing.assert_array_equal(a, b)

    def test_symmetric_area_weighted(self):
        """diag(cos φ_T) · L is symmetric with zero column sums."""
        g = SphericalGrid2D.from_interior(
            6, 6, lon_range=(0.0, 60.0), lat_range=(-30.0, 30.0), R=1.0
        )
        shape = (g.Ny, g.Nx)
        k = jnp.asarray(np.random.default_rng(9).uniform(0.5, 1.5, size=shape))
        op = SphericalDiffusion2D(grid=g)
        n = g.Ny * g.Nx
        M = np.asarray(
            jax.jit(jax.jacfwd(lambda x: op(x.reshape(shape), k).ravel()))(jnp.zeros(n))
        )
        wet = np.zeros(shape, dtype=bool)
        wet[1:-1, 1:-1] = True
        wv = wet.ravel()
        S = np.asarray(g.cos_lat_T).ravel()[wv][:, None] * M[np.ix_(wv, wv)]
        np.testing.assert_allclose(S, S.T, atol=1e-12 * np.abs(S).max())
        np.testing.assert_allclose(S.sum(axis=0), 0.0, atol=1e-11 * np.abs(S).max())
