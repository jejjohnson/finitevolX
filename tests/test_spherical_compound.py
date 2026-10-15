"""Tests for spherical compound operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.spherical import (
    SphericalGrid2D,
    SphericalGrid3D,
)
from finitevolx._src.mask.cartesian import Mask2D
from finitevolx._src.operators.reductions import spherical_area_weights
from finitevolx._src.operators.spherical_compound import (
    SphericalDivergence2D,
    SphericalDivergence3D,
    SphericalLaplacian2D,
    SphericalLaplacian3D,
    SphericalVorticity2D,
    SphericalVorticity3D,
    geostrophic_velocity_sphere,
)
from finitevolx._src.operators.spherical_difference import SphericalDifference2D
from finitevolx._src.operators.vorticity import Vorticity2D

R = 1.0
NX_INT, NY_INT = 10, 8


@pytest.fixture
def grid():
    return SphericalGrid2D.from_interior(
        nx_interior=NX_INT,
        ny_interior=NY_INT,
        lon_range=(0.0, 360.0),
        lat_range=(-80.0, 80.0),
        R=R,
    )


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


# ======================================================================
# SphericalDivergence2D
# ======================================================================


class TestSphericalDivergence2D:
    @pytest.fixture
    def div_op(self, grid):
        return SphericalDivergence2D(grid=grid)

    def test_output_shape(self, div_op, grid):
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        assert div_op(u, v).shape == (grid.Ny, grid.Nx)

    def test_uniform_u_zero_v(self, div_op, grid):
        u = 3.0 * jnp.ones((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        result = div_op(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_zero_u_uniform_v(self, div_op, grid):
        """Uniform v, zero u => divergence is non-zero due to spherical metric."""
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = 1.0 * jnp.ones((grid.Ny, grid.Nx))
        result = div_op(u, v)
        assert not jnp.allclose(result[2:-2, 2:-2], 0.0, atol=1e-10)


# ======================================================================
# SphericalVorticity2D
# ======================================================================


class TestSphericalVorticity2D:
    @pytest.fixture
    def vort_op(self, grid):
        return SphericalVorticity2D(grid=grid)

    def test_output_shape(self, vort_op, grid):
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        assert vort_op.relative_vorticity(u, v).shape == (grid.Ny, grid.Nx)

    def test_zero_velocity(self, vort_op, grid):
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        result = vort_op.relative_vorticity(u, v)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    @pytest.mark.slow
    def test_masked_pv_grad_finite_and_forward_unchanged(self, grid):
        """Masked PV has finite reverse-mode gradients (#287)."""
        m = np.ones((grid.Ny, grid.Nx), bool)
        m[[0, -1], :] = False
        m[:, [0, -1]] = False
        m[3:5, 4:7] = False
        mask = Mask2D.from_mask(jnp.asarray(m))
        op = SphericalVorticity2D(grid, mask=mask)
        ku, kv, kh = jax.random.split(jax.random.PRNGKey(5), 3)
        shp = (grid.Ny, grid.Nx)
        u = jax.random.normal(ku, shp)
        v = jax.random.normal(kv, shp)
        h = 10.0 + jax.random.uniform(kh, shp)
        f = jnp.ones(shp)

        def total(u, v, h):
            return op.potential_vorticity(u, v, h, f).sum()

        for g in jax.grad(total, argnums=(0, 1, 2))(u, v, h):
            assert bool(jnp.all(jnp.isfinite(g)))

        # Forward values on wet corners equal the unguarded formula.
        pv = op.potential_vorticity(u, v, h, f)
        zeta = op.relative_vorticity(u, v)
        num = zeta + op._interp.T_to_X(f)
        den = op._interp.T_to_X(h)
        wet = np.asarray(mask.xy_corner_strict)
        wet_int = wet.copy()
        wet_int[[0, -1], :] = False
        wet_int[:, [0, -1]] = False
        ref = np.asarray(num / den)
        np.testing.assert_allclose(np.asarray(pv)[wet_int], ref[wet_int], rtol=1e-14)
        np.testing.assert_array_equal(np.asarray(pv)[~wet], 0.0)

    def test_pv_flux_matches_cartesian(self, vort_op, grid, rng):
        """PV flux methods are coordinate-independent — should match Cartesian."""
        cart_vort = Vorticity2D(grid=grid)
        k1, k2, k3 = jax.random.split(rng, 3)
        q = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))

        # Energy-conserving
        qu_s, qv_s = vort_op.pv_flux_energy_conserving(q, u, v)
        qu_c, qv_c = cart_vort.pv_flux_energy_conserving(q, u, v)
        np.testing.assert_allclose(qu_s, qu_c, atol=1e-10)
        np.testing.assert_allclose(qv_s, qv_c, atol=1e-10)

        # Enstrophy-conserving
        qu_s, qv_s = vort_op.pv_flux_enstrophy_conserving(q, u, v)
        qu_c, qv_c = cart_vort.pv_flux_enstrophy_conserving(q, u, v)
        np.testing.assert_allclose(qu_s, qu_c, atol=1e-10)
        np.testing.assert_allclose(qv_s, qv_c, atol=1e-10)

        # energy/enstrophy blend
        qu_s, qv_s = vort_op.pv_flux_blend(q, u, v)
        qu_c, qv_c = cart_vort.pv_flux_blend(q, u, v)
        np.testing.assert_allclose(qu_s, qu_c, atol=1e-10)
        np.testing.assert_allclose(qv_s, qv_c, atol=1e-10)


# ======================================================================
# SphericalLaplacian2D
# ======================================================================


class TestSphericalLaplacian2D:
    @pytest.fixture
    def lap_op(self, grid):
        return SphericalLaplacian2D(grid=grid)

    def test_output_shape(self, lap_op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        assert lap_op(h).shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero(self, lap_op, grid):
        h = 7.0 * jnp.ones((grid.Ny, grid.Nx))
        result = lap_op(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_equals_div_of_grad(self, lap_op, grid):
        """Lap(h) == Div(Grad h) to roundoff (#288): both read cos_lat_V."""
        h = jax.random.normal(jax.random.PRNGKey(1), (grid.Ny, grid.Nx))
        h = h.at[:, 0].set(h[:, -2]).at[:, -1].set(h[:, 1])  # periodic in lon
        diff = SphericalDifference2D(grid=grid)
        gu = diff.diff_lon_T_to_U(h)  # [Ny, Nx]
        # west boundary face U[:, 0] is BC-owned; periodic wrap of the east one
        gu = gu.at[:, 0].set(gu[:, -2])
        gv = diff.diff_lat_T_to_V(h)  # [Ny, Nx]
        # the gradient writes only [1:-1]; fill the south boundary face
        # V[0] = (h[1] - h[0]) / (R dlat) that the divergence of row 1 reads
        gv = gv.at[0, :].set((h[1, :] - h[0, :]) / (grid.R * grid.dlat))
        div_grad = SphericalDivergence2D(grid=grid)(gu, gv)
        np.testing.assert_allclose(
            lap_op(h)[1:-1, 1:-1], div_grad[1:-1, 1:-1], rtol=0, atol=1e-12
        )

    def test_laplacian_merid_equals_div_of_lat_grad(self, grid):
        """laplacian_merid(h) == Div(0, d_lat h) to roundoff (#288)."""
        h = jax.random.normal(jax.random.PRNGKey(2), (grid.Ny, grid.Nx))
        diff = SphericalDifference2D(grid=grid)
        gv = diff.diff_lat_T_to_V(h)
        gv = gv.at[0, :].set((h[1, :] - h[0, :]) / (grid.R * grid.dlat))
        div_gv = SphericalDivergence2D(grid=grid)(jnp.zeros_like(gv), gv)
        np.testing.assert_allclose(
            diff.laplacian_merid(h)[1:-1, 1:-1],
            div_gv[1:-1, 1:-1],
            rtol=0,
            atol=1e-12,
        )

    def test_closed_strip_conservation(self, lap_op, grid):
        """sum(area * Lap(h)) = 0 with periodic lon and no-flux lat walls."""
        h = jax.random.normal(jax.random.PRNGKey(3), (grid.Ny, grid.Nx))
        h = h.at[0, :].set(h[1, :]).at[-1, :].set(h[-2, :])  # zero lat flux
        h = h.at[:, 0].set(h[:, -2]).at[:, -1].set(h[:, 1])  # periodic in lon
        area = spherical_area_weights(grid)
        total = jnp.sum((area * lap_op(h))[1:-1, 1:-1])
        scale = jnp.sum(jnp.abs(area * lap_op(h))[1:-1, 1:-1])
        assert abs(float(total)) < 1e-14 * float(scale)


# ======================================================================
# Geostrophic velocity
# ======================================================================


class TestGeostrophicVelocity:
    def test_output_shapes(self, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        f = 1e-4 * jnp.ones((grid.Ny, grid.Nx))
        u_g, v_g = geostrophic_velocity_sphere(h, f, grid)
        assert u_g.shape == (grid.Ny, grid.Nx)
        assert v_g.shape == (grid.Ny, grid.Nx)

    def test_constant_h_zero(self, grid):
        h = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        f = 1e-4 * jnp.ones((grid.Ny, grid.Nx))
        u_g, v_g = geostrophic_velocity_sphere(h, f, grid)
        np.testing.assert_allclose(u_g[1:-1, 1:-1], 0.0, atol=1e-10)
        np.testing.assert_allclose(v_g[1:-1, 1:-1], 0.0, atol=1e-10)


# ======================================================================
# 3D operators
# ======================================================================


class TestSpherical3DCompound:
    @pytest.fixture
    def grid3d(self):
        return SphericalGrid3D.from_interior(
            nx_interior=NX_INT,
            ny_interior=NY_INT,
            nz_interior=4,
            lon_range=(0.0, 360.0),
            lat_range=(-80.0, 80.0),
            Lz=100.0,
            R=R,
        )

    def test_div3d_shape(self, grid3d):
        div3d = SphericalDivergence3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert div3d(u, v).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_vort3d_shape(self, grid3d):
        vort3d = SphericalVorticity3D(grid=grid3d)
        u = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert vort3d.relative_vorticity(u, v).shape == (
            grid3d.Nz,
            grid3d.Ny,
            grid3d.Nx,
        )

    def test_lap3d_shape(self, grid3d):
        lap3d = SphericalLaplacian3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert lap3d(h).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_lap3d_constant_zero(self, grid3d):
        lap3d = SphericalLaplacian3D(grid=grid3d)
        h = 7.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = lap3d(h)
        np.testing.assert_allclose(result[:, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_div3d_z_ghost_slices_zero(self, grid3d):
        div3d = SphericalDivergence3D(grid=grid3d)
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jax.random.normal(k2, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = div3d(u, v)
        np.testing.assert_allclose(result[0, :, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :, :], 0.0, atol=1e-10)

    def test_lap3d_z_ghost_slices_zero(self, grid3d):
        lap3d = SphericalLaplacian3D(grid=grid3d)
        h = jax.random.normal(jax.random.PRNGKey(7), (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = lap3d(h)
        np.testing.assert_allclose(result[0, :, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :, :], 0.0, atol=1e-10)


# ======================================================================
# JIT + grad
# ======================================================================


class TestJitGradCompound:
    def test_jit_divergence(self, grid, rng):
        div_op = SphericalDivergence2D(grid=grid)
        k1, k2 = jax.random.split(rng)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        result = jax.jit(lambda a, b: div_op(a, b))(u, v)  # noqa: PLW0108
        assert result.shape == (grid.Ny, grid.Nx)

    def test_grad_laplacian(self, grid, rng):
        lap_op = SphericalLaplacian2D(grid=grid)
        h = jax.random.normal(rng, (grid.Ny, grid.Nx))
        g = jax.grad(lambda x: lap_op(x).sum())(h)
        assert g.shape == h.shape
