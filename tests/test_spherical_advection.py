"""Tests for SphericalAdvection2D and SphericalAdvection3D."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.advection import Advection2D, Advection3D
from finitevolx._src.advection.spherical_advection import (
    SphericalAdvection2D,
    SphericalAdvection3D,
)
from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators.reductions import spherical_area_weights

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


# ======================================================================
# SphericalAdvection2D
# ======================================================================


class TestSphericalAdvection2D:
    @pytest.fixture
    def op(self, grid):
        return SphericalAdvection2D(grid=grid)

    @pytest.mark.slow
    def test_output_shape(self, op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        out = op(h, u, v)
        assert out.shape == (grid.Ny, grid.Nx)

    def test_zero_velocity_zero_tendency(self, op, grid):
        key = jax.random.PRNGKey(0)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        u = jnp.zeros_like(h)
        v = jnp.zeros_like(h)
        out = op(h, u, v, method="upwind1")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "method",
        [
            "naive",
            "upwind1",
            "upwind2",
            "upwind3",
            "weno3",
            "weno5",
            "wenoz5",
            "weno7",
            "weno9",
            "minmod",
            "van_leer",
            "superbee",
            "mc",
        ],
    )
    @pytest.mark.slow
    def test_methods_produce_finite_output(self, op, grid, method):
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = op(h, u, v, method=method)
        assert jnp.all(jnp.isfinite(out))

    def test_ghost_ring_zero(self, op, grid):
        key = jax.random.PRNGKey(2)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = op(h, u, v, method="upwind1")
        np.testing.assert_allclose(out[0, :], 0.0)
        np.testing.assert_allclose(out[-1, :], 0.0)
        np.testing.assert_allclose(out[:, 0], 0.0)
        np.testing.assert_allclose(out[:, -1], 0.0)

    @pytest.mark.slow
    def test_matches_cartesian_at_narrow_equatorial_band(self):
        """Spherical advection at the equator (cos≈1) should match
        Cartesian advection with dx=R·dlon, dy=R·dlat."""
        nx, ny = 24, 6
        lon_range = (0.0, 30.0)
        lat_range = (-0.5, 0.5)
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
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (sphere.Ny, sphere.Nx))
        u = jax.random.normal(k2, (sphere.Ny, sphere.Nx))
        v = jax.random.normal(k3, (sphere.Ny, sphere.Nx))

        sph_op = SphericalAdvection2D(grid=sphere)
        cart_op = Advection2D(grid=cart)

        for method in ("upwind1", "weno3", "weno5"):
            t_s = sph_op(h, u, v, method=method)
            t_c = cart_op(h, u, v, method=method)
            # Across 1-degree latitude, cos varies by O(1e-4).
            np.testing.assert_allclose(
                t_s[2:-2, 2:-2],
                t_c[2:-2, 2:-2],
                rtol=5e-3,
                atol=1e-3,
            )

    def test_jit(self, op, grid):
        key = jax.random.PRNGKey(4)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = jax.jit(lambda a, b, c: op(a, b, c, method="weno5"))(h, u, v)
        assert out.shape == h.shape
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.slow
    def test_grad(self, op, grid):
        key = jax.random.PRNGKey(5)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))

        def loss(h_, u_, v_):
            return op(h_, u_, v_, method="upwind1").sum()

        gh, gu, gv = jax.grad(loss, argnums=(0, 1, 2))(h, u, v)
        for g in (gh, gu, gv):
            assert g.shape == h.shape
            assert jnp.all(jnp.isfinite(g))

    def test_uniform_h_positive_u_zero_v(self, grid):
        """Uniform h with uniform u>0 → -div(h·u·cosφ) dependence.

        For h constant and v=0:
            dh/dt = -(1/(R·cosφ))·∂(h·u)/∂λ
        If additionally u is uniform in λ, this term is zero. Verify.
        """
        op = SphericalAdvection2D(grid=grid)
        h = 2.0 * jnp.ones((grid.Ny, grid.Nx))
        u = 1.5 * jnp.ones((grid.Ny, grid.Nx))
        v = jnp.zeros_like(h)
        out = op(h, u, v, method="upwind1")
        np.testing.assert_allclose(out[2:-2, 2:-2], 0.0, atol=1e-10)


# ======================================================================
# Masking
# ======================================================================


class TestSphericalAdvection2DMasked:
    def test_all_dry_mask_zero(self, grid):
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        op = SphericalAdvection2D(grid=grid, mask=all_dry)
        key = jax.random.PRNGKey(6)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = op(h, u, v, method="weno3")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_all_wet_matches_unmasked_upwind1(self, grid):
        all_wet = Mask2D.from_mask(np.ones((grid.Ny, grid.Nx), dtype=bool))
        op_m = SphericalAdvection2D(grid=grid, mask=all_wet)
        op_u = SphericalAdvection2D(grid=grid)
        key = jax.random.PRNGKey(7)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        # Non-dispatchable method — should match bit-for-bit.
        np.testing.assert_allclose(
            op_m(h, u, v, method="upwind1"),
            op_u(h, u, v, method="upwind1"),
            atol=1e-12,
        )

    def test_dry_cell_tendency_zero(self, grid):
        mask = np.ones((grid.Ny, grid.Nx), dtype=bool)
        mask[4, 6] = False
        op = SphericalAdvection2D(grid=grid, mask=Mask2D.from_mask(mask))
        key = jax.random.PRNGKey(8)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = op(h, u, v, method="weno3")
        assert float(out[4, 6]) == 0.0


# ======================================================================
# SphericalAdvection3D
# ======================================================================


class TestSphericalAdvection3D:
    @pytest.fixture
    def op(self, grid3d):
        return SphericalAdvection3D(grid=grid3d)

    def test_output_shape(self, op, grid3d):
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jnp.ones(shape)
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)
        out = op(h, u, v)
        assert out.shape == shape

    def test_zero_velocity_zero_tendency(self, op, grid3d):
        key = jax.random.PRNGKey(9)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.zeros_like(h)
        v = jnp.zeros_like(h)
        out = op(h, u, v, method="upwind1")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "method",
        [
            "naive",
            "upwind1",
            "weno3",
            "weno5",
            "weno7",
            "weno9",
            "minmod",
            "van_leer",
            "superbee",
            "mc",
        ],
    )
    @pytest.mark.slow
    def test_methods_produce_finite_output(self, op, grid3d, method):
        key = jax.random.PRNGKey(15)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)
        out = op(h, u, v, method=method)
        assert jnp.all(jnp.isfinite(out))

    def test_z_ghost_slices_zero(self, op, grid3d):
        key = jax.random.PRNGKey(10)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)
        out = op(h, u, v, method="weno3")
        np.testing.assert_allclose(out[0, :, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(out[-1, :, :], 0.0, atol=1e-10)

    def test_matches_2d_per_level(self, op, grid3d):
        """On a z-broadcast field, the 3-D spherical advection equals
        the 2-D spherical advection at each interior level."""
        op2d = SphericalAdvection2D(grid=grid3d.horizontal_grid())
        key = jax.random.PRNGKey(11)
        k1, k2, k3 = jax.random.split(key, 3)
        h_2d = jax.random.normal(k1, (grid3d.Ny, grid3d.Nx))
        u_2d = jax.random.normal(k2, (grid3d.Ny, grid3d.Nx))
        v_2d = jax.random.normal(k3, (grid3d.Ny, grid3d.Nx))
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h_3d = jnp.broadcast_to(h_2d, shape)
        u_3d = jnp.broadcast_to(u_2d, shape)
        v_3d = jnp.broadcast_to(v_2d, shape)

        t_2d = op2d(h_2d, u_2d, v_2d, method="weno3")
        t_3d = op(h_3d, u_3d, v_3d, method="weno3")

        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(t_3d[k], t_2d, atol=1e-12)

    @pytest.mark.slow
    def test_matches_cartesian_at_narrow_equatorial_band(self):
        """Narrow lat band around equator → spherical 3D ≈ Cartesian 3D."""
        nx, ny, nz = 16, 6, 3
        sphere = SphericalGrid3D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            nz_interior=nz,
            lon_range=(0.0, 20.0),
            lat_range=(-0.5, 0.5),
            Lz=100.0,
            R=R,
        )
        cart = CartesianGrid3D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            nz_interior=nz,
            Lx=sphere.Lx,
            Ly=sphere.Ly,
            Lz=100.0,
        )
        key = jax.random.PRNGKey(12)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (sphere.Nz, sphere.Ny, sphere.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)

        op_s = SphericalAdvection3D(grid=sphere)
        op_c = Advection3D(grid=cart)

        for method in ("upwind1", "weno3"):
            t_s = op_s(h, u, v, method=method)
            t_c = op_c(h, u, v, method=method)
            np.testing.assert_allclose(
                t_s[1:-1, 2:-2, 2:-2],
                t_c[1:-1, 2:-2, 2:-2],
                rtol=5e-3,
                atol=1e-3,
            )

    def test_masked_all_dry_zero(self, grid3d):
        all_dry = Mask3D.from_mask(
            np.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        op = SphericalAdvection3D(grid=grid3d, mask=all_dry)
        key = jax.random.PRNGKey(13)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)
        out = op(h, u, v, method="weno3")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    @pytest.mark.slow
    def test_jit_grad(self, op, grid3d):
        key = jax.random.PRNGKey(14)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)
        jitted = jax.jit(lambda a, b, c: op(a, b, c, method="weno3"))
        out = jitted(h, u, v)
        assert out.shape == shape
        g = jax.grad(lambda a: op(a, u, v, method="upwind1").sum())(h)
        assert g.shape == shape
        assert jnp.all(jnp.isfinite(g))


# ======================================================================
# Telescoping against spherical_area_weights (#247, #297)
# ======================================================================

#: Rounding of a sum of ~1000 O(1)-relative terms (measured 1e-18 - 3e-17);
#: the closed-wall defect of #247 is 5e-3.
TELESCOPE_RTOL = 1e-12

NP_LON, NY_LAT = 24, 12  # physical longitudes (360 degrees) and latitudes

#: Minimum periodic halo per method (module docstring of
#: ``spherical_advection.py``).
HALO = {"upwind1": 1, "van_leer": 2, "weno5": 3}


def _periodic_setup(halo, seed=0):
    """A zonally periodic grid with ``halo`` wrapped columns on each side.

    The grid carries ``halo - 1`` extra columns beyond the ghost ring, the
    physical longitudes are columns ``[halo:-halo]``, and every field is
    filled by wrapping the ``NP_LON`` physical columns.  ``v = 0`` on the
    south / north boundary faces, so nothing crosses those walls.
    """
    dl = 360.0 / NP_LON
    pad = (halo - 1) * dl
    grid = SphericalGrid2D.from_interior(
        nx_interior=NP_LON + 2 * (halo - 1),
        ny_interior=NY_LAT,
        lon_range=(-pad, 360.0 + pad),
        lat_range=(-60.0, 60.0),
        R=R,
    )
    rng = np.random.default_rng(seed)
    shape = (grid.Ny, NP_LON)
    lon = np.deg2rad(np.arange(NP_LON) * dl)
    h = 1000.0 + 10.0 * np.sin(lon)[None, :] + rng.standard_normal(shape)
    u = rng.standard_normal(shape)
    v = rng.standard_normal(shape)
    v[0] = 0.0
    v[-2] = 0.0
    cols = (np.arange(grid.Nx) - halo) % NP_LON
    fields = tuple(jnp.asarray(f[:, cols]) for f in (h, u, v))
    physical = (slice(1, -1), slice(halo, halo + NP_LON))
    return grid, fields, physical


def _advect(op, h, u, v, method, wall="closed"):
    """``op(h, u, v, method, wall)`` under ``jax.jit`` (one compiled program:
    ~5x faster than eager on these grids, keeping the tests in the fast tier)."""
    return jax.jit(lambda a, b, c: op(a, b, c, method=method, wall=wall))(h, u, v)


def _relative_net(area, tendency, region=(slice(1, -1), slice(1, -1))):
    """``sum(A t) / sum(A |t|)`` over ``region``: 0 when the fluxes telescope."""
    weighted = area[region] * tendency[region]
    return float(jnp.sum(weighted) / jnp.sum(jnp.abs(weighted)))


class TestSphericalTelescoping:
    def test_divergence_zonally_periodic(self):
        """``Σ A · SphericalDivergence2D(u, v)`` over the interior: the metric
        is self-consistent (#297 audit), so this is rounding."""
        from finitevolx import SphericalDivergence2D

        grid, (_, u, v), _ = _periodic_setup(halo=1)
        div = SphericalDivergence2D(grid=grid)(u, v)
        area = spherical_area_weights(grid)
        assert abs(_relative_net(area, div)) < TELESCOPE_RTOL

    def test_divergence_zero_boundary_velocity(self):
        from finitevolx import SphericalDivergence2D

        grid, (_, u, v), _ = _periodic_setup(halo=1)
        u = u.at[:, 0].set(0.0).at[:, -2].set(0.0)  # west / east wall faces
        div = SphericalDivergence2D(grid=grid)(u, v)
        area = spherical_area_weights(grid)
        assert abs(_relative_net(area, div)) < TELESCOPE_RTOL

    @pytest.mark.parametrize("method", ["upwind1", "weno5", "van_leer"])
    def test_issue_247_script(self, method):
        """The #247 reproduction (48x24, ±80°, one wrapped ghost column):
        -5.0e-3 with the default wall="closed", rounding with "open"."""
        grid = SphericalGrid2D.from_interior(48, 24, (0.0, 360.0), (-80.0, 80.0), R=R)
        rng = np.random.RandomState(0)

        def wrap(a):
            return a.at[:, 0].set(a[:, -2]).at[:, -1].set(a[:, 1])

        h = wrap(jnp.asarray(1000.0 + 10 * rng.randn(grid.Ny, grid.Nx)))
        u = wrap(jnp.asarray(rng.randn(grid.Ny, grid.Nx)))
        op = SphericalAdvection2D(grid=grid)
        area = spherical_area_weights(grid)
        closed = _advect(op, h, u, jnp.zeros_like(u), method)
        opened = _advect(op, h, u, jnp.zeros_like(u), method, wall="open")
        assert abs(_relative_net(area, closed)) > 1e-3  # the #247 defect
        assert abs(_relative_net(area, opened)) < TELESCOPE_RTOL

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["upwind1", "weno5", "van_leer"])
    def test_periodic_with_the_documented_halo(self, method):
        """With the documented halo the physical tendency is the periodic
        scheme's (equal to a much wider halo) and telescopes."""
        grid, (h, u, v), physical = _periodic_setup(halo=HALO[method])
        out = _advect(SphericalAdvection2D(grid=grid), h, u, v, method, "open")
        area = spherical_area_weights(grid)
        assert abs(_relative_net(area, out, physical)) < TELESCOPE_RTOL

        wide, (hw, uw, vw), physical_w = _periodic_setup(halo=6)
        ref = _advect(SphericalAdvection2D(grid=wide), hw, uw, vw, method, "open")
        np.testing.assert_allclose(
            out[physical],
            ref[physical_w],
            rtol=0.0,
            atol=1e-13 * float(jnp.max(jnp.abs(ref[physical_w]))),
        )

    @pytest.mark.parametrize("method", ["upwind1", "weno5", "van_leer"])
    def test_zero_velocity_on_the_boundary_faces(self, method):
        """No flux through any wall face: Σ A · tendency over the full
        interior is rounding (the closed write range drops the ring)."""
        grid, (h, u, v), _ = _periodic_setup(halo=1, seed=3)
        u = u.at[:, 0].set(0.0).at[:, -2].set(0.0)
        out = _advect(SphericalAdvection2D(grid=grid), h, u, v, method, "open")
        area = spherical_area_weights(grid)
        assert abs(_relative_net(area, out)) < TELESCOPE_RTOL


class TestSphericalOpenWall:
    @pytest.mark.parametrize("method", ["upwind1", "weno5", "van_leer"])
    def test_closed_is_the_strict_interior_of_open(self, grid, method):
        """The default is unchanged, and "open" only adds the outer ring."""
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(21), 3)
        shape = (grid.Ny, grid.Nx)
        h, u, v = (jax.random.normal(k, shape) for k in (k1, k2, k3))
        op = SphericalAdvection2D(grid=grid)
        closed = _advect(op, h, u, v, method)
        opened = _advect(op, h, u, v, method, wall="open")
        np.testing.assert_array_equal(closed, _advect(op, h, u, v, method, "closed"))
        # same expressions on [2:-2, 2:-2]; only XLA's fusion can differ (ulp)
        np.testing.assert_allclose(
            opened[2:-2, 2:-2], closed[2:-2, 2:-2], rtol=1e-14, atol=1e-14
        )
        assert float(jnp.max(jnp.abs(opened[1, 1:-1]))) > 0.0
        np.testing.assert_array_equal(opened[0], 0.0)
        np.testing.assert_array_equal(opened[:, -1], 0.0)

    @pytest.mark.slow
    def test_matches_cartesian_open_wall_at_the_equator(self):
        """Mirrors Advection2D(wall="open"), ring included, where cos ~ 1."""
        sphere = SphericalGrid2D.from_interior(
            nx_interior=24,
            ny_interior=6,
            lon_range=(0.0, 30.0),
            lat_range=(-0.5, 0.5),
            R=R,
        )
        cart = CartesianGrid2D.from_interior(
            nx_interior=24, ny_interior=6, Lx=sphere.Lx, Ly=sphere.Ly
        )
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(22), 3)
        shape = (sphere.Ny, sphere.Nx)
        h, u, v = (jax.random.normal(k, shape) for k in (k1, k2, k3))
        for method in ("upwind1", "weno5", "van_leer"):
            t_s = SphericalAdvection2D(grid=sphere)(h, u, v, method, wall="open")
            t_c = Advection2D(grid=cart)(h, u, v, method, wall="open")
            # Across 1 degree of latitude, cos varies by O(1e-4).
            # metric departure at |lat| <= 0.5 deg: 1 - cos(0.5 deg) ~ 3.8e-5
            np.testing.assert_allclose(
                t_s, t_c, rtol=0, atol=1e-4 * float(jnp.max(jnp.abs(t_c)))
            )

    @pytest.mark.slow
    def test_3d_matches_2d_per_level(self, grid, grid3d):
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(23), 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h, u, v = (jax.random.normal(k, shape) for k in (k1, k2, k3))
        out3 = SphericalAdvection3D(grid=grid3d)(h, u, v, "weno5", wall="open")
        op2 = SphericalAdvection2D(grid=grid)
        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(
                out3[k], op2(h[k], u[k], v[k], "weno5", wall="open"), rtol=1e-12
            )
        np.testing.assert_array_equal(out3[0], 0.0)
        np.testing.assert_array_equal(out3[-1], 0.0)

    def test_open_with_mask_raises(self, grid):
        mask = Mask2D.from_mask(np.ones((grid.Ny, grid.Nx), dtype=bool))
        z = jnp.zeros((grid.Ny, grid.Nx))
        with pytest.raises(NotImplementedError, match="wall='open'"):
            SphericalAdvection2D(grid=grid, mask=mask)(z, z, z, wall="open")

    def test_unknown_wall_raises(self, grid):
        z = jnp.zeros((grid.Ny, grid.Nx))
        with pytest.raises(ValueError, match="wall must be one of"):
            SphericalAdvection2D(grid=grid)(z, z, z, wall="periodic")
