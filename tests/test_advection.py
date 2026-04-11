"""Tests for Advection1D, Advection2D, Advection3D."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.advection import Advection1D, Advection2D, Advection3D
from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)
from finitevolx._src.mask.cgrid_mask import ArakawaCGridMask


@pytest.fixture
def grid1d():
    return CartesianGrid1D.from_interior(8, 1.0)


@pytest.fixture
def grid2d():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


class TestAdvection1D:
    def test_output_shape(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert adv(h, u).shape == (grid1d.Nx,)

    def test_constant_field_zero_tendency(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = adv(h, u, method="upwind1")
        # strictly interior (away from ghost edges): flux difference = 0
        np.testing.assert_allclose(result[2:-2], 0.0, atol=1e-10)

    def test_all_methods_run(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        for method in [
            "naive",
            "upwind1",
            "upwind2",
            "upwind3",
            "weno3",
            "weno5",
            "weno7",
            "weno9",
        ]:
            result = adv(h, u, method=method)
            assert result.shape == (grid1d.Nx,)

    def test_weno_constant_zero_tendency(self):
        grid = CartesianGrid1D.from_interior(16, 1.0)
        adv = Advection1D(grid=grid)
        h = jnp.ones(grid.Nx)
        u = jnp.ones(grid.Nx)
        for method, depth in [("weno3", 2), ("weno5", 3), ("weno7", 4), ("weno9", 5)]:
            result = adv(h, u, method=method)
            np.testing.assert_allclose(result[depth:-depth], 0.0, atol=1e-6)

    def test_tvd_methods_run(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, method=method)
            assert result.shape == (grid1d.Nx,)

    def test_tvd_constant_zero_tendency(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, method=method)
            np.testing.assert_allclose(result[2:-2], 0.0, atol=1e-10)

    def test_unknown_method_raises(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        with pytest.raises(ValueError, match="Unknown method"):
            adv(h, u, method="invalid")

    def test_ghost_zero(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = adv(h, u)
        # Ghost ring and boundary layer stay zero
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[1], 0.0)
        np.testing.assert_allclose(result[-2], 0.0)
        np.testing.assert_allclose(result[-1], 0.0)


class TestAdvection2D:
    def test_output_shape(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert adv(h, u, v).shape == (grid2d.Ny, grid2d.Nx)

    def test_constant_field_zero_tendency(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = adv(h, u, v, method="upwind1")
        # strictly interior (away from ghost edges): flux difference = 0
        np.testing.assert_allclose(result[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_all_methods_run(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        for method in [
            "naive",
            "upwind1",
            "upwind2",
            "upwind3",
            "weno3",
            "weno5",
            "weno7",
            "weno9",
        ]:
            result = adv(h, u, v, method=method)
            assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_weno_constant_zero_tendency(self):
        grid = CartesianGrid2D.from_interior(16, 16, 1.0, 1.0)
        adv = Advection2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        for method, depth in [("weno3", 2), ("weno5", 3), ("weno7", 4), ("weno9", 5)]:
            result = adv(h, u, v, method=method)
            np.testing.assert_allclose(
                result[depth:-depth, depth:-depth], 0.0, atol=1e-6
            )

    def test_tvd_methods_run(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, v, method=method)
            assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_tvd_constant_zero_tendency(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, v, method=method)
            np.testing.assert_allclose(result[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_ghost_ring_zero(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = adv(h, u, v)
        # Ghost ring and boundary layers stay zero
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[1, :], 0.0)
        np.testing.assert_array_equal(result[-2, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, 1], 0.0)
        np.testing.assert_array_equal(result[:, -2], 0.0)
        np.testing.assert_array_equal(result[:, -1], 0.0)

    def test_unknown_method_raises(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        with pytest.raises(ValueError, match="Unknown method"):
            adv(h, u, v, method="bogus")


class TestAdvection3D:
    def test_output_shape(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert adv(h, u, v).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_constant_zero_tendency(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = adv(h, u, v, method="upwind1")
        # Valid region: all z-interior levels, deep horizontal interior
        # (avoids ghost-adjacent horizontal cells where flux ghosts are 0).
        np.testing.assert_allclose(result[1:-1, 2:-2, 2:-2], 0.0, atol=1e-10)

    def test_ghost_ring_zero(self, grid3d):
        """Ghost and boundary-adjacent rings must stay zero (no flux ghost set)."""
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = adv(h, u, v)
        # Outer ghost rows/cols
        np.testing.assert_array_equal(result[:, 0, :], 0.0)
        np.testing.assert_array_equal(result[:, -1, :], 0.0)
        np.testing.assert_array_equal(result[:, :, 0], 0.0)
        np.testing.assert_array_equal(result[:, :, -1], 0.0)
        # Second ring (boundary-adjacent horizontal cells, flux ghost not set)
        np.testing.assert_array_equal(result[:, 1, :], 0.0)
        np.testing.assert_array_equal(result[:, -2, :], 0.0)
        np.testing.assert_array_equal(result[:, :, 1], 0.0)
        np.testing.assert_array_equal(result[:, :, -2], 0.0)

    def test_tvd_methods_run(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, v, method=method)
            assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_weno_methods_run(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for method in ["weno3", "weno5", "weno7", "weno9"]:
            result = adv(h, u, v, method=method)
            assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_weno_constant_zero_tendency(self):
        grid = CartesianGrid3D.from_interior(16, 16, 4, 1.0, 1.0, 1.0)
        adv = Advection3D(grid=grid)
        h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        u = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        v = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        for method, depth in [("weno3", 2), ("weno5", 3), ("weno7", 4), ("weno9", 5)]:
            result = adv(h, u, v, method=method)
            np.testing.assert_allclose(
                result[1:-1, depth:-depth, depth:-depth], 0.0, atol=1e-6
            )

    def test_tvd_constant_zero_tendency(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, v, method=method)
            np.testing.assert_allclose(result[1:-1, 2:-2, 2:-2], 0.0, atol=1e-10)


# ── Mask-aware Advection2D ────────────────────────────────────────────────────


class TestAdvection2DMask:
    """Verify Advection2D mask=... integrates with upwind_flux."""

    @pytest.fixture
    def grid(self):
        return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)

    @pytest.fixture
    def all_ocean(self):
        return ArakawaCGridMask.from_dimensions(10, 10)

    @pytest.fixture
    def coastal(self):
        import numpy as _np

        h = _np.ones((10, 10), dtype=bool)
        h[:, 4:6] = False
        return ArakawaCGridMask.from_mask(h)

    def test_output_shape(self, grid, all_ocean):
        adv = Advection2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = adv(h, u, v, method="weno5", mask=all_ocean)
        assert result.shape == (grid.Ny, grid.Nx)

    def test_constant_zero_tendency_all_methods(self, grid, all_ocean):
        """Constant field with mask: tendency must be zero in the deep interior."""
        adv = Advection2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        for method in [
            "upwind1",
            "upwind2",
            "upwind3",
            "weno3",
            "weno5",
            "weno7",
            "weno9",
            "minmod",
            "van_leer",
        ]:
            result = adv(h, u, v, method=method, mask=all_ocean)
            np.testing.assert_allclose(
                result[3:-3, 3:-3],
                0.0,
                atol=1e-6,
                err_msg=f"Non-zero tendency for method={method!r} with mask",
            )

    def test_naive_with_mask_falls_through(self, grid, all_ocean):
        """naive + mask must behave identically to naive without mask."""
        adv = Advection2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        with_mask = adv(h, u, v, method="naive", mask=all_ocean)
        without_mask = adv(h, u, v, method="naive")
        np.testing.assert_array_equal(with_mask, without_mask)

    def test_masked_matches_unmasked_all_ocean_weno5_interior(self, grid, all_ocean):
        """On all-ocean mask, masked weno5 must equal unmasked weno5 in the
        deep interior where the full 6-point stencil is available."""
        adv = Advection2D(grid=grid)
        q = jnp.broadcast_to(
            jnp.arange(grid.Ny, dtype=float)[:, None], (grid.Ny, grid.Nx)
        )
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        masked = adv(q, u, v, method="weno5", mask=all_ocean)
        unmasked = adv(q, u, v, method="weno5")
        # Deep interior (depth ≥ 3 from ghost) must match
        np.testing.assert_allclose(masked[3:-3, 3:-3], unmasked[3:-3, 3:-3], atol=1e-6)

    def test_coastal_mask_finite_everywhere(self, grid, coastal):
        """Coastal mask: result must be finite everywhere."""
        adv = Advection2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = adv(h, u, v, method="weno5", mask=coastal)
        assert jnp.all(jnp.isfinite(result)).item()


# ── Mask-aware Advection3D ────────────────────────────────────────────────────


class TestAdvection3DMask:
    """Verify Advection3D mask=... routes through masked Reconstruction3D methods."""

    @pytest.fixture
    def grid(self):
        return CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)

    @pytest.fixture
    def all_ocean(self):
        return ArakawaCGridMask.from_dimensions(8, 8)

    def test_output_shape(self, grid, all_ocean):
        adv = Advection3D(grid=grid)
        h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        u = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        v = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        result = adv(h, u, v, method="weno5", mask=all_ocean)
        assert result.shape == (grid.Nz, grid.Ny, grid.Nx)

    def test_constant_zero_tendency_supported_methods(self, grid, all_ocean):
        """Constant field with mask: tendency must be zero in the deep interior."""
        adv = Advection3D(grid=grid)
        h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        u = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        v = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        for method in ["upwind1", "weno3", "weno5", "minmod", "van_leer"]:
            result = adv(h, u, v, method=method, mask=all_ocean)
            np.testing.assert_allclose(
                result[1:-1, 3:-3, 3:-3],
                0.0,
                atol=1e-6,
                err_msg=f"Non-zero tendency for method={method!r} with mask (3D)",
            )

    def test_weno7_weno9_with_mask_falls_through(self, grid, all_ocean):
        """weno7/weno9 with mask: no masked 3D variant, falls through to unmasked."""
        adv = Advection3D(grid=grid)
        h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        u = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        v = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        for method in ["weno7", "weno9"]:
            with_mask = adv(h, u, v, method=method, mask=all_ocean)
            without_mask = adv(h, u, v, method=method)
            np.testing.assert_array_equal(with_mask, without_mask)

    def test_coastal_mask_finite_everywhere(self):
        """Coastal mask with 3D: result must be finite everywhere."""
        import numpy as _np

        h_mask = _np.ones((8, 8), dtype=bool)
        h_mask[:, 3:5] = False
        coastal = ArakawaCGridMask.from_mask(h_mask)
        grid = CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
        adv = Advection3D(grid=grid)
        h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        u = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        v = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        result = adv(h, u, v, method="weno5", mask=coastal)
        assert jnp.all(jnp.isfinite(result)).item()


class TestAdvection2DRotationStability:
    """Regression test: WENO5/WENOZ5 must remain stable under solid-body rotation.

    Before the right-biased reconstruction fix, WENO5 would blow up
    within ~165 steps for any spatially varying velocity field with
    sign changes (e.g., rotation).
    """

    @pytest.fixture
    def rotation_setup(self):
        """Set up a small solid-body rotation test case."""
        nx = ny = 32
        ng = 4
        Lx = Ly = 1.0
        omega = 2 * jnp.pi
        dx = Lx / nx
        dy = Ly / ny
        grid = CartesianGrid2D(
            Nx=nx + 2 * ng, Ny=ny + 2 * ng, Lx=Lx, Ly=Ly, dx=dx, dy=dy
        )
        # Staggered coordinates
        x_t = (jnp.arange(grid.Nx) - ng + 0.5) * dx
        y_t = (jnp.arange(grid.Ny) - ng + 0.5) * dy
        _Xu, Yu = jnp.meshgrid(x_t + 0.5 * dx, y_t)
        Xv, _Yv = jnp.meshgrid(x_t, y_t + 0.5 * dy)
        u = -omega * (Yu - 0.5)
        v = omega * (Xv - 0.5)
        # Cosine bell IC
        Xt, Yt = jnp.meshgrid(x_t, y_t)
        r = jnp.sqrt((Xt - 0.25) ** 2 + (Yt - 0.5) ** 2)
        q0 = jnp.where(r < 0.15, 0.5 * (1 + jnp.cos(jnp.pi * r / 0.15)), 0.0)

        def pbc(h):
            h = h.at[:ng, :].set(h[-2 * ng : -ng, :])
            h = h.at[-ng:, :].set(h[ng : 2 * ng, :])
            h = h.at[:, :ng].set(h[:, -2 * ng : -ng])
            h = h.at[:, -ng:].set(h[:, ng : 2 * ng])
            return h

        q0 = pbc(q0)
        u_max = float(jnp.max(jnp.abs(u)))
        v_max = float(jnp.max(jnp.abs(v)))
        dt = 0.3 / (u_max / dx + v_max / dy)
        T = 0.5  # half revolution
        nsteps = int(T / dt)
        dt = T / nsteps
        return grid, q0, u, v, pbc, dt, nsteps, ng

    @pytest.mark.parametrize(
        "method,min_peak",
        [
            ("weno3", 0.3),
            ("weno5", 0.5),
            ("wenoz5", 0.5),
            ("weno7", 0.5),
            ("weno9", 0.5),
        ],
    )
    def test_rotation_stays_finite(self, rotation_setup, method, min_peak):
        grid, q0, u, v, pbc, dt, nsteps, ng = rotation_setup
        from finitevolx import Advection2D, rk3_ssp_step

        advect = Advection2D(grid)

        def rhs(q):
            q = pbc(q)
            return advect(q, u, v, method=method)

        rhs_jit = jax.jit(rhs)
        q = q0.copy()
        for _ in range(nsteps):
            q = rk3_ssp_step(q, rhs_jit, dt)
            q = pbc(q)

        assert jnp.all(jnp.isfinite(q)).item(), f"{method} produced NaN/Inf"
        peak = float(jnp.max(q[ng:-ng, ng:-ng]))
        assert peak > min_peak, f"{method} peak eroded to {peak}, expected > {min_peak}"
        assert peak < 1.2, f"{method} peak grew to {peak}, expected < 1.2"


# ── Negative-flow polynomial exactness ──────────────────────────────────────


class TestNegativeFlowExactness:
    """Each scheme must give zero tendency for constant fields under negative flow.

    These tests use u = -1 (negative velocity) and verify that, in both 1-D and
    2-D, the discrete advection tendency vanishes in the interior region when
    the advected field h is spatially constant.
    """

    @pytest.mark.parametrize(
        "method",
        ["upwind1", "weno3", "weno5", "weno7", "weno9", "minmod", "van_leer"],
    )
    def test_constant_negative_flow_zero_tendency(self, method):
        grid = CartesianGrid1D.from_interior(24, 1.0)
        adv = Advection1D(grid=grid)
        h = jnp.ones(grid.Nx)
        u = -jnp.ones(grid.Nx)
        result = adv(h, u, method=method)
        # Wide stencils (weno9) accumulate ~1e-6 error in float32
        np.testing.assert_allclose(result[4:-4], 0.0, atol=1e-5)

    @pytest.mark.parametrize(
        "method",
        ["upwind1", "weno3", "weno5", "weno7", "weno9", "minmod", "van_leer"],
    )
    def test_constant_negative_flow_2d(self, method):
        grid = CartesianGrid2D.from_interior(16, 16, 1.0, 1.0)
        adv = Advection2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        u = -jnp.ones((grid.Ny, grid.Nx))
        v = -jnp.ones((grid.Ny, grid.Nx))
        result = adv(h, u, v, method=method)
        np.testing.assert_allclose(result[4:-4, 4:-4], 0.0, atol=1e-5)


# ── 1-D convergence rate under negative flow ───────────────────────────────


class TestNegativeFlowConvergence:
    """Verify convergence order under negative flow for each WENO scheme.

    A smooth sinusoidal profile advected with u = -1 for one period must
    converge at the expected order as the grid is refined.
    """

    @staticmethod
    def _run_1d_advection(n_interior, method, n_periods=1):
        """Advect sin(2π x) with u=-1 for *n_periods* on a periodic domain."""
        from finitevolx import Advection1D, rk3_ssp_step

        ng = 4
        Lx = 1.0
        dx = Lx / n_interior
        grid = CartesianGrid1D(Nx=n_interior + 2 * ng, Lx=Lx, dx=dx)

        x = (jnp.arange(grid.Nx) - ng + 0.5) * dx
        q0 = jnp.sin(2 * jnp.pi * x)

        u = -jnp.ones(grid.Nx)

        def pbc(h):
            h = h.at[:ng].set(h[-2 * ng : -ng])
            h = h.at[-ng:].set(h[ng : 2 * ng])
            return h

        q0 = pbc(q0)
        u_max = 1.0
        dt = 0.3 * dx / u_max
        T = n_periods * Lx
        nsteps = int(T / dt)
        dt = T / nsteps

        advect = Advection1D(grid)

        def rhs(q):
            q = pbc(q)
            return advect(q, u, method=method)

        rhs_jit = jax.jit(rhs)
        q = q0.copy()
        for _ in range(nsteps):
            q = rk3_ssp_step(q, rhs_jit, dt)
            q = pbc(q)

        error = float(jnp.max(jnp.abs(q[ng:-ng] - q0[ng:-ng])))
        return error

    @pytest.mark.parametrize(
        "method,min_order",
        [
            ("upwind1", 0.8),
            ("weno3", 1.2),
            ("weno5", 2.0),
            ("weno7", 2.5),
            ("weno9", 2.5),
        ],
    )
    def test_convergence_rate_negative_flow(self, method, min_order):
        n1, n2 = 32, 128
        e1 = self._run_1d_advection(n1, method)
        e2 = self._run_1d_advection(n2, method)
        ratio = n2 / n1
        order = jnp.log(e1 / e2) / jnp.log(ratio)
        assert float(order) > min_order, (
            f"{method}: convergence order {float(order):.2f} < {min_order}"
        )


# ── Conservation ────────────────────────────────────────────────────────────


class TestAdvectionConservation:
    """Total mass (sum * dx) must be conserved to float32 accumulation tolerance."""

    @pytest.mark.parametrize(
        "method",
        ["upwind1", "weno3", "weno5", "weno7", "weno9", "minmod", "van_leer"],
    )
    def test_1d_conservation(self, method):
        from finitevolx import Advection1D, rk3_ssp_step

        ng = 4
        n = 64
        Lx = 1.0
        dx = Lx / n
        grid = CartesianGrid1D(Nx=n + 2 * ng, Lx=Lx, dx=dx)

        x = (jnp.arange(grid.Nx) - ng + 0.5) * dx
        r = jnp.abs(x - 0.5)
        q0 = jnp.where(r < 0.15, 0.5 * (1 + jnp.cos(jnp.pi * r / 0.15)), 0.0)
        u = jnp.ones(grid.Nx)

        def pbc(h):
            h = h.at[:ng].set(h[-2 * ng : -ng])
            h = h.at[-ng:].set(h[ng : 2 * ng])
            return h

        q0 = pbc(q0)
        dt = 0.3 * dx
        advect = Advection1D(grid)

        def rhs(q):
            q = pbc(q)
            return advect(q, u, method=method)

        rhs_jit = jax.jit(rhs)
        q = q0.copy()
        mass0 = float(jnp.sum(q[ng:-ng]) * dx)
        for _ in range(50):
            q = rk3_ssp_step(q, rhs_jit, dt)
            q = pbc(q)

        mass = float(jnp.sum(q[ng:-ng]) * dx)
        np.testing.assert_allclose(mass, mass0, rtol=1e-5)
