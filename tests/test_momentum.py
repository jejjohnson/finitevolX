"""Tests for MomentumAdvection2D and MomentumAdvection3D."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.diffusion.momentum import MomentumAdvection2D, MomentumAdvection3D
from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D


@pytest.fixture
def grid2d():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


class TestMomentumAdvection2D:
    def test_output_shapes(self, grid2d):
        madv = MomentumAdvection2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        du, dv = madv(u, v)
        assert du.shape == (grid2d.Ny, grid2d.Nx)
        assert dv.shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_zero(self, grid2d):
        madv = MomentumAdvection2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        du, dv = madv(u, v)
        # Ghost rows/cols must be zero
        np.testing.assert_allclose(du[0, :], 0.0)
        np.testing.assert_allclose(du[-1, :], 0.0)
        np.testing.assert_allclose(du[:, 0], 0.0)
        np.testing.assert_allclose(du[:, -1], 0.0)
        np.testing.assert_allclose(dv[0, :], 0.0)
        np.testing.assert_allclose(dv[-1, :], 0.0)
        np.testing.assert_allclose(dv[:, 0], 0.0)
        np.testing.assert_allclose(dv[:, -1], 0.0)

    def test_uniform_flow_zero_tendency(self, grid2d):
        """Uniform velocity (u=c, v=0): ζ=0 and ∇K=0, so tendencies are zero.

        Checks [2:-2, 2:-2] to avoid the innermost ghost-adjacent ring where
        dK/dx reads the zero ghost T-cell and produces a spurious gradient.
        """
        madv = MomentumAdvection2D(grid=grid2d)
        c = 2.5
        u = c * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        for scheme in ("energy", "enstrophy", "al"):
            du, dv = madv(u, v, scheme=scheme)
            np.testing.assert_allclose(du[2:-2, 2:-2], 0.0, atol=1e-10)
            np.testing.assert_allclose(dv[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_all_schemes_run(self, grid2d):
        madv = MomentumAdvection2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        for scheme in ("energy", "enstrophy", "al"):
            du, dv = madv(u, v, scheme=scheme)
            assert du.shape == (grid2d.Ny, grid2d.Nx)
            assert dv.shape == (grid2d.Ny, grid2d.Nx)

    def test_unknown_scheme_raises(self, grid2d):
        madv = MomentumAdvection2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        with pytest.raises(ValueError, match="Unknown scheme"):
            madv(u, v, scheme="invalid")

    def test_no_nan_output(self, grid2d):
        """Tendencies must not contain NaN for well-defined inputs."""
        madv = MomentumAdvection2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))
        for scheme in ("energy", "enstrophy", "al"):
            du, dv = madv(u, v, scheme=scheme)
            assert jnp.all(jnp.isfinite(du)), f"{scheme}: du contains NaN/Inf"
            assert jnp.all(jnp.isfinite(dv)), f"{scheme}: dv contains NaN/Inf"

    def test_al_is_blend_of_energy_and_enstrophy(self, grid2d):
        """AL scheme = ⅓ energy + ⅔ enstrophy."""
        madv = MomentumAdvection2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))
        du_e, dv_e = madv(u, v, scheme="energy")
        du_s, dv_s = madv(u, v, scheme="enstrophy")
        du_al, dv_al = madv(u, v, scheme="al")
        expected_du = (1.0 / 3.0) * du_e + (2.0 / 3.0) * du_s
        expected_dv = (1.0 / 3.0) * dv_e + (2.0 / 3.0) * dv_s
        np.testing.assert_allclose(du_al, expected_du, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dv_al, expected_dv, rtol=1e-5, atol=1e-6)

    def test_energy_conservation_cross_term(self, grid2d):
        """Energy-scheme cross terms cancel: Σ u*(ζ·v) = Σ v*(ζ·u) in the interior.

        For the Sadourny E-scheme, the vorticity-flux contribution to the
        energy tendency is:
            Σ_U u * (ζ_on_u * v_on_u) - Σ_V v * (ζ_on_v * u_on_v)
        This must be zero (up to floating point) for any velocity field,
        verifying that the scheme conserves kinetic energy from the cross term.
        """
        from finitevolx._src.operators.difference import Difference2D
        from finitevolx._src.operators.interpolation import Interpolation2D

        diff = Difference2D(grid=grid2d)
        interp = Interpolation2D(grid=grid2d)

        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        # Solid-body rotation: u = -y, v = x  →  constant zeta = 2
        u = jnp.broadcast_to(-y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))

        zeta = diff.curl(u, v)
        zeta_on_u = interp.X_to_U(zeta)
        zeta_on_v = interp.X_to_V(zeta)
        v_on_u = interp.V_to_U(v)
        u_on_v = interp.U_to_V(u)

        # vorticity-flux cross terms at strict interior points (avoid ghost-adjacent ring)
        cross_u = jnp.sum(u[2:-2, 2:-2] * (zeta_on_u[2:-2, 2:-2] * v_on_u[2:-2, 2:-2]))
        cross_v = jnp.sum(v[2:-2, 2:-2] * (zeta_on_v[2:-2, 2:-2] * u_on_v[2:-2, 2:-2]))
        # For non-periodic BCs the exact cancellation holds only approximately;
        # check that the imbalance is small relative to the signal.
        np.testing.assert_allclose(cross_u, cross_v, rtol=1e-6)

    def test_zero_velocity_zero_tendency(self, grid2d):
        """Zero velocity gives zero advection tendency."""
        madv = MomentumAdvection2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        for scheme in ("energy", "enstrophy", "al"):
            du, dv = madv(u, v, scheme=scheme)
            np.testing.assert_allclose(du, 0.0, atol=1e-15)
            np.testing.assert_allclose(dv, 0.0, atol=1e-15)

    def test_nonzero_tendency_for_nontrivial_field(self, grid2d):
        """Non-trivial velocity field should produce non-zero tendency."""
        madv = MomentumAdvection2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        # Solid-body rotation has non-zero ζ and non-zero ∇K.
        u = jnp.broadcast_to(-y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))
        du, _ = madv(u, v)
        # At least some interior cells must be non-zero.
        assert jnp.any(jnp.abs(du[2:-2, 2:-2]) > 0), "du_adv is unexpectedly zero"


class TestMomentumAdvection3D:
    def test_output_shapes(self, grid3d):
        madv = MomentumAdvection3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        du, dv = madv(u, v)
        assert du.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert dv.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_ghost_ring_zero(self, grid3d):
        madv = MomentumAdvection3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        du, dv = madv(u, v)
        np.testing.assert_allclose(du[0, :, :], 0.0)
        np.testing.assert_allclose(du[-1, :, :], 0.0)
        np.testing.assert_allclose(du[:, 0, :], 0.0)
        np.testing.assert_allclose(du[:, -1, :], 0.0)
        np.testing.assert_allclose(du[:, :, 0], 0.0)
        np.testing.assert_allclose(du[:, :, -1], 0.0)
        np.testing.assert_allclose(dv[0, :, :], 0.0)
        np.testing.assert_allclose(dv[-1, :, :], 0.0)
        np.testing.assert_allclose(dv[:, 0, :], 0.0)
        np.testing.assert_allclose(dv[:, -1, :], 0.0)
        np.testing.assert_allclose(dv[:, :, 0], 0.0)
        np.testing.assert_allclose(dv[:, :, -1], 0.0)

    def test_uniform_flow_zero_tendency(self, grid3d):
        """Uniform flow gives zero tendency in the strict interior [2:-2, 2:-2, 2:-2].

        Checks [2:-2, 2:-2, 2:-2] to avoid the ghost-adjacent ring where
        dK/dx reads the zero ghost T-cell and produces a spurious gradient.
        """
        madv = MomentumAdvection3D(grid=grid3d)
        c = 1.5
        u = c * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for scheme in ("energy", "enstrophy", "al"):
            du, dv = madv(u, v, scheme=scheme)
            np.testing.assert_allclose(du[1:-1, 2:-2, 2:-2], 0.0, atol=1e-10)
            np.testing.assert_allclose(dv[1:-1, 2:-2, 2:-2], 0.0, atol=1e-10)

    def test_all_schemes_run(self, grid3d):
        madv = MomentumAdvection3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for scheme in ("energy", "enstrophy", "al"):
            du, dv = madv(u, v, scheme=scheme)
            assert du.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
            assert dv.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_unknown_scheme_raises(self, grid3d):
        madv = MomentumAdvection3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        with pytest.raises(ValueError, match="Unknown scheme"):
            madv(u, v, scheme="bad")

    def test_no_nan_output(self, grid3d):
        madv = MomentumAdvection3D(grid=grid3d)
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        y = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        u2d = jnp.broadcast_to(-y[:, None], (grid3d.Ny, grid3d.Nx))
        v2d = jnp.broadcast_to(x, (grid3d.Ny, grid3d.Nx))
        u = jnp.broadcast_to(u2d[None], (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.broadcast_to(v2d[None], (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for scheme in ("energy", "enstrophy", "al"):
            du, dv = madv(u, v, scheme=scheme)
            assert jnp.all(jnp.isfinite(du)), f"3D {scheme}: du has NaN/Inf"
            assert jnp.all(jnp.isfinite(dv)), f"3D {scheme}: dv has NaN/Inf"

    def test_zero_velocity_zero_tendency(self, grid3d):
        madv = MomentumAdvection3D(grid=grid3d)
        u = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for scheme in ("energy", "enstrophy", "al"):
            du, dv = madv(u, v, scheme=scheme)
            np.testing.assert_allclose(du, 0.0, atol=1e-15)
            np.testing.assert_allclose(dv, 0.0, atol=1e-15)

    def test_al_is_blend_of_energy_and_enstrophy(self, grid3d):
        """AL scheme = ⅓ energy + ⅔ enstrophy (3D)."""
        madv = MomentumAdvection3D(grid=grid3d)
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        y = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        u2d = jnp.broadcast_to(-y[:, None], (grid3d.Ny, grid3d.Nx))
        v2d = jnp.broadcast_to(x, (grid3d.Ny, grid3d.Nx))
        u = jnp.broadcast_to(u2d[None], (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.broadcast_to(v2d[None], (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        du_e, dv_e = madv(u, v, scheme="energy")
        du_s, dv_s = madv(u, v, scheme="enstrophy")
        du_al, dv_al = madv(u, v, scheme="al")
        expected_du = (1.0 / 3.0) * du_e + (2.0 / 3.0) * du_s
        expected_dv = (1.0 / 3.0) * dv_e + (2.0 / 3.0) * dv_s
        np.testing.assert_allclose(du_al, expected_du, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dv_al, expected_dv, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Discrete conservation of the vortex-force schemes (#283)
# ---------------------------------------------------------------------------


def _compact_nondivergent_flow(n: int = 36, seed: int = 0):
    """Random non-divergent (u, v) that vanish within 5 cells of every wall.

    psi lives at X-points; u = -d psi/dy at U-points, v = d psi/dx at
    V-points, so the C-grid divergence is zero to roundoff.  The compact
    support keeps every term of the budgets inside MomentumAdvection2D's
    [2:-2] write region, so the sums below are exact summations by parts.
    """
    grid = CartesianGrid2D.from_interior(n - 2, n - 2, 1.0, 1.0)
    rng = np.random.default_rng(seed)
    window = np.zeros((n, n))
    window[6:-6, 6:-6] = 1.0
    psi = jnp.asarray(rng.standard_normal((n, n)) * window)
    u = (
        jnp.zeros((n, n))
        .at[1:-1, 1:-1]
        .set(-(psi[1:-1, 1:-1] - psi[:-2, 1:-1]) / grid.dy)
    )
    v = (
        jnp.zeros((n, n))
        .at[1:-1, 1:-1]
        .set((psi[1:-1, 1:-1] - psi[1:-1, :-2]) / grid.dx)
    )
    return grid, u, v


def _budgets(grid, u, v, scheme):
    """(dKE, dZ, scale) for the momentum-advection tendency."""
    from finitevolx._src.operators.difference import Difference2D

    madv = MomentumAdvection2D(grid=grid)
    du, dv = madv(u, v, scheme=scheme)
    d2 = Difference2D(grid=grid)
    zeta = d2.curl(u, v)
    dzeta = d2.curl(du, dv)
    dke = float(jnp.sum(u * du + v * dv))
    dz = float(jnp.sum(zeta * dzeta))
    scale_ke = float(jnp.sum(jnp.abs(u * du) + jnp.abs(v * dv)))
    scale_z = float(jnp.sum(jnp.abs(zeta * dzeta)))
    return dke, dz, scale_ke, scale_z


class TestVortexForceConservation:
    """scheme='energy' conserves KE, scheme='enstrophy' conserves enstrophy.

    Non-uniform vorticity matters: for solid-body rotation the two Sadourny
    forms coincide, which is how the swapped labels went unnoticed (#283).
    The 1e-12 tolerance is relative to the sum of |terms| (float64 roundoff
    over ~10^3 terms); the non-conserved invariant is O(1) relative.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_energy_scheme_conserves_kinetic_energy(self, seed):
        grid, u, v = _compact_nondivergent_flow(seed=seed)
        dke, dz, s_ke, s_z = _budgets(grid, u, v, "energy")
        assert abs(dke) < 1e-12 * s_ke
        assert abs(dz) > 1e-6 * s_z  # and does not conserve enstrophy

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_enstrophy_scheme_conserves_enstrophy(self, seed):
        grid, u, v = _compact_nondivergent_flow(seed=seed)
        dke, dz, s_ke, s_z = _budgets(grid, u, v, "enstrophy")
        assert abs(dz) < 1e-12 * s_z
        assert abs(dke) > 1e-6 * s_ke  # and does not conserve energy

    def test_flow_is_non_divergent(self):
        from finitevolx._src.operators.difference import Difference2D

        grid, u, v = _compact_nondivergent_flow()
        div = Difference2D(grid=grid).divergence(u, v)
        assert float(jnp.abs(div).max()) < 1e-10 * float(jnp.abs(u).max()) / grid.dx
