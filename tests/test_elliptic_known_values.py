"""Tests for ``known_values`` / ``known_mask`` on the elliptic convenience wrappers.

The core check is a manufactured solution: pick an exact field that does
*not* vanish on the boundary, build its discrete right-hand side with the
same 5-point operator the solvers invert, then solve with the exact field as
``known_values`` and recover it on every wet cell.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx as fvx

jax.config.update("jax_enable_x64", True)

NY, NX = 24, 28
DX, DY = 1.0 / (NX - 2), 1.0 / (NY - 2)
# CG runs at rtol=1e-8 by default, so solutions agree to a few 1e-8.
CG_TOL = 1e-6


def _basin_mask() -> jnp.ndarray:
    """Standard rectangular basin: dry ghost ring, wet interior."""
    # mask[j, i] = 1 for 1 <= j <= NY-2, 1 <= i <= NX-2
    return jnp.zeros((NY, NX)).at[1:-1, 1:-1].set(1.0)


def _island_mask() -> jnp.ndarray:
    """Basin with a 3x3 island in the middle."""
    cy, cx = NY // 2, NX // 2
    # mask[j, i] = 0 for cy <= j <= cy+2, cx <= i <= cx+2
    return _basin_mask().at[cy : cy + 3, cx : cx + 3].set(0.0)


def _exact() -> jnp.ndarray:
    """psi = sin(pi x) sin(pi y) + 0.1 sin(2 pi y): non-zero on the walls."""
    # y[j, 0] = j / (NY-1),  x[0, i] = i / (NX-1)  (broadcast to [NY, NX])
    y = jnp.linspace(0.0, 1.0, NY)[:, None]
    x = jnp.linspace(0.0, 1.0, NX)[None, :]
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + 0.1 * jnp.sin(2 * jnp.pi * y)


def _rhs(psi: jnp.ndarray, mask: jnp.ndarray, lambda_: float) -> jnp.ndarray:
    """Discrete RHS f = (A - lambda) psi on the wet cells."""
    return fvx.masked_laplacian(psi, mask, DX, DY, lambda_=lambda_)


def _capacitance(mask, lambda_):
    """Capacitance solver on the wet mask.

    It holds its own inner ring at zero, so its unknowns are exactly the
    lifted solve domain.  The DST base keeps lambda_ = 0 non-singular.
    """
    return fvx.build_capacitance_solver(
        np.asarray(mask > 0.5), DX, DY, lambda_=lambda_, base_bc="dst"
    )


def _wet_max_err(psi, exact, mask) -> float:
    return float(jnp.max(jnp.abs((psi - exact) * mask)))


class TestManufacturedSolution:
    @pytest.mark.parametrize("lambda_", [0.0, 4.0])
    def test_cg_basin(self, lambda_):
        mask, exact = _basin_mask(), _exact()
        psi = fvx.streamfunction_from_vorticity(
            _rhs(exact, mask, lambda_),
            DX,
            DY,
            lambda_=lambda_,
            method="cg",
            mask=mask,
            known_values=exact,
        )
        assert _wet_max_err(psi, exact, mask) < CG_TOL

    @pytest.mark.parametrize("lambda_", [0.0, 4.0])
    def test_spectral_basin_exact(self, lambda_):
        """The spectral path solves the rectangular basin directly."""
        mask, exact = _basin_mask(), _exact()
        psi = fvx.streamfunction_from_vorticity(
            _rhs(exact, mask, lambda_),
            DX,
            DY,
            bc="dst",
            lambda_=lambda_,
            known_values=exact,
        )
        assert _wet_max_err(psi, exact, mask) < 1e-10

    @pytest.mark.parametrize("lambda_", [0.0, 4.0])
    def test_capacitance_basin(self, lambda_):
        mask, exact = _basin_mask(), _exact()
        psi = fvx.streamfunction_from_vorticity(
            _rhs(exact, mask, lambda_),
            DX,
            DY,
            lambda_=lambda_,
            method="capacitance",
            mask=mask,
            capacitance_solver=_capacitance(mask, lambda_),
            known_values=exact,
        )
        assert _wet_max_err(psi, exact, mask) < 1e-8

    def test_capacitance_island(self):
        mask, exact = _island_mask(), _exact()
        psi = fvx.streamfunction_from_vorticity(
            _rhs(exact, mask, 4.0),
            DX,
            DY,
            lambda_=4.0,
            method="capacitance",
            mask=mask,
            capacitance_solver=_capacitance(mask, 4.0),
            known_values=exact,
        )
        assert _wet_max_err(psi, exact, mask) < 1e-8

    def test_multigrid_preconditioned_cg_island(self):
        """Multigrid enters as the CG preconditioner on the effective domain."""
        mask, exact = _island_mask(), _exact()
        domain = fvx.SolveDomain(mask)
        mg = fvx.build_multigrid_solver(
            np.asarray(domain.effective_mask, dtype=float), DX, DY, lambda_=4.0
        )
        psi = fvx.streamfunction_from_vorticity(
            _rhs(exact, mask, 4.0),
            DX,
            DY,
            lambda_=4.0,
            method="cg",
            mask=mask,
            known_values=exact,
            preconditioner=fvx.make_multigrid_preconditioner(mg),
        )
        assert _wet_max_err(psi, exact, mask) < CG_TOL

    def test_cg_island(self):
        mask, exact = _island_mask(), _exact()
        psi = fvx.streamfunction_from_vorticity(
            _rhs(exact, mask, 4.0),
            DX,
            DY,
            lambda_=4.0,
            method="cg",
            mask=mask,
            known_values=exact,
        )
        assert _wet_max_err(psi, exact, mask) < CG_TOL

    def test_pressure_cg(self):
        mask, exact = _basin_mask(), _exact()
        p = fvx.pressure_from_divergence(
            _rhs(exact, mask, 0.0), DX, DY, method="cg", mask=mask, known_values=exact
        )
        assert _wet_max_err(p, exact, mask) < CG_TOL

    def test_pressure_spectral_dst(self):
        mask, exact = _basin_mask(), _exact()
        p = fvx.pressure_from_divergence(
            _rhs(exact, mask, 0.0), DX, DY, bc="dst", known_values=exact
        )
        assert _wet_max_err(p, exact, mask) < 1e-10


class TestConsistency:
    def test_spectral_matches_cg_on_basin(self):
        mask = _basin_mask()
        rng = np.random.default_rng(0)
        rhs = jnp.asarray(rng.standard_normal((NY, NX))) * mask
        g = jnp.asarray(rng.standard_normal((NY, NX)))
        kw = {"lambda_": 4.0, "known_values": g}
        psi_sp = fvx.streamfunction_from_vorticity(rhs, DX, DY, **kw)
        psi_cg = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, method="cg", mask=mask, **kw
        )
        np.testing.assert_allclose(psi_sp, psi_cg, atol=CG_TOL)

    def test_zero_known_values_is_homogeneous_on_effective_domain(self):
        mask = _basin_mask()
        rhs = _rhs(_exact(), mask, 0.0)
        psi = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, method="cg", mask=mask, known_values=jnp.zeros((NY, NX))
        )
        eff = fvx.SolveDomain(mask).effective_mask.astype(rhs.dtype)
        psi_hom = fvx.streamfunction_from_vorticity(
            rhs * eff, DX, DY, method="cg", mask=eff
        )
        np.testing.assert_allclose(psi, psi_hom, atol=CG_TOL)

    def test_known_values_ignored_off_known_cells(self):
        """Only boundary-ring values matter; interior garbage is ignored."""
        mask, exact = _basin_mask(), _exact()
        rhs = _rhs(exact, mask, 0.0)
        ring = fvx.boundary_ring(mask)
        noisy = jnp.where(ring, exact, 123.0)
        psi = fvx.streamfunction_from_vorticity(rhs, DX, DY, known_values=noisy)
        psi_ref = fvx.streamfunction_from_vorticity(rhs, DX, DY, known_values=exact)
        np.testing.assert_allclose(psi, psi_ref, atol=1e-12)


class TestKnownMask:
    def test_sparse_observations_pinned(self):
        mask = _basin_mask()
        rng = np.random.default_rng(1)
        rhs = jnp.asarray(rng.standard_normal((NY, NX))) * mask
        obs = jnp.zeros((NY, NX), dtype=bool).at[8, 10].set(True).at[15, 20].set(True)
        values = jnp.zeros((NY, NX)).at[8, 10].set(0.3).at[15, 20].set(-0.2)
        psi = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, method="cg", mask=mask, known_values=values, known_mask=obs
        )
        np.testing.assert_allclose(psi[8, 10], 0.3, atol=1e-12)
        np.testing.assert_allclose(psi[15, 20], -0.2, atol=1e-12)
        # The boundary ring is still pinned to known_values (zero here).
        np.testing.assert_allclose(psi[fvx.boundary_ring(mask)], 0.0, atol=1e-12)

    def test_sparse_observations_manufactured(self):
        mask, exact = _basin_mask(), _exact()
        obs = jnp.zeros((NY, NX), dtype=bool).at[10, 12].set(True)
        psi = fvx.streamfunction_from_vorticity(
            _rhs(exact, mask, 0.0),
            DX,
            DY,
            method="cg",
            mask=mask,
            known_values=exact,
            known_mask=obs,
        )
        assert _wet_max_err(psi, exact, mask) < CG_TOL


class TestPVInversion:
    @pytest.mark.parametrize("method", ["cg", "spectral"])
    def test_array_lambda_per_layer_known_values(self, method):
        mask = _basin_mask()
        lambdas = jnp.array([0.0, 4.0, 25.0])
        exact = jnp.stack([(k + 1.0) * _exact() for k in range(3)])
        pv = jnp.stack([_rhs(exact[k], mask, float(lambdas[k])) for k in range(3)])
        kw = {"method": method, "mask": mask} if method == "cg" else {}
        psi = fvx.pv_inversion(pv, DX, DY, lambdas, known_values=exact, **kw)
        assert psi.shape == pv.shape
        tol = CG_TOL if method == "cg" else 1e-10
        for k in range(3):
            assert _wet_max_err(psi[k], exact[k], mask) < tol

    def test_array_lambda_broadcast_known_values(self):
        """A single (Ny, Nx) field is shared by every layer."""
        mask, exact = _basin_mask(), _exact()
        lambdas = jnp.array([1.0, 9.0])
        pv = jnp.stack([_rhs(exact, mask, float(lam)) for lam in lambdas])
        psi = fvx.pv_inversion(pv, DX, DY, lambdas, known_values=exact)
        for k in range(2):
            assert _wet_max_err(psi[k], exact, mask) < 1e-10

    def test_scalar_lambda_batched_matches_single(self):
        mask = _basin_mask()
        rng = np.random.default_rng(2)
        pv = jnp.asarray(rng.standard_normal((2, NY, NX))) * mask
        g = jnp.asarray(rng.standard_normal((2, NY, NX)))
        psi = fvx.pv_inversion(pv, DX, DY, 4.0, method="cg", mask=mask, known_values=g)
        for b in range(2):
            ref = fvx.streamfunction_from_vorticity(
                pv[b], DX, DY, lambda_=4.0, method="cg", mask=mask, known_values=g[b]
            )
            np.testing.assert_allclose(psi[b], ref, atol=CG_TOL)

    def test_batched_array_lambda(self):
        """Leading batch dims on top of the layer axis."""
        mask, exact = _basin_mask(), _exact()
        lambdas = jnp.array([1.0, 9.0])
        layer = jnp.stack([_rhs(exact, mask, float(lam)) for lam in lambdas])
        pv = jnp.stack([layer, layer])
        psi = fvx.pv_inversion(pv, DX, DY, lambdas, known_values=exact)
        assert psi.shape == pv.shape
        assert _wet_max_err(psi[1, 1], exact, mask) < 1e-10


class TestJIT:
    def test_jit_time_varying_known_values(self):
        mask = _basin_mask()
        rhs = _rhs(_exact(), mask, 0.0)

        @jax.jit
        def solve(g):
            return fvx.streamfunction_from_vorticity(
                rhs, DX, DY, method="cg", mask=mask, known_values=g
            )

        for scale in (1.0, 2.0):
            g = scale * _exact()
            psi = solve(g)
            ring = fvx.boundary_ring(mask)
            np.testing.assert_allclose(psi[ring], g[ring], atol=1e-12)

    def test_jit_spectral(self):
        exact = _exact()
        rhs = _rhs(exact, _basin_mask(), 4.0)
        solve = jax.jit(
            lambda g: fvx.streamfunction_from_vorticity(
                rhs, DX, DY, lambda_=4.0, known_values=g
            )
        )
        assert _wet_max_err(solve(exact), exact, _basin_mask()) < 1e-10


class TestErrors:
    def test_spectral_requires_dst(self):
        with pytest.raises(ValueError, match="bc='dst'"):
            fvx.pressure_from_divergence(
                jnp.zeros((NY, NX)), DX, DY, known_values=jnp.zeros((NY, NX))
            )

    def test_spectral_rejects_mask(self):
        with pytest.raises(ValueError, match="does not accept a mask"):
            fvx.streamfunction_from_vorticity(
                jnp.zeros((NY, NX)),
                DX,
                DY,
                mask=_island_mask(),
                known_values=jnp.zeros((NY, NX)),
            )

    def test_spectral_rejects_known_mask(self):
        with pytest.raises(ValueError, match="known_mask"):
            fvx.streamfunction_from_vorticity(
                jnp.zeros((NY, NX)),
                DX,
                DY,
                known_values=jnp.zeros((NY, NX)),
                known_mask=jnp.zeros((NY, NX), dtype=bool),
            )

    def test_capacitance_rejects_known_mask(self):
        mask = _basin_mask()
        with pytest.raises(ValueError, match="does not support known_mask"):
            fvx.streamfunction_from_vorticity(
                jnp.zeros((NY, NX)),
                DX,
                DY,
                method="capacitance",
                mask=mask,
                capacitance_solver=_capacitance(mask, 1.0),
                known_values=jnp.zeros((NY, NX)),
                known_mask=jnp.zeros((NY, NX), dtype=bool),
            )

    def test_mask_based_requires_mask(self):
        with pytest.raises(ValueError, match="requires a mask"):
            fvx.streamfunction_from_vorticity(
                jnp.zeros((NY, NX)),
                DX,
                DY,
                method="cg",
                known_values=jnp.zeros((NY, NX)),
            )

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method must be"):
            fvx.streamfunction_from_vorticity(
                jnp.zeros((NY, NX)),
                DX,
                DY,
                method="bogus",
                mask=_basin_mask(),
                known_values=jnp.zeros((NY, NX)),
            )
