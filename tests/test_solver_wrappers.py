"""Tests for streamfunction, pressure, and PV-inversion convenience wrappers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid import CartesianGrid2D
from finitevolx._src.mask import Mask2D
from finitevolx._src.solvers.elliptic import (
    build_capacitance_solver,
    dst1_eigenvalues,
    make_spectral_preconditioner,
    masked_laplacian,
    pressure_from_divergence,
    pv_inversion,
    solve_cg,
    solve_helmholtz_dct1,
    solve_poisson_dct,
    streamfunction_from_vorticity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dirichlet_grid():
    """Full [Ny, Nx] C-grid array; ψ's wall nodes are X index 0 and N-2."""
    Ny, Nx = 10, 12
    dx = 1.0 / (Nx - 2)
    dy = 1.0 / (Ny - 2)
    return Ny, Nx, dx, dy


def _x_eigenmode(Ny, Nx, dx, dy):
    """Gravest discrete Dirichlet mode on the X points, and its eigenvalue.

    psi[j, i] = sin(pi j / (Ny-2)) sin(pi i / (Nx-2)) vanishes on the wall
    nodes 0 and N-2; the unknowns are the N-3 nodes [1:-2] (DST-I), so the
    eigenvalue is dst1_eigenvalues(N-3, d)[0] per axis.  Index N-1 is outside
    the domain and the solution is zero there.
    """
    j = jnp.arange(Ny)[:, None]
    i = jnp.arange(Nx)[None, :]
    psi = jnp.sin(jnp.pi * j / (Ny - 2)) * jnp.sin(jnp.pi * i / (Nx - 2))
    # psi[Ny-1, :] = psi[:, Nx-1] = 0  (outside the domain)
    psi = psi.at[-1, :].set(0.0).at[:, -1].set(0.0)
    eig = dst1_eigenvalues(Nx - 3, dx)[0] + dst1_eigenvalues(Ny - 3, dy)[0]
    return psi, eig


@pytest.fixture
def neumann_grid():
    Ny, Nx = 8, 10
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    return Ny, Nx, dx, dy


def _inner_boundary_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute inner-boundary (j, i) indices from a boolean mask."""
    from scipy.ndimage import binary_dilation

    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    dilated = binary_dilation(~np.asarray(mask, dtype=bool), structure=struct)
    inner_boundary = np.asarray(mask, dtype=bool) & dilated
    return np.where(inner_boundary)


@pytest.fixture
def rect_mask():
    """Rectangular mask with 1-cell land border for capacitance / CG tests."""
    Ny, Nx = 10, 12
    mask = np.ones((Ny, Nx), dtype=bool)
    mask[:1, :] = mask[-1:, :] = False
    mask[:, :1] = mask[:, -1:] = False
    return mask


@pytest.fixture
def cgrid_mask(rect_mask):
    """Mask2D built from the rectangular mask."""
    return Mask2D.from_mask(rect_mask)


# ---------------------------------------------------------------------------
# streamfunction_from_vorticity — spectral method
# ---------------------------------------------------------------------------


class TestStreamfunctionSpectral:
    def test_dst_eigenmode_on_x_wall(self, dirichlet_grid):
        """The discrete X-point eigenmode is recovered, wall nodes held at 0."""
        Ny, Nx, dx, dy = dirichlet_grid
        psi_exact, eig = _x_eigenmode(Ny, Nx, dx, dy)
        zeta = eig * psi_exact

        psi = streamfunction_from_vorticity(zeta, dx, dy, bc="dst", method="spectral")
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)
        # The wall nodes 0 and N-2 (and the outside index N-1) are zero.
        for k in (0, -2, -1):
            np.testing.assert_array_equal(np.array(psi[k, :]), 0.0)
            np.testing.assert_array_equal(np.array(psi[:, k]), 0.0)

    def test_roundtrip_dct(self, neumann_grid):
        """Neumann ψ: DCT-I on the X nodes [0:-1, 0:-1], reflected about 0 and N-2."""
        Ny, Nx, dx, dy = neumann_grid
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        expected = (
            jnp.zeros((Ny, Nx))
            .at[:-1, :-1]
            .set(solve_helmholtz_dct1(rhs[:-1, :-1], dx, dy, 0.0))
        )
        result = streamfunction_from_vorticity(rhs, dx, dy, bc="dct", method="spectral")
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-12)

    def test_roundtrip_fft(self):
        """Periodic-BC spectral wrapper."""
        Ny, Nx = 8, 10
        dx = 2.0 * np.pi / Nx
        dy = 2.0 * np.pi / Ny
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        result = streamfunction_from_vorticity(rhs, dx, dy, bc="fft", method="spectral")
        assert result.shape == (Ny, Nx)

    def test_default_method_is_spectral(self, dirichlet_grid):
        """Default method is spectral (no mask or solver needed)."""
        Ny, Nx, dx, dy = dirichlet_grid
        rhs = jnp.ones((Ny, Nx))
        result = streamfunction_from_vorticity(rhs, dx, dy, bc="dst")
        assert result.shape == (Ny, Nx)


# ---------------------------------------------------------------------------
# streamfunction_from_vorticity — CG method
# ---------------------------------------------------------------------------


class TestStreamfunctionCG:
    @pytest.mark.slow
    def test_with_raw_mask(self, rect_mask):
        """CG with raw float mask array."""
        Ny, Nx = rect_mask.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * jnp.pi / Ny) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :] * jnp.pi / Nx
        )
        mask_arr = jnp.array(rect_mask, dtype=float)
        psi = streamfunction_from_vorticity(
            rhs, dx, dy, method="cg", mask=mask_arr, lambda_=1.0
        )
        # Should be zero outside mask
        outside = np.array(psi * (1 - mask_arr))
        np.testing.assert_allclose(outside, 0.0, atol=1e-12)

    def test_with_cgrid_mask(self, cgrid_mask):
        """CG with Mask2D extracts psi mask."""
        Ny, Nx = cgrid_mask.h.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * jnp.pi / Ny) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :] * jnp.pi / Nx
        )
        psi = streamfunction_from_vorticity(
            rhs, dx, dy, method="cg", mask=cgrid_mask, lambda_=1.0
        )
        assert psi.shape == (Ny, Nx)
        # Should be zero outside the psi mask
        psi_mask = np.array(cgrid_mask.xy_corner_strict)
        outside = np.array(psi) * (~psi_mask)
        np.testing.assert_allclose(outside, 0.0, atol=1e-12)

    def test_custom_preconditioner(self, rect_mask):
        """CG accepts a custom preconditioner callable."""
        Ny, Nx = rect_mask.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        mask_arr = jnp.array(rect_mask, dtype=float)
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * jnp.pi / Ny) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :] * jnp.pi / Nx
        )
        precond = make_spectral_preconditioner(dx, dy, lambda_=-1.0, bc="dst")
        psi = streamfunction_from_vorticity(
            rhs,
            dx,
            dy,
            method="cg",
            mask=mask_arr,
            lambda_=-1.0,
            preconditioner=precond,
        )
        outside = np.array(psi * (1 - mask_arr))
        np.testing.assert_allclose(outside, 0.0, atol=1e-12)

    def test_cg_requires_mask(self, dirichlet_grid):
        """CG without mask raises ValueError."""
        Ny, Nx, dx, dy = dirichlet_grid
        rhs = jnp.ones((Ny, Nx))
        with pytest.raises(ValueError, match="requires a mask"):
            streamfunction_from_vorticity(rhs, dx, dy, method="cg")


# ---------------------------------------------------------------------------
# streamfunction_from_vorticity — capacitance method
# ---------------------------------------------------------------------------


class TestStreamfunctionCapacitance:
    def test_with_prebuilt_solver(self, rect_mask):
        """Capacitance method with pre-built solver."""
        Ny, Nx = rect_mask.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        solver = build_capacitance_solver(
            rect_mask, dx, dy, lambda_=-1.0, base_bc="fft"
        )
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * jnp.pi / Ny) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :] * jnp.pi / Nx
        )
        psi = streamfunction_from_vorticity(
            rhs, dx, dy, method="capacitance", lambda_=-1.0, capacitance_solver=solver
        )
        # psi = 0 on the coast (every dry cell); the wet cells are unknowns
        np.testing.assert_array_equal(np.array(psi)[~rect_mask], 0.0)
        j_b, i_b = _inner_boundary_indices(rect_mask)
        assert float(jnp.max(jnp.abs(psi[j_b, i_b]))) > 0

    @pytest.mark.slow
    def test_with_cgrid_mask_solver(self, cgrid_mask):
        """Capacitance solver built from Mask2D."""
        Ny, Nx = cgrid_mask.h.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        solver = build_capacitance_solver(
            cgrid_mask, dx, dy, lambda_=-1.0, base_bc="fft"
        )
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        rhs = jnp.sin(jnp.pi * j / Ny) * jnp.cos(jnp.pi * i / Nx)
        psi = streamfunction_from_vorticity(
            rhs, dx, dy, method="capacitance", lambda_=-1.0, capacitance_solver=solver
        )
        # psi = 0 on the X wall nodes and every other dry node
        wet = np.asarray(cgrid_mask.xy_corner_strict)
        np.testing.assert_array_equal(np.array(psi)[~wet], 0.0)

    def test_capacitance_requires_solver(self, dirichlet_grid):
        """Capacitance method without solver raises ValueError."""
        Ny, Nx, dx, dy = dirichlet_grid
        rhs = jnp.ones((Ny, Nx))
        with pytest.raises(ValueError, match="requires a pre-built"):
            streamfunction_from_vorticity(rhs, dx, dy, method="capacitance")


# ---------------------------------------------------------------------------
# streamfunction — invalid method
# ---------------------------------------------------------------------------


class TestStreamfunctionInvalidMethod:
    def test_invalid_method(self, dirichlet_grid):
        Ny, Nx, dx, dy = dirichlet_grid
        rhs = jnp.ones((Ny, Nx))
        with pytest.raises(ValueError, match="method must be"):
            streamfunction_from_vorticity(rhs, dx, dy, method="invalid")


# ---------------------------------------------------------------------------
# pressure_from_divergence
# ---------------------------------------------------------------------------


class TestPressureFromDivergence:
    def test_spectral_neumann(self, neumann_grid):
        """Spectral (default): DCT-II on the T interior [1:-1, 1:-1]."""
        Ny, Nx, dx, dy = neumann_grid
        div_u = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        expected = (
            jnp.zeros((Ny, Nx))
            .at[1:-1, 1:-1]
            .set(solve_poisson_dct(div_u[1:-1, 1:-1], dx, dy))
        )
        result = pressure_from_divergence(div_u, dx, dy, method="spectral")
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-12)

    def test_zero_divergence(self, neumann_grid):
        """div=0 → p=0 (up to zero-mean gauge)."""
        Ny, Nx, dx, dy = neumann_grid
        div_u = jnp.zeros((Ny, Nx))
        p = pressure_from_divergence(div_u, dx, dy)
        np.testing.assert_allclose(np.array(p), 0.0, atol=1e-12)

    def test_output_shape(self, neumann_grid):
        Ny, Nx, dx, dy = neumann_grid
        div_u = jnp.ones((Ny, Nx))
        assert pressure_from_divergence(div_u, dx, dy).shape == (Ny, Nx)

    def test_cg_method(self, rect_mask):
        """Pressure from divergence via CG on a masked domain."""
        Ny, Nx = rect_mask.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        mask_arr = jnp.array(rect_mask, dtype=float)
        div_u = jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * jnp.pi / Ny) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :] * jnp.pi / Nx
        )
        p = pressure_from_divergence(div_u, dx, dy, method="cg", mask=mask_arr)
        # Zero outside mask
        outside = np.array(p * (1 - mask_arr))
        np.testing.assert_allclose(outside, 0.0, atol=1e-12)

    def test_capacitance_method(self, rect_mask):
        """Pressure from divergence via capacitance solver."""
        Ny, Nx = rect_mask.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        solver = build_capacitance_solver(rect_mask, dx, dy, base_bc="fft")
        div_u = jnp.zeros((Ny, Nx))
        p = pressure_from_divergence(
            div_u, dx, dy, method="capacitance", capacitance_solver=solver
        )
        assert p.shape == (Ny, Nx)


# ---------------------------------------------------------------------------
# pv_inversion
# ---------------------------------------------------------------------------


class TestPVInversionSpectral:
    def test_single_layer(self, dirichlet_grid):
        """Scalar lambda with 2-D PV field."""
        Ny, Nx, dx, dy = dirichlet_grid
        psi_exact, eig = _x_eigenmode(Ny, Nx, dx, dy)
        lam = -0.5
        rhs = (eig - lam) * psi_exact

        psi = pv_inversion(rhs, dx, dy, lambda_=lam, bc="dst", method="spectral")
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)

    @pytest.mark.slow
    def test_multilayer_vmap(self, dirichlet_grid):
        """Array of lambda values — each layer inverted independently."""
        Ny, Nx, dx, dy = dirichlet_grid
        psi_exact, eig = _x_eigenmode(Ny, Nx, dx, dy)

        lambdas = jnp.array([-0.5, -1.0, -2.0])
        nl = len(lambdas)
        rhs_layers = jnp.stack([(eig - lam) * psi_exact for lam in lambdas], axis=0)
        assert rhs_layers.shape == (nl, Ny, Nx)

        psi_out = pv_inversion(
            rhs_layers, dx, dy, lambda_=lambdas, bc="dst", method="spectral"
        )
        assert psi_out.shape == (nl, Ny, Nx)
        for k in range(nl):
            np.testing.assert_allclose(
                np.array(psi_out[k]),
                np.array(psi_exact),
                atol=1e-9,
            )

    def test_scalar_lambda_batched(self, dirichlet_grid):
        """Scalar lambda with 3-D PV (batch dim) uses vmap."""
        Ny, Nx, dx, dy = dirichlet_grid
        psi_exact, eig = _x_eigenmode(Ny, Nx, dx, dy)
        lam = -0.5
        rhs = (eig - lam) * psi_exact
        rhs_batch = jnp.stack([rhs, rhs, rhs], axis=0)

        psi_out = pv_inversion(rhs_batch, dx, dy, lambda_=lam, bc="dst")
        assert psi_out.shape == (3, Ny, Nx)
        np.testing.assert_allclose(
            np.array(psi_out[0]), np.array(psi_exact), atol=1e-10
        )


class TestPVInversionCG:
    def test_cg_with_mask(self, rect_mask):
        """PV inversion via CG on masked domain."""
        Ny, Nx = rect_mask.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        mask_arr = jnp.array(rect_mask, dtype=float)
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * jnp.pi / Ny) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :] * jnp.pi / Nx
        )
        psi = pv_inversion(rhs, dx, dy, lambda_=1.0, method="cg", mask=mask_arr)
        outside = np.array(psi * (1 - mask_arr))
        np.testing.assert_allclose(outside, 0.0, atol=1e-12)


class TestPVInversionBatchDims:
    @pytest.mark.slow
    def test_array_lambda_with_batch_dims(self, dirichlet_grid):
        """Array lambda with leading batch dims: (batch, nl, Ny, Nx)."""
        Ny, Nx, dx, dy = dirichlet_grid
        psi_exact, eig = _x_eigenmode(Ny, Nx, dx, dy)

        lambdas = jnp.array([-0.5, -1.0, -2.0])
        nl = len(lambdas)
        rhs_layers = jnp.stack([(eig - lam) * psi_exact for lam in lambdas], axis=0)
        # Add a batch dimension: (2, nl, Ny, Nx)
        rhs_batch = jnp.stack([rhs_layers, rhs_layers], axis=0)
        assert rhs_batch.shape == (2, nl, Ny, Nx)

        psi_out = pv_inversion(
            rhs_batch, dx, dy, lambda_=lambdas, bc="dst", method="spectral"
        )
        assert psi_out.shape == (2, nl, Ny, Nx)
        for b in range(2):
            for k in range(nl):
                np.testing.assert_allclose(
                    np.array(psi_out[b, k]),
                    np.array(psi_exact),
                    atol=1e-9,
                )


class TestPVInversionErrors:
    def test_array_lambda_shape_mismatch(self, dirichlet_grid):
        """Mismatched lambda array length raises ValueError."""
        Ny, Nx, dx, dy = dirichlet_grid
        pv = jnp.ones((3, Ny, Nx))
        lambdas = jnp.array([-0.5, -1.0])  # length 2 vs 3 layers
        with pytest.raises(ValueError, match="does not match"):
            pv_inversion(pv, dx, dy, lambda_=lambdas, bc="dst")

    def test_array_lambda_2d_pv_raises(self, dirichlet_grid):
        """Array lambda with 2-D PV raises ValueError."""
        Ny, Nx, dx, dy = dirichlet_grid
        pv = jnp.ones((Ny, Nx))
        lambdas = jnp.array([-0.5, -1.0])
        with pytest.raises(ValueError, match="at least 3 dims"):
            pv_inversion(pv, dx, dy, lambda_=lambdas, bc="dst")

    def test_array_lambda_capacitance_raises(self, dirichlet_grid):
        """Array lambda with method='capacitance' raises ValueError."""
        Ny, Nx, dx, dy = dirichlet_grid
        pv = jnp.ones((3, Ny, Nx))
        lambdas = jnp.array([-0.5, -1.0, -2.0])
        with pytest.raises(ValueError, match="does not support array-valued"):
            pv_inversion(pv, dx, dy, lambda_=lambdas, bc="dst", method="capacitance")


# ---------------------------------------------------------------------------
# Mask2D integration with existing solvers
# ---------------------------------------------------------------------------


class TestMaskedLaplacianWithCGridMask:
    def test_cgrid_mask_matches_array_mask(self, rect_mask, cgrid_mask):
        """Mask2D path matches float-array path using psi mask."""
        Ny, Nx = rect_mask.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        psi = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        psi_mask_arr = jnp.array(cgrid_mask.xy_corner_strict, dtype=float)
        result_arr = masked_laplacian(psi, psi_mask_arr, dx, dy, lambda_=-1.0)
        result_cgrid = masked_laplacian(psi, cgrid_mask, dx, dy, lambda_=-1.0)
        np.testing.assert_allclose(
            np.array(result_cgrid), np.array(result_arr), atol=1e-14
        )


class TestCapacitanceSolverWithCGridMask:
    def test_build_from_cgrid_mask(self, cgrid_mask):
        """build_capacitance_solver accepts Mask2D."""
        Ny, Nx = cgrid_mask.h.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        solver = build_capacitance_solver(
            cgrid_mask, dx, dy, lambda_=-1.0, base_bc="fft"
        )
        rhs = jnp.zeros((Ny, Nx))
        psi = solver(rhs)
        assert psi.shape == (Ny, Nx)

    def test_dirichlet_bc_enforced(self, cgrid_mask):
        """psi = 0 on the X wall nodes; every wet X node solves the PDE (#306)."""
        Ny, Nx = cgrid_mask.h.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        solver = build_capacitance_solver(
            cgrid_mask, dx, dy, lambda_=-1.0, base_bc="fft"
        )
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        rhs = jnp.sin(jnp.pi * j / Ny) * jnp.cos(jnp.pi * i / Nx)
        psi = solver(rhs)
        wet = np.asarray(cgrid_mask.xy_corner_strict)
        # The wall nodes X 0 and N-2 (and everything dry) are zero.
        np.testing.assert_array_equal(np.array(psi)[~wet], 0.0)
        res = masked_laplacian(psi, cgrid_mask, dx, dy, lambda_=-1.0) - rhs * wet
        # Direct solve: round-off only.
        assert float(jnp.max(jnp.abs(res))) < 1e-10


# ---------------------------------------------------------------------------
# Mask dtype follows the right-hand side (#302)
# ---------------------------------------------------------------------------


class TestMaskDtypeFollowsRhs:
    """A ``Mask2D`` must not downcast a weakly typed float64 right-hand side."""

    N = 34

    def _problem(self):
        n = self.N
        dx = 1.0 / (n - 2)
        m = np.zeros((n, n), dtype=bool)
        m[1:-1, 1:-1] = True
        mask = Mask2D.from_mask(m)
        # arange * python scalar: weakly typed float64, |b| ~ 800
        x = jnp.arange(n) * dx
        rhs = (
            800.0 * jnp.sin(3 * jnp.pi * x[None, :]) * jnp.sin(2 * jnp.pi * x[:, None])
        )
        return rhs, mask, dx

    @pytest.mark.slow
    def test_weak_float64_rhs_with_mask2d_stays_float64(self):
        from finitevolx._src.solvers.elliptic import _resolve_mask_arr, solve_cg

        rhs, mask, dx = self._problem()
        assert rhs.dtype == jnp.float64 and rhs.weak_type
        psi = streamfunction_from_vorticity(
            rhs, dx, dx, lambda_=4.0, method="cg", mask=mask
        )
        assert psi.dtype == jnp.float64

        # The same solve the wrapper runs, with its diagnostics: float64 CG
        # converges in ~11 steps (float32 stalled at |r| ~ 1e-2).
        m_arr = _resolve_mask_arr(mask, rhs.dtype)
        assert m_arr.dtype == jnp.float64
        _, info = solve_cg(
            lambda p: masked_laplacian(p, m_arr, dx, dx, lambda_=4.0),
            rhs * m_arr,
            preconditioner=make_spectral_preconditioner(dx, dx, lambda_=4.0, bc="fft"),
        )
        assert bool(info.converged)
        assert int(info.iterations) <= 15

    @pytest.mark.slow
    def test_float32_rhs_stays_float32(self):
        rhs, mask, dx = self._problem()
        psi = streamfunction_from_vorticity(
            rhs.astype(jnp.float32), dx, dx, lambda_=4.0, method="cg", mask=mask
        )
        assert psi.dtype == jnp.float32

    @pytest.mark.slow
    def test_known_values_with_mask2d_stays_float64(self):
        rhs, mask, dx = self._problem()
        kv = jnp.ones_like(rhs)
        psi = streamfunction_from_vorticity(
            rhs, dx, dx, lambda_=4.0, method="cg", mask=mask, known_values=kv
        )
        assert psi.dtype == jnp.float64

    def test_solve_domain_masks_are_boolean(self):
        from finitevolx._src.solvers.inhomogeneous import SolveDomain

        _, mask, _ = self._problem()
        domain = SolveDomain(mask)
        for arr in (domain.wet_mask, domain.boundary_ring, domain.effective_mask):
            assert arr.dtype == jnp.bool_


# ---------------------------------------------------------------------------
# Wall placement on the C-grid (#300)
# ---------------------------------------------------------------------------


def _manufactured(staggering, bc, n):
    """Manufactured field, its continuous (∇² - 0) source and the unknown slice.

    ``from_interior(n, n, 1, 1)``: dx = 1/n, the walls are at x = 0 and 1.
    X node i sits at x = i dx (wall nodes 0 and n = N-2); T cell i at
    x = (i - 1/2) dx (walls on the faces).  Returns ``(exact, k2, sl)`` with
    ∇² exact = -k2 exact and ``sl`` the slice of the unknowns.
    """
    N = n + 2
    d = 1.0 / n
    idx = jnp.arange(N, dtype=float)
    x = idx * d if staggering == "X" else (idx - 0.5) * d
    X, Y = x[None, :], x[:, None]
    if bc == "dst":  # zero on the walls
        exact = jnp.sin(jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)
        k2 = 5 * jnp.pi**2
    elif bc == "dct":  # zero normal derivative on the walls
        exact = jnp.cos(jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)
        k2 = 5 * jnp.pi**2
    else:  # periodic with period 1
        exact = jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)
        k2 = 8 * jnp.pi**2
    sl = {("X", "dst"): slice(1, -2), ("X", "dct"): slice(0, -1)}.get(
        (staggering, bc), slice(1, -1)
    )
    return exact, k2, sl, d


def _wall_error(staggering, bc, n, lambda_):
    exact, k2, sl, d = _manufactured(staggering, bc, n)
    rhs = -(k2 + lambda_) * exact
    if staggering == "X":
        psi = streamfunction_from_vorticity(rhs, d, d, bc=bc, lambda_=lambda_)
    else:
        assert lambda_ == 0.0
        psi = pressure_from_divergence(rhs, d, d, bc=bc)
    err = psi[sl, sl] - exact[sl, sl]
    if lambda_ == 0.0 and bc != "dst":
        err = err - jnp.mean(err)  # the gauge (null mode) is arbitrary
    return float(jnp.max(jnp.abs(err)))


class TestWallPlacement:
    """Spectral solves put the wall on the C-grid wall: second-order convergence.

    Before the fix the whole array (ghost ring included) went to the
    transform, which put the wall one cell outside: rate ~ 0.9.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("staggering", "bc", "lambda_"),
        [
            ("X", "dst", 0.0),
            ("X", "dst", 4.0),
            ("X", "dct", 4.0),
            ("X", "dct", 0.0),
            ("X", "fft", 4.0),
            ("T", "dct", 0.0),
            ("T", "dst", 0.0),
            ("T", "fft", 0.0),
        ],
    )
    def test_second_order(self, staggering, bc, lambda_):
        errs = [_wall_error(staggering, bc, n, lambda_) for n in (16, 32, 64)]
        rates = [np.log2(errs[k] / errs[k + 1]) for k in range(2)]
        # Measured 1.97-2.01 for every combination (truncation O(dx^2)).
        assert min(rates) >= 1.9, (errs, rates)

    @pytest.mark.slow
    @pytest.mark.slow
    @pytest.mark.parametrize("base_bc", ["dst", "fft"])
    def test_capacitance_second_order(self, base_bc):
        """#306: capacitance on the rectangle's Mask2D has the C-grid wall.

        Before, its wall was the mask's inner wet ring: rate ~0.97.
        """
        errs = []
        for n in (16, 32, 64):
            exact, k2, sl, d = _manufactured("X", "dst", n)
            m = np.zeros((n + 2, n + 2), dtype=bool)
            m[1:-1, 1:-1] = True
            mask = Mask2D.from_mask(m)
            solver = build_capacitance_solver(mask, d, d, lambda_=4.0, base_bc=base_bc)
            psi = streamfunction_from_vorticity(
                -(k2 + 4.0) * exact,
                d,
                d,
                lambda_=4.0,
                method="capacitance",
                capacitance_solver=solver,
            )
            errs.append(float(jnp.max(jnp.abs(psi[sl, sl] - exact[sl, sl]))))
        rates = [np.log2(errs[k] / errs[k + 1]) for k in range(2)]
        # Same discrete problem as the spectral DST: rate 2.00.
        assert min(rates) >= 1.9, (errs, rates)

    def test_x_dst_matches_masked_laplacian_cg(self):
        """Spectral DST equals CG on the rectangle's Mask2D (dry neighbours = 0)."""
        n = 16
        grid = CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
        m = np.zeros((n + 2, n + 2), dtype=bool)
        m[1:-1, 1:-1] = True
        mask = Mask2D.from_mask(m)
        rng = np.random.default_rng(0)
        rhs = jnp.asarray(rng.standard_normal((n + 2, n + 2)))
        lam = 4.0
        psi_sp = streamfunction_from_vorticity(rhs, grid.dx, grid.dy, lambda_=lam)

        m_psi = jnp.asarray(mask.xy_corner_strict, dtype=float)
        psi_cg, info = solve_cg(
            lambda p: masked_laplacian(p, m_psi, grid.dx, grid.dy, lambda_=lam),
            rhs * m_psi,
            rtol=1e-14,
            atol=1e-14,
            max_steps=2000,
        )
        assert bool(info.converged)
        # Same discrete problem; both exact to round-off (~1e-14 relative).
        np.testing.assert_allclose(psi_sp, psi_cg * m_psi, atol=1e-10)
        # The default-tolerance wrapper agrees to CG's rtol/atol (1e-6).
        psi_cg_w = streamfunction_from_vorticity(
            rhs, grid.dx, grid.dy, lambda_=lam, method="cg", mask=mask
        )
        np.testing.assert_allclose(psi_sp, psi_cg_w, atol=1e-6)

    def test_pv_inversion_array_lambda_uses_same_wall(self, dirichlet_grid):
        """The per-layer (traced lambda) path solves the same problem."""
        Ny, Nx, dx, dy = dirichlet_grid
        rng = np.random.default_rng(1)
        pv = jnp.asarray(rng.standard_normal((2, Ny, Nx)))
        lambdas = jnp.array([1.0, 9.0])
        psi = pv_inversion(pv, dx, dy, lambdas)
        for k in range(2):
            ref = streamfunction_from_vorticity(
                pv[k], dx, dy, lambda_=float(lambdas[k])
            )
            np.testing.assert_allclose(psi[k], ref, atol=1e-12)

    @pytest.mark.slow
    def test_mask2d_pressure_uses_h_mask(self):
        """pressure_from_divergence(method='cg', Mask2D) solves on mask.h."""
        n = 8
        m = np.zeros((n + 2, n + 2), dtype=bool)
        m[1:-1, 1:-1] = True
        mask = Mask2D.from_mask(m)
        rng = np.random.default_rng(2)
        rhs = jnp.asarray(rng.standard_normal((n + 2, n + 2)))
        p_m = pressure_from_divergence(rhs, 0.1, 0.1, method="cg", mask=mask)
        p_h = pressure_from_divergence(
            rhs, 0.1, 0.1, method="cg", mask=jnp.asarray(mask.h, dtype=float)
        )
        np.testing.assert_allclose(p_m, p_h, atol=1e-12)
        assert float(jnp.max(jnp.abs(p_m[1:-1, -2]))) > 0  # last T column is wet
