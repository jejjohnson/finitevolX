"""Tests for streamfunction, pressure, and PV-inversion convenience wrappers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.solvers.elliptic import (
    build_capacitance_solver,
    dst1_eigenvalues,
    make_spectral_preconditioner,
    masked_laplacian,
    pressure_from_divergence,
    pv_inversion,
    solve_poisson_dct,
    streamfunction_from_vorticity,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dirichlet_grid():
    """Interior-only grid for DST-I tests."""
    Ny, Nx = 10, 12
    dx = 1.0 / (Nx + 1)
    dy = 1.0 / (Ny + 1)
    return Ny, Nx, dx, dy


@pytest.fixture
def neumann_grid():
    Ny, Nx = 8, 10
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    return Ny, Nx, dx, dy


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
    """ArakawaCGridMask built from the rectangular mask."""
    return ArakawaCGridMask.from_mask(rect_mask)


# ---------------------------------------------------------------------------
# streamfunction_from_vorticity — spectral method
# ---------------------------------------------------------------------------


class TestStreamfunctionSpectral:
    def test_roundtrip_dst(self, dirichlet_grid):
        """∇²ψ_exact → ζ, invert via DST → ψ ≈ ψ_exact."""
        Ny, Nx, dx, dy = dirichlet_grid
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.sin(jnp.pi * (j + 1) / (Ny + 1)) * jnp.sin(
            jnp.pi * (i + 1) / (Nx + 1)
        )
        lx0 = dst1_eigenvalues(Nx, dx)[0]
        ly0 = dst1_eigenvalues(Ny, dy)[0]
        zeta = (lx0 + ly0) * psi_exact

        psi = streamfunction_from_vorticity(zeta, dx, dy, bc="dst", method="spectral")
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)

    def test_roundtrip_dct(self, neumann_grid):
        """Neumann-BC spectral wrapper matches solve_poisson_dct."""
        Ny, Nx, dx, dy = neumann_grid
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        expected = solve_poisson_dct(rhs, dx, dy)
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
            rhs, dx, dy, method="cg", mask=mask_arr, lambda_=-1.0
        )
        # Should be zero outside mask
        outside = np.array(psi * (1 - mask_arr))
        np.testing.assert_allclose(outside, 0.0, atol=1e-12)

    def test_with_cgrid_mask(self, cgrid_mask):
        """CG with ArakawaCGridMask extracts psi mask."""
        Ny, Nx = cgrid_mask.h.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * jnp.pi / Ny) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :] * jnp.pi / Nx
        )
        psi = streamfunction_from_vorticity(
            rhs, dx, dy, method="cg", mask=cgrid_mask, lambda_=-1.0
        )
        assert psi.shape == (Ny, Nx)
        # Should be zero outside the psi mask
        psi_mask = np.array(cgrid_mask.psi)
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
        # BC enforcement: zero at boundary indices
        bc_vals = psi[solver._j_b, solver._i_b]
        np.testing.assert_allclose(np.array(bc_vals), 0.0, atol=1e-10)

    def test_with_cgrid_mask_solver(self, cgrid_mask):
        """Capacitance solver built from ArakawaCGridMask."""
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
        bc_vals = psi[solver._j_b, solver._i_b]
        np.testing.assert_allclose(np.array(bc_vals), 0.0, atol=1e-10)

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
        """Spectral (default) with DCT matches solve_poisson_dct."""
        Ny, Nx, dx, dy = neumann_grid
        div_u = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        expected = solve_poisson_dct(div_u, dx, dy)
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
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.sin(jnp.pi * (j + 1) / (Ny + 1)) * jnp.sin(
            jnp.pi * (i + 1) / (Nx + 1)
        )
        lx0 = dst1_eigenvalues(Nx, dx)[0]
        ly0 = dst1_eigenvalues(Ny, dy)[0]
        lam = -0.5
        rhs = (lx0 + ly0 - lam) * psi_exact

        psi = pv_inversion(rhs, dx, dy, lambda_=lam, bc="dst", method="spectral")
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)

    def test_multilayer_vmap(self, dirichlet_grid):
        """Array of lambda values — each layer inverted independently."""
        Ny, Nx, dx, dy = dirichlet_grid
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.sin(jnp.pi * (j + 1) / (Ny + 1)) * jnp.sin(
            jnp.pi * (i + 1) / (Nx + 1)
        )
        lx0 = dst1_eigenvalues(Nx, dx)[0]
        ly0 = dst1_eigenvalues(Ny, dy)[0]

        lambdas = jnp.array([-0.5, -1.0, -2.0])
        nl = len(lambdas)
        rhs_layers = jnp.stack(
            [(lx0 + ly0 - lam) * psi_exact for lam in lambdas], axis=0
        )
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
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.sin(jnp.pi * (j + 1) / (Ny + 1)) * jnp.sin(
            jnp.pi * (i + 1) / (Nx + 1)
        )
        lx0 = dst1_eigenvalues(Nx, dx)[0]
        ly0 = dst1_eigenvalues(Ny, dy)[0]
        lam = -0.5
        rhs = (lx0 + ly0 - lam) * psi_exact
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
        psi = pv_inversion(rhs, dx, dy, lambda_=-1.0, method="cg", mask=mask_arr)
        outside = np.array(psi * (1 - mask_arr))
        np.testing.assert_allclose(outside, 0.0, atol=1e-12)


class TestPVInversionBatchDims:
    def test_array_lambda_with_batch_dims(self, dirichlet_grid):
        """Array lambda with leading batch dims: (batch, nl, Ny, Nx)."""
        Ny, Nx, dx, dy = dirichlet_grid
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.sin(jnp.pi * (j + 1) / (Ny + 1)) * jnp.sin(
            jnp.pi * (i + 1) / (Nx + 1)
        )
        lx0 = dst1_eigenvalues(Nx, dx)[0]
        ly0 = dst1_eigenvalues(Ny, dy)[0]

        lambdas = jnp.array([-0.5, -1.0, -2.0])
        nl = len(lambdas)
        rhs_layers = jnp.stack(
            [(lx0 + ly0 - lam) * psi_exact for lam in lambdas], axis=0
        )
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
# ArakawaCGridMask integration with existing solvers
# ---------------------------------------------------------------------------


class TestMaskedLaplacianWithCGridMask:
    def test_cgrid_mask_matches_array_mask(self, rect_mask, cgrid_mask):
        """ArakawaCGridMask path matches float-array path using psi mask."""
        Ny, Nx = rect_mask.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        psi = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        psi_mask_arr = jnp.array(cgrid_mask.psi, dtype=float)
        result_arr = masked_laplacian(psi, psi_mask_arr, dx, dy, lambda_=-1.0)
        result_cgrid = masked_laplacian(psi, cgrid_mask, dx, dy, lambda_=-1.0)
        np.testing.assert_allclose(
            np.array(result_cgrid), np.array(result_arr), atol=1e-14
        )


class TestCapacitanceSolverWithCGridMask:
    def test_build_from_cgrid_mask(self, cgrid_mask):
        """build_capacitance_solver accepts ArakawaCGridMask."""
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
        """Capacitance solver from CGridMask enforces zero at boundary."""
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
        bc_vals = psi[solver._j_b, solver._i_b]
        np.testing.assert_allclose(np.array(bc_vals), 0.0, atol=1e-10)
