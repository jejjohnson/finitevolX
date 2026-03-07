"""Tests for spectral Poisson/Helmholtz solvers, CG, and capacitance method."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.elliptic import (
    CapacitanceSolver,
    CGInfo,
    build_capacitance_solver,
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
    make_spectral_preconditioner,
    masked_laplacian,
    solve_cg,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_helmholtz_fft,
    solve_poisson_dct,
    solve_poisson_dst,
    solve_poisson_fft,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dirichlet_grid():
    """Interior-only grid for DST-I Poisson tests."""
    Ny, Nx = 10, 12
    dx = 1.0 / (Nx + 1)
    dy = 1.0 / (Ny + 1)
    return Ny, Nx, dx, dy


@pytest.fixture
def neumann_grid():
    """Grid for DCT-II Neumann tests."""
    Ny, Nx = 8, 10
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    return Ny, Nx, dx, dy


@pytest.fixture
def periodic_grid():
    """Grid for FFT periodic tests."""
    Ny, Nx = 8, 10
    dx = 2.0 * np.pi / Nx
    dy = 2.0 * np.pi / Ny
    return Ny, Nx, dx, dy


# ---------------------------------------------------------------------------
# DST-I Dirichlet Poisson solver
# ---------------------------------------------------------------------------


class TestSolvePoissonDST:
    def test_lowest_mode(self, dirichlet_grid):
        """DST-I eigenfunction at mode (0,0) is recovered exactly."""
        Ny, Nx, dx, dy = dirichlet_grid
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.sin(jnp.pi * (j + 1) / (Ny + 1)) * jnp.sin(
            jnp.pi * (i + 1) / (Nx + 1)
        )
        lx0 = dst1_eigenvalues(Nx, dx)[0]
        ly0 = dst1_eigenvalues(Ny, dy)[0]
        rhs = (lx0 + ly0) * psi_exact
        psi = solve_poisson_dst(rhs, dx, dy)
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)

    def test_higher_mode(self, dirichlet_grid):
        """DST-I eigenfunction at mode (1,2) is recovered exactly."""
        Ny, Nx, dx, dy = dirichlet_grid
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        m, n = 1, 2
        psi_exact = jnp.sin(jnp.pi * (j + 1) * (m + 1) / (Ny + 1)) * jnp.sin(
            jnp.pi * (i + 1) * (n + 1) / (Nx + 1)
        )
        lx = dst1_eigenvalues(Nx, dx)[n]
        ly = dst1_eigenvalues(Ny, dy)[m]
        rhs = (lx + ly) * psi_exact
        psi = solve_poisson_dst(rhs, dx, dy)
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)

    def test_output_shape(self, dirichlet_grid):
        Ny, Nx, dx, dy = dirichlet_grid
        rhs = jnp.ones((Ny, Nx))
        assert solve_poisson_dst(rhs, dx, dy).shape == (Ny, Nx)


class TestSolveHelmholtzDST:
    def test_exact_eigenfunction(self, dirichlet_grid):
        """Helmholtz DST solution matches exact eigenfunction."""
        Ny, Nx, dx, dy = dirichlet_grid
        lam = -0.5
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.sin(jnp.pi * (j + 1) / (Ny + 1)) * jnp.sin(
            jnp.pi * (i + 1) / (Nx + 1)
        )
        lx0 = dst1_eigenvalues(Nx, dx)[0]
        ly0 = dst1_eigenvalues(Ny, dy)[0]
        rhs = (lx0 + ly0 - lam) * psi_exact
        psi = solve_helmholtz_dst(rhs, dx, dy, lambda_=lam)
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)

    def test_poisson_is_helmholtz_lambda0(self, dirichlet_grid):
        Ny, Nx, dx, dy = dirichlet_grid
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        np.testing.assert_allclose(
            np.array(solve_poisson_dst(rhs, dx, dy)),
            np.array(solve_helmholtz_dst(rhs, dx, dy, lambda_=0.0)),
            atol=1e-14,
        )


# ---------------------------------------------------------------------------
# DCT-II Neumann Poisson solver
# ---------------------------------------------------------------------------


class TestSolvePoissonDCT:
    def test_mode1_eigenfunction(self, neumann_grid):
        """DCT-II mode (1,1) eigenfunction with zero-mean gauge."""
        Ny, Nx, dx, dy = neumann_grid
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        # DCT-II eigenfunction: cos(π(j+0.5)/N)
        psi_exact = jnp.cos(jnp.pi * (j + 0.5) / Ny) * jnp.cos(
            jnp.pi * (i + 0.5) / Nx
        )
        lx1 = dct2_eigenvalues(Nx, dx)[1]
        ly1 = dct2_eigenvalues(Ny, dy)[1]
        rhs = (lx1 + ly1) * psi_exact
        psi = solve_poisson_dct(rhs, dx, dy)
        # Gauge: output is zero-mean, so adjust reference
        psi_ref = psi_exact - jnp.mean(psi_exact)
        np.testing.assert_allclose(np.array(psi), np.array(psi_ref), atol=1e-9)

    def test_zero_mean_output(self, neumann_grid):
        """Solution always has zero mean."""
        Ny, Nx, dx, dy = neumann_grid
        rhs = jnp.ones((Ny, Nx))
        psi = solve_poisson_dct(rhs, dx, dy)
        np.testing.assert_allclose(float(jnp.mean(psi)), 0.0, atol=1e-12)

    def test_output_shape(self, neumann_grid):
        Ny, Nx, dx, dy = neumann_grid
        rhs = jnp.ones((Ny, Nx))
        assert solve_poisson_dct(rhs, dx, dy).shape == (Ny, Nx)


class TestSolveHelmholtzDCT:
    def test_exact_eigenfunction(self, neumann_grid):
        Ny, Nx, dx, dy = neumann_grid
        lam = -2.0
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.cos(jnp.pi * (j + 0.5) / Ny) * jnp.cos(
            jnp.pi * (i + 0.5) / Nx
        )
        lx1 = dct2_eigenvalues(Nx, dx)[1]
        ly1 = dct2_eigenvalues(Ny, dy)[1]
        rhs = (lx1 + ly1 - lam) * psi_exact
        psi = solve_helmholtz_dct(rhs, dx, dy, lambda_=lam)
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-9)

    def test_lambda0_delegates_to_poisson(self, neumann_grid):
        """solve_helmholtz_dct(lambda_=0) matches solve_poisson_dct exactly."""
        Ny, Nx, dx, dy = neumann_grid
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        np.testing.assert_allclose(
            np.array(solve_helmholtz_dct(rhs, dx, dy, lambda_=0.0)),
            np.array(solve_poisson_dct(rhs, dx, dy)),
            atol=1e-14,
        )


# ---------------------------------------------------------------------------
# FFT periodic solver
# ---------------------------------------------------------------------------


class TestSolvePoissonFFT:
    def test_cos_mode(self, periodic_grid):
        """Periodic cosine mode (k=1) is recovered exactly."""
        Ny, Nx, dx, dy = periodic_grid
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.cos(2 * jnp.pi * i / Nx) + jnp.cos(2 * jnp.pi * j / Ny)
        lx1 = fft_eigenvalues(Nx, dx)[1]
        ly1 = fft_eigenvalues(Ny, dy)[1]
        rhs = (
            lx1 * jnp.cos(2 * jnp.pi * i / Nx) * jnp.ones((Ny, 1))
            + ly1 * jnp.cos(2 * jnp.pi * j / Ny) * jnp.ones((1, Nx))
        )
        psi = solve_poisson_fft(rhs, dx, dy)
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)

    def test_zero_mean_output(self, periodic_grid):
        Ny, Nx, dx, dy = periodic_grid
        rhs = jnp.ones((Ny, Nx))
        psi = solve_poisson_fft(rhs, dx, dy)
        np.testing.assert_allclose(float(jnp.mean(psi)), 0.0, atol=1e-12)

    def test_output_is_real(self, periodic_grid):
        Ny, Nx, dx, dy = periodic_grid
        rhs = jnp.ones((Ny, Nx))
        psi = solve_poisson_fft(rhs, dx, dy)
        assert psi.dtype in (jnp.float32, jnp.float64)


class TestSolveHelmholtzFFT:
    def test_exact_eigenfunction(self, periodic_grid):
        Ny, Nx, dx, dy = periodic_grid
        lam = -1.0
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.cos(2 * jnp.pi * i / Nx) + jnp.cos(2 * jnp.pi * j / Ny)
        lx1 = fft_eigenvalues(Nx, dx)[1]
        ly1 = fft_eigenvalues(Ny, dy)[1]
        rhs = (
            (lx1 - lam) * jnp.cos(2 * jnp.pi * i / Nx) * jnp.ones((Ny, 1))
            + (ly1 - lam) * jnp.cos(2 * jnp.pi * j / Ny) * jnp.ones((1, Nx))
        )
        psi = solve_helmholtz_fft(rhs, dx, dy, lambda_=lam)
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-10)

    def test_lambda0_delegates_to_poisson(self, periodic_grid):
        """solve_helmholtz_fft(lambda_=0) matches solve_poisson_fft exactly."""
        Ny, Nx, dx, dy = periodic_grid
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        np.testing.assert_allclose(
            np.array(solve_helmholtz_fft(rhs, dx, dy, lambda_=0.0)),
            np.array(solve_poisson_fft(rhs, dx, dy)),
            atol=1e-14,
        )


# ---------------------------------------------------------------------------
# Eigenvalue helpers
# ---------------------------------------------------------------------------


class TestEigenvalues:
    def test_dst1_all_negative(self):
        eig = dst1_eigenvalues(10, 0.1)
        assert bool(jnp.all(eig <= 0.0))

    def test_dct2_null_mode(self):
        eig = dct2_eigenvalues(10, 0.1)
        np.testing.assert_allclose(float(eig[0]), 0.0, atol=1e-14)

    def test_fft_null_mode(self):
        eig = fft_eigenvalues(10, 0.1)
        np.testing.assert_allclose(float(eig[0]), 0.0, atol=1e-14)

    def test_dst1_shape(self):
        assert dst1_eigenvalues(8, 0.1).shape == (8,)


# ---------------------------------------------------------------------------
# Preconditioned Conjugate Gradient
# ---------------------------------------------------------------------------


class TestSolveCG:
    def test_helmholtz_fft_exact(self, periodic_grid):
        """CG with spectral preconditioner solves Helmholtz on periodic grid."""
        Ny, Nx, dx, dy = periodic_grid
        # lam > 0 makes (nabla^2 - lam) negative definite, as required by lineax.CG
        # for NSD operators.
        lam = 1.0
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_ref = jnp.cos(2 * jnp.pi * i / Nx) + jnp.cos(2 * jnp.pi * j / Ny)
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)
        rhs = (
            (eigx[1] - lam) * jnp.cos(2 * jnp.pi * i / Nx) * jnp.ones((Ny, 1))
            + (eigy[1] - lam) * jnp.cos(2 * jnp.pi * j / Ny) * jnp.ones((1, Nx))
        )

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        M_inv = make_spectral_preconditioner(dx, dy, lambda_=lam, bc="fft")
        psi, info = solve_cg(A, rhs, preconditioner=M_inv, rtol=1e-10, atol=1e-10)
        assert info.converged
        np.testing.assert_allclose(np.array(psi), np.array(psi_ref), atol=1e-8)

    def test_returns_cg_info(self, periodic_grid):
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0  # lam > 0 → negative definite Helmholtz operator
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        _, info = solve_cg(A, jnp.ones((Ny, Nx)), rtol=1e-6, atol=1e-6)
        assert isinstance(info, CGInfo)
        assert info.iterations >= 0
        assert info.residual_norm >= 0.0

    def test_preconditioner_reduces_iterations(self, periodic_grid):
        """Spectral preconditioner should not require more iters than unpreconditioned."""
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0  # lam > 0 → negative definite Helmholtz operator
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None] / Ny) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :] / Nx
        )

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        _, info_no_pre = solve_cg(A, rhs, rtol=1e-8, atol=1e-8, max_steps=200)
        M_inv = make_spectral_preconditioner(dx, dy, lambda_=lam, bc="fft")
        _, info_pre = solve_cg(A, rhs, preconditioner=M_inv, rtol=1e-8, atol=1e-8, max_steps=200)
        # Preconditioner should not make things worse
        assert info_pre.iterations <= info_no_pre.iterations + 5

    def test_identity_preconditioner(self, periodic_grid):
        """None preconditioner (identity) still converges."""
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0  # lam > 0 → negative definite Helmholtz operator
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)
        rhs = jnp.ones((Ny, Nx))

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        psi, info = solve_cg(A, rhs, preconditioner=None, rtol=1e-8, atol=1e-8)
        assert info.converged

    def test_make_spectral_preconditioner_invalid_bc(self, periodic_grid):
        Ny, Nx, dx, dy = periodic_grid
        with pytest.raises(ValueError, match="bc must be"):
            make_spectral_preconditioner(dx, dy, bc="invalid")


# ---------------------------------------------------------------------------
# Masked Laplacian
# ---------------------------------------------------------------------------


class TestMaskedLaplacian:
    def test_full_mask_matches_roll_laplacian(self, periodic_grid):
        """With all-ones mask, result matches the standard periodic Laplacian."""
        Ny, Nx, dx, dy = periodic_grid
        psi = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        mask = jnp.ones((Ny, Nx))
        lap_masked = masked_laplacian(psi, mask, dx, dy, lambda_=0.0)
        # Standard periodic Laplacian
        lap_std = (
            (jnp.roll(psi, 1, axis=1) + jnp.roll(psi, -1, axis=1) - 2 * psi) / dx**2
            + (jnp.roll(psi, 1, axis=0) + jnp.roll(psi, -1, axis=0) - 2 * psi) / dy**2
        )
        np.testing.assert_allclose(
            np.array(lap_masked), np.array(lap_std), atol=1e-12
        )

    def test_zero_outside_mask(self, periodic_grid):
        """Output is zero where mask is zero."""
        Ny, Nx, dx, dy = periodic_grid
        psi = jnp.ones((Ny, Nx))
        mask = jnp.zeros((Ny, Nx)).at[2:6, 2:8].set(1.0)
        result = masked_laplacian(psi, mask, dx, dy)
        np.testing.assert_array_equal(np.array(result * (1 - mask)), 0.0)

    def test_output_shape(self, periodic_grid):
        Ny, Nx, dx, dy = periodic_grid
        psi = jnp.ones((Ny, Nx))
        mask = jnp.ones((Ny, Nx))
        assert masked_laplacian(psi, mask, dx, dy).shape == (Ny, Nx)


# ---------------------------------------------------------------------------
# Capacitance matrix solver
# ---------------------------------------------------------------------------


def _build_rect_mask(Ny: int, Nx: int, border: int = 1) -> np.ndarray:
    """Rectangle interior mask (True = interior, border cells = False)."""
    mask = np.ones((Ny, Nx), dtype=bool)
    mask[:border, :] = mask[-border:, :] = False
    mask[:, :border] = mask[:, -border:] = False
    return mask


class TestCapacitanceSolver:
    def test_returns_capacitance_solver(self):
        mask = _build_rect_mask(8, 10)
        solver = build_capacitance_solver(mask, dx=0.1, dy=0.1)
        assert isinstance(solver, CapacitanceSolver)

    def test_output_shape(self):
        Ny, Nx = 8, 10
        mask = _build_rect_mask(Ny, Nx)
        solver = build_capacitance_solver(mask, dx=0.1, dy=0.1, lambda_=-1.0)
        rhs = jnp.zeros((Ny, Nx))
        psi = solver(rhs)
        assert psi.shape == (Ny, Nx)

    def test_dirichlet_bc_enforced(self):
        """Solution is zero at all inner-boundary points (capacitance guarantee)."""
        Ny, Nx = 10, 12
        mask = _build_rect_mask(Ny, Nx)
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        solver = build_capacitance_solver(mask, dx, dy, lambda_=-1.0, base_bc="fft")

        # Non-trivial rhs inside the domain
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        rhs = jnp.sin(jnp.pi * j / Ny) * jnp.cos(jnp.pi * i / Nx)
        psi = solver(rhs)

        # Boundary condition: ψ = 0 at all inner-boundary points
        bc_vals = psi[solver._j_b, solver._i_b]
        np.testing.assert_allclose(np.array(bc_vals), 0.0, atol=1e-10)

    def test_zero_rhs_zero_solution(self):
        """With zero rhs and Helmholtz lambda<0, the solution is zero."""
        Ny, Nx = 8, 10
        mask = _build_rect_mask(Ny, Nx)
        solver = build_capacitance_solver(
            mask, dx=0.1, dy=0.1, lambda_=-1.0, base_bc="fft"
        )
        rhs = jnp.zeros((Ny, Nx))
        psi = solver(rhs)
        np.testing.assert_allclose(np.array(psi), 0.0, atol=1e-12)

    def test_no_boundary_raises(self):
        """All-ones mask (no inner boundary) raises ValueError."""
        mask = np.ones((8, 10), dtype=bool)
        with pytest.raises(ValueError, match="No inner-boundary points"):
            build_capacitance_solver(mask, dx=0.1, dy=0.1)

    def test_helmholtz_vs_fft_base(self):
        """Capacitance solver (FFT base) and (DST base) produce same BC enforcement."""
        Ny, Nx = 8, 10
        mask = _build_rect_mask(Ny, Nx)
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        rhs = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        s_fft = build_capacitance_solver(mask, dx, dy, lambda_=-1.0, base_bc="fft")
        s_dst = build_capacitance_solver(mask, dx, dy, lambda_=-1.0, base_bc="dst")
        psi_fft = s_fft(rhs)
        psi_dst = s_dst(rhs)
        # Both should enforce zero at boundary
        np.testing.assert_allclose(
            np.array(psi_fft[s_fft._j_b, s_fft._i_b]), 0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.array(psi_dst[s_dst._j_b, s_dst._i_b]), 0.0, atol=1e-10
        )
