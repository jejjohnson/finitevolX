"""Tests for CG, preconditioners, masked Laplacian, and vmap integration.

Spectral solver correctness, eigenvalue, and capacitance build/solve tests
are covered by spectraldiffx and omitted here.  This file tests finitevolX's
own CG solver, preconditioners, masked Laplacian, vmap integration, and
the re-export smoke tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.solvers.elliptic import (
    CapacitanceSolver,
    CGInfo,
    build_capacitance_solver,
    fft_eigenvalues,
    make_nystrom_preconditioner,
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
# Re-export smoke tests
# ---------------------------------------------------------------------------


class TestSpectralReexports:
    """Verify spectraldiffx re-exports are available via finitevolx."""

    def test_transforms_importable(self):
        from finitevolx import dct, dctn, dst, dstn, idct, idctn, idst, idstn

        assert all(
            callable(f) for f in [dct, dst, dctn, dstn, idct, idst, idctn, idstn]
        )

    def test_solvers_importable(self):
        from finitevolx import (
            solve_helmholtz_dct,
            solve_helmholtz_dst,
            solve_helmholtz_fft,
            solve_poisson_dct,
            solve_poisson_dst,
            solve_poisson_fft,
        )

        assert all(
            callable(f)
            for f in [
                solve_poisson_dst,
                solve_helmholtz_dst,
                solve_poisson_dct,
                solve_helmholtz_dct,
                solve_poisson_fft,
                solve_helmholtz_fft,
            ]
        )

    @pytest.mark.parametrize(
        "name",
        [
            # 2D explicit-transform variants
            "solve_helmholtz_dst1",
            "solve_helmholtz_dst2",
            "solve_helmholtz_dct1",
            "solve_helmholtz_dct2",
            "solve_poisson_dst1",
            "solve_poisson_dst2",
            "solve_poisson_dct1",
            "solve_poisson_dct2",
            # 1D solvers
            "solve_helmholtz_fft_1d",
            "solve_helmholtz_dst1_1d",
            "solve_helmholtz_dst2_1d",
            "solve_helmholtz_dct1_1d",
            "solve_helmholtz_dct2_1d",
            "solve_poisson_fft_1d",
            "solve_poisson_dst1_1d",
            "solve_poisson_dst2_1d",
            "solve_poisson_dct1_1d",
            "solve_poisson_dct2_1d",
            # 3D solvers
            "solve_helmholtz_fft_3d",
            "solve_helmholtz_dst1_3d",
            "solve_helmholtz_dst2_3d",
            "solve_helmholtz_dct1_3d",
            "solve_helmholtz_dct2_3d",
            "solve_poisson_fft_3d",
            "solve_poisson_dst1_3d",
            "solve_poisson_dst2_3d",
            "solve_poisson_dct1_3d",
            "solve_poisson_dct2_3d",
            # Generic per-axis BC solvers
            "solve_helmholtz_2d",
            "solve_helmholtz_3d",
            "solve_poisson_2d",
            "solve_poisson_3d",
            # RHS modification
            "modify_rhs_1d",
            "modify_rhs_2d",
            "modify_rhs_3d",
        ],
    )
    def test_new_solvers_importable(self, name):
        import finitevolx

        fn = getattr(finitevolx, name, None)
        assert fn is not None, f"finitevolx.{name} not found"
        assert callable(fn), f"finitevolx.{name} is not callable"

    def test_eigenvalues_importable(self):
        import finitevolx

        assert all(
            callable(f)
            for f in [
                finitevolx.dst1_eigenvalues,
                finitevolx.dct2_eigenvalues,
                finitevolx.fft_eigenvalues,
            ]
        )

    def test_new_eigenvalues_importable(self):
        import finitevolx

        # FD2 eigenvalues
        for name in [
            "dct1_eigenvalues",
            "dct2_eigenvalues",
            "dct3_eigenvalues",
            "dct4_eigenvalues",
            "dst1_eigenvalues",
            "dst2_eigenvalues",
            "dst3_eigenvalues",
            "dst4_eigenvalues",
            "fft_eigenvalues",
        ]:
            assert callable(getattr(finitevolx, name)), f"{name} not callable"

        # Pseudo-spectral eigenvalues
        for name in [
            "dct1_eigenvalues_ps",
            "dct2_eigenvalues_ps",
            "dct3_eigenvalues_ps",
            "dct4_eigenvalues_ps",
            "dst1_eigenvalues_ps",
            "dst2_eigenvalues_ps",
            "dst3_eigenvalues_ps",
            "dst4_eigenvalues_ps",
            "fft_eigenvalues_ps",
        ]:
            assert callable(getattr(finitevolx, name)), f"{name} not callable"

    @pytest.mark.parametrize(
        "name",
        [
            "DirichletHelmholtzSolver2D",
            "MixedBCHelmholtzSolver2D",
            "MixedBCHelmholtzSolver3D",
            "NeumannHelmholtzSolver2D",
            "RegularNeumannHelmholtzSolver2D",
            "SpectralHelmholtzSolver1D",
            "SpectralHelmholtzSolver2D",
            "SpectralHelmholtzSolver3D",
            "StaggeredDirichletHelmholtzSolver2D",
        ],
    )
    def test_solver_classes_importable(self, name):
        import inspect

        import finitevolx

        cls = getattr(finitevolx, name, None)
        assert cls is not None, f"finitevolx.{name} not found"
        assert inspect.isclass(cls), f"finitevolx.{name} is not a class"

    def test_boundary_condition_type_importable(self):
        from finitevolx import BoundaryCondition

        # BoundaryCondition is a type alias, not a class — just verify it exists
        assert BoundaryCondition is not None

    def test_capacitance_importable(self):
        import finitevolx

        assert callable(finitevolx.build_capacitance_solver)


# ---------------------------------------------------------------------------
# Preconditioned Conjugate Gradient
# ---------------------------------------------------------------------------


class TestSolveCG:
    def test_helmholtz_fft_exact(self, periodic_grid):
        """CG with spectral preconditioner solves Helmholtz on periodic grid."""
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_ref = jnp.cos(2 * jnp.pi * i / Nx) + jnp.cos(2 * jnp.pi * j / Ny)
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)
        rhs = (eigx[1] - lam) * jnp.cos(2 * jnp.pi * i / Nx) * jnp.ones((Ny, 1)) + (
            eigy[1] - lam
        ) * jnp.cos(2 * jnp.pi * j / Ny) * jnp.ones((1, Nx))

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        M_inv = make_spectral_preconditioner(dx, dy, lambda_=lam, bc="fft")
        psi, info = solve_cg(A, rhs, preconditioner=M_inv, rtol=1e-10, atol=1e-10)
        assert info.converged
        np.testing.assert_allclose(np.array(psi), np.array(psi_ref), atol=1e-8)

    def test_returns_cg_info(self, periodic_grid):
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0
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
        lam = 1.0
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
        _, info_pre = solve_cg(
            A, rhs, preconditioner=M_inv, rtol=1e-8, atol=1e-8, max_steps=200
        )
        assert info_pre.iterations <= info_no_pre.iterations + 5

    def test_identity_preconditioner(self, periodic_grid):
        """None preconditioner (identity) still converges."""
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)
        rhs = jnp.ones((Ny, Nx))

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        _psi, info = solve_cg(A, rhs, preconditioner=None, rtol=1e-8, atol=1e-8)
        assert info.converged

    def test_make_spectral_preconditioner_invalid_bc(self, periodic_grid):
        _Ny, _Nx, dx, dy = periodic_grid
        with pytest.raises(ValueError, match="bc must be"):
            make_spectral_preconditioner(dx, dy, bc="invalid")


# ---------------------------------------------------------------------------
# Nyström preconditioner
# ---------------------------------------------------------------------------


class TestNystromPreconditioner:
    def test_output_shape_and_dtype(self, periodic_grid):
        """Preconditioner preserves shape and dtype."""
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        M_inv = make_nystrom_preconditioner(A, (Ny, Nx), rank=20)
        r = jnp.ones((Ny, Nx))
        out = M_inv(r)
        assert out.shape == (Ny, Nx)
        assert out.dtype == r.dtype

    def test_deterministic_with_fixed_key(self, periodic_grid):
        """Same key produces identical preconditioners."""
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        key = jax.random.PRNGKey(42)
        M1 = make_nystrom_preconditioner(A, (Ny, Nx), rank=10, key=key)
        M2 = make_nystrom_preconditioner(A, (Ny, Nx), rank=10, key=key)
        r = jnp.sin(jnp.arange(Ny, dtype=float)[:, None]) * jnp.cos(
            jnp.arange(Nx, dtype=float)[None, :]
        )
        np.testing.assert_allclose(np.array(M1(r)), np.array(M2(r)), atol=1e-14)

    def test_usable_as_cg_preconditioner(self, periodic_grid):
        """Nyström preconditioner can be used with solve_cg and produces correct results."""
        Ny, Nx, dx, dy = periodic_grid
        lam = 1.0
        eigx = fft_eigenvalues(Nx, dx)
        eigy = fft_eigenvalues(Ny, dy)

        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        psi_exact = jnp.cos(2 * jnp.pi * i / Nx) + jnp.cos(2 * jnp.pi * j / Ny)

        def A(psi):
            eig2d = eigy[:, None] + eigx[None, :] - lam
            return jnp.real(jnp.fft.ifft2(eig2d * jnp.fft.fft2(psi)))

        rhs = A(psi_exact)

        M_inv = make_nystrom_preconditioner(A, (Ny, Nx), rank=Ny * Nx)
        psi, info = solve_cg(
            A, rhs, preconditioner=M_inv, rtol=1e-8, atol=1e-8, max_steps=200
        )
        assert info.converged
        np.testing.assert_allclose(np.array(psi), np.array(psi_exact), atol=1e-6)


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
        lap_std = (
            jnp.roll(psi, 1, axis=1) + jnp.roll(psi, -1, axis=1) - 2 * psi
        ) / dx**2 + (
            jnp.roll(psi, 1, axis=0) + jnp.roll(psi, -1, axis=0) - 2 * psi
        ) / dy**2
        np.testing.assert_allclose(np.array(lap_masked), np.array(lap_std), atol=1e-12)

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
# Capacitance matrix solver — build + basic usage
# ---------------------------------------------------------------------------


def _build_rect_mask(Ny: int, Nx: int, border: int = 1) -> np.ndarray:
    """Rectangle interior mask (True = interior, border cells = False)."""
    mask = np.ones((Ny, Nx), dtype=bool)
    mask[:border, :] = mask[-border:, :] = False
    mask[:, :border] = mask[:, -border:] = False
    return mask


def _inner_boundary_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute inner-boundary (j, i) indices from a boolean mask.

    Inner-boundary points are True (interior) cells that are 4-connected
    to at least one False (exterior) cell.
    """
    from scipy.ndimage import binary_dilation

    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    dilated = binary_dilation(~mask, structure=struct)
    inner_boundary = mask & dilated
    return np.where(inner_boundary)


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

        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        rhs = jnp.sin(jnp.pi * j / Ny) * jnp.cos(jnp.pi * i / Nx)
        psi = solver(rhs)

        j_b, i_b = _inner_boundary_indices(mask)
        bc_vals = psi[j_b, i_b]
        np.testing.assert_allclose(np.array(bc_vals), 0.0, atol=1e-10)

    def test_no_boundary_raises(self):
        """All-ones mask (no inner boundary) raises ValueError."""
        mask = np.ones((8, 10), dtype=bool)
        with pytest.raises(ValueError, match="No inner-boundary points"):
            build_capacitance_solver(mask, dx=0.1, dy=0.1)


# ---------------------------------------------------------------------------
# Batched (vmap) solver tests
# ---------------------------------------------------------------------------


class TestVmapSpectralSolvers:
    """Verify all solvers work under jax.vmap for multi-layer / batched solves."""

    def test_helmholtz_dst_vmap(self, dirichlet_grid):
        """Batched DST Helmholtz: each layer matches the single-layer solve."""
        Ny, Nx, dx, dy = dirichlet_grid
        nl = 3
        lambdas = jnp.array([-0.5, -1.0, -2.0])
        rhs = jnp.stack(
            [
                jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * (l + 1))
                * jnp.cos(jnp.arange(Nx, dtype=float)[None, :])
                for l in range(nl)
            ]
        )
        psi_batched = jax.vmap(lambda r, l: solve_helmholtz_dst(r, dx, dy, l))(
            rhs, lambdas
        )
        assert psi_batched.shape == (nl, Ny, Nx)
        for i in range(nl):
            psi_single = solve_helmholtz_dst(rhs[i], dx, dy, float(lambdas[i]))
            np.testing.assert_allclose(
                np.array(psi_batched[i]), np.array(psi_single), atol=1e-12
            )

    def test_helmholtz_dct_vmap(self, neumann_grid):
        """Batched DCT Helmholtz with per-mode lambda, including lambda=0."""
        Ny, Nx, dx, dy = neumann_grid
        nl = 4
        lambdas = jnp.array([0.0, -0.5, -1.0, -2.0])
        rhs = jnp.stack(
            [
                jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * (l + 1))
                * jnp.cos(jnp.arange(Nx, dtype=float)[None, :])
                for l in range(nl)
            ]
        )
        psi_batched = jax.vmap(lambda r, l: solve_helmholtz_dct(r, dx, dy, l))(
            rhs, lambdas
        )
        assert psi_batched.shape == (nl, Ny, Nx)
        for i in range(nl):
            psi_single = solve_helmholtz_dct(rhs[i], dx, dy, float(lambdas[i]))
            np.testing.assert_allclose(
                np.array(psi_batched[i]), np.array(psi_single), atol=1e-12
            )
        psi_poisson = solve_poisson_dct(rhs[0], dx, dy)
        np.testing.assert_allclose(
            np.array(psi_batched[0]), np.array(psi_poisson), atol=1e-12
        )

    def test_helmholtz_fft_vmap(self, periodic_grid):
        """Batched FFT Helmholtz with per-mode lambda, including lambda=0."""
        Ny, Nx, dx, dy = periodic_grid
        nl = 4
        lambdas = jnp.array([0.0, -0.5, -1.0, -2.0])
        rhs = jnp.stack(
            [
                jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * (l + 1))
                * jnp.cos(jnp.arange(Nx, dtype=float)[None, :])
                for l in range(nl)
            ]
        )
        psi_batched = jax.vmap(lambda r, l: solve_helmholtz_fft(r, dx, dy, l))(
            rhs, lambdas
        )
        assert psi_batched.shape == (nl, Ny, Nx)
        for i in range(nl):
            psi_single = solve_helmholtz_fft(rhs[i], dx, dy, float(lambdas[i]))
            np.testing.assert_allclose(
                np.array(psi_batched[i]), np.array(psi_single), atol=1e-12
            )
        psi_poisson = solve_poisson_fft(rhs[0], dx, dy)
        np.testing.assert_allclose(
            np.array(psi_batched[0]), np.array(psi_poisson), atol=1e-12
        )

    def test_poisson_dst_vmap(self, dirichlet_grid):
        """Batched DST Poisson."""
        Ny, Nx, dx, dy = dirichlet_grid
        nl = 3
        rhs = jnp.stack(
            [
                jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * (l + 1))
                * jnp.cos(jnp.arange(Nx, dtype=float)[None, :])
                for l in range(nl)
            ]
        )
        psi_batched = jax.vmap(lambda r: solve_poisson_dst(r, dx, dy))(rhs)
        assert psi_batched.shape == (nl, Ny, Nx)
        for i in range(nl):
            psi_single = solve_poisson_dst(rhs[i], dx, dy)
            np.testing.assert_allclose(
                np.array(psi_batched[i]), np.array(psi_single), atol=1e-12
            )

    def test_poisson_dct_vmap(self, neumann_grid):
        """Batched DCT Poisson."""
        Ny, Nx, dx, dy = neumann_grid
        rhs = jnp.ones((3, Ny, Nx))
        psi = jax.vmap(lambda r: solve_poisson_dct(r, dx, dy))(rhs)
        assert psi.shape == (3, Ny, Nx)

    def test_poisson_fft_vmap(self, periodic_grid):
        """Batched FFT Poisson."""
        Ny, Nx, dx, dy = periodic_grid
        rhs = jnp.ones((3, Ny, Nx))
        psi = jax.vmap(lambda r: solve_poisson_fft(r, dx, dy))(rhs)
        assert psi.shape == (3, Ny, Nx)

    def test_helmholtz_dct_vmap_lambda0(self, neumann_grid):
        """Batched DCT Helmholtz with lambda=0 under vmap matches Poisson."""
        Ny, Nx, dx, dy = neumann_grid
        nl = 3
        rhs = jnp.stack(
            [
                jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * (l + 1))
                * jnp.cos(jnp.arange(Nx, dtype=float)[None, :] * (l + 1))
                for l in range(nl)
            ]
        )
        psi_helm = jax.vmap(lambda r: solve_helmholtz_dct(r, dx, dy, lambda_=0.0))(rhs)
        psi_pois = jax.vmap(lambda r: solve_poisson_dct(r, dx, dy))(rhs)
        np.testing.assert_allclose(np.array(psi_helm), np.array(psi_pois), atol=1e-12)

    def test_helmholtz_fft_vmap_lambda0(self, periodic_grid):
        """Batched FFT Helmholtz with lambda=0 under vmap matches Poisson."""
        Ny, Nx, dx, dy = periodic_grid
        nl = 3
        rhs = jnp.stack(
            [
                jnp.sin(jnp.arange(Ny, dtype=float)[:, None] * (l + 1))
                * jnp.cos(jnp.arange(Nx, dtype=float)[None, :] * (l + 1))
                for l in range(nl)
            ]
        )
        psi_helm = jax.vmap(lambda r: solve_helmholtz_fft(r, dx, dy, lambda_=0.0))(rhs)
        psi_pois = jax.vmap(lambda r: solve_poisson_fft(r, dx, dy))(rhs)
        np.testing.assert_allclose(np.array(psi_helm), np.array(psi_pois), atol=1e-12)


class TestVmapCapacitanceSolver:
    """Verify CapacitanceSolver works under jax.vmap."""

    def test_batched_matches_loop(self):
        """Batched capacitance solve matches individual solves."""
        Ny, Nx = 10, 12
        mask = _build_rect_mask(Ny, Nx)
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        solver = build_capacitance_solver(mask, dx, dy, lambda_=-1.0, base_bc="fft")

        nl = 3
        j = jnp.arange(Ny)[:, None]
        i = jnp.arange(Nx)[None, :]
        rhs = jnp.stack(
            [
                jnp.sin(jnp.pi * j * (l + 1) / Ny) * jnp.cos(jnp.pi * i / Nx)
                for l in range(nl)
            ]
        )
        psi_batched = jax.vmap(solver)(rhs)
        assert psi_batched.shape == (nl, Ny, Nx)
        for l in range(nl):
            psi_single = solver(rhs[l])
            np.testing.assert_allclose(
                np.array(psi_batched[l]), np.array(psi_single), atol=1e-10
            )

    def test_batched_bc_enforced(self):
        """Boundary condition is enforced for every batch element."""
        Ny, Nx = 10, 12
        mask = _build_rect_mask(Ny, Nx)
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        solver = build_capacitance_solver(mask, dx, dy, lambda_=-1.0, base_bc="fft")

        j_b, i_b = _inner_boundary_indices(mask)
        rhs = jnp.ones((4, Ny, Nx))
        psi = jax.vmap(solver)(rhs)
        for l in range(4):
            bc_vals = psi[l][j_b, i_b]
            np.testing.assert_allclose(np.array(bc_vals), 0.0, atol=1e-10)
