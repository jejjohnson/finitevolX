"""Tests for the geometric multigrid Helmholtz solver."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from finitevolx._src.solvers.multigrid import (
    MultigridSolver,
    _apply_operator,
    _compute_n_levels,
    _prolongate,
    _restrict,
    _weighted_jacobi,
    build_multigrid_solver,
)
from finitevolx._src.solvers.preconditioners import make_multigrid_preconditioner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mg_grid_16():
    """16x16 square grid."""
    ny, nx = 16, 16
    dx = 1.0 / nx
    dy = 1.0 / ny
    return ny, nx, dx, dy


@pytest.fixture
def mg_grid_32():
    """32x32 square grid (3 multigrid levels with min_coarse=8)."""
    ny, nx = 32, 32
    dx = 1.0 / nx
    dy = 1.0 / ny
    return ny, nx, dx, dy


@pytest.fixture
def mg_grid_rect():
    """16x32 rectangular grid."""
    ny, nx = 16, 32
    dx = 1.0 / nx
    dy = 1.0 / ny
    return ny, nx, dx, dy


def _all_ones_solver(ny, nx, dx, dy, lambda_=0.0, **kwargs):
    """Build a solver on a fully rectangular (all-ones mask) domain."""
    mask = np.ones((ny, nx), dtype=np.float64)
    return build_multigrid_solver(mask, dx, dy, lambda_=lambda_, **kwargs)


def _circular_mask(ny, nx, radius_frac=0.4):
    """Create a circular mask centred in the domain."""
    jj, ii = np.meshgrid(
        np.linspace(-0.5, 0.5, ny), np.linspace(-0.5, 0.5, nx), indexing="ij"
    )
    return (jj**2 + ii**2 < radius_frac**2).astype(np.float64)


# ---------------------------------------------------------------------------
# TestBuildMultigridSolver
# ---------------------------------------------------------------------------


class TestBuildMultigridSolver:
    def test_returns_solver(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy)
        assert isinstance(solver, MultigridSolver)

    def test_n_levels_auto(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy)
        expected = _compute_n_levels(ny, nx)
        assert solver.n_levels == expected

    def test_n_levels_manual(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy, n_levels=2)
        assert solver.n_levels == 2

    def test_non_divisible_raises(self):
        mask = np.ones((15, 15))
        with pytest.raises(ValueError, match="not divisible"):
            build_multigrid_solver(mask, 1.0, 1.0, n_levels=2)

    def test_variable_coefficient(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        mask = np.ones((ny, nx))
        coeff = np.random.default_rng(42).uniform(0.5, 2.0, (ny, nx))
        solver = build_multigrid_solver(mask, dx, dy, coeff=coeff)
        assert isinstance(solver, MultigridSolver)

    def test_level_shapes(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy)
        for i, level in enumerate(solver.levels):
            factor = 2**i
            assert level.mask.shape == (ny // factor, nx // factor)


# ---------------------------------------------------------------------------
# TestApplyOperator
# ---------------------------------------------------------------------------


class TestApplyOperator:
    def test_zero_field(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy)
        u = jnp.zeros((ny, nx))
        result = _apply_operator(u, solver.levels[0])
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-15)

    def test_linearity(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy)
        level = solver.levels[0]
        rng = np.random.default_rng(0)
        u = jnp.array(rng.standard_normal((ny, nx)))
        v = jnp.array(rng.standard_normal((ny, nx)))
        a, b = 2.5, -1.3
        Au = _apply_operator(u, level)
        Av = _apply_operator(v, level)
        Aab = _apply_operator(a * u + b * v, level)
        np.testing.assert_allclose(
            np.array(Aab), np.array(a * Au + b * Av), rtol=1e-5, atol=1e-3
        )

    def test_masked_output_zero_outside(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        mask = _circular_mask(ny, nx)
        solver = build_multigrid_solver(mask, dx, dy, n_levels=1)
        level = solver.levels[0]
        u = jnp.ones((ny, nx))
        result = _apply_operator(u, level)
        outside = np.array(mask) == 0
        np.testing.assert_allclose(np.array(result)[outside], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestGridTransfer
# ---------------------------------------------------------------------------


class TestGridTransfer:
    def test_restrict_constant(self, mg_grid_16):
        ny, nx, _, _ = mg_grid_16
        mask_f = jnp.ones((ny, nx))
        mask_c = jnp.ones((ny // 2, nx // 2))
        v = jnp.full((ny, nx), 3.14)
        result = _restrict(v, mask_f, mask_c)
        np.testing.assert_allclose(np.array(result), 3.14, atol=1e-12)

    def test_restrict_shape(self, mg_grid_16):
        ny, nx, _, _ = mg_grid_16
        mask_f = jnp.ones((ny, nx))
        mask_c = jnp.ones((ny // 2, nx // 2))
        v = jnp.ones((ny, nx))
        result = _restrict(v, mask_f, mask_c)
        assert result.shape == (ny // 2, nx // 2)

    def test_prolong_constant(self, mg_grid_16):
        ny, nx, _, _ = mg_grid_16
        ny_c, nx_c = ny // 2, nx // 2
        mask_c = jnp.ones((ny_c, nx_c))
        mask_f = jnp.ones((ny, nx))
        v = jnp.full((ny_c, nx_c), 2.71)
        result = _prolongate(v, mask_c, mask_f)
        np.testing.assert_allclose(np.array(result), 2.71, atol=1e-12)

    def test_prolong_shape(self, mg_grid_16):
        ny, nx, _, _ = mg_grid_16
        ny_c, nx_c = ny // 2, nx // 2
        mask_c = jnp.ones((ny_c, nx_c))
        mask_f = jnp.ones((ny, nx))
        v = jnp.ones((ny_c, nx_c))
        result = _prolongate(v, mask_c, mask_f)
        assert result.shape == (ny, nx)

    def test_masked_restrict_no_contamination(self, mg_grid_16):
        ny, nx, _, _ = mg_grid_16
        mask_f = jnp.ones((ny, nx)).at[:, : nx // 2].set(0.0)
        mask_c = jnp.ones((ny // 2, nx // 2)).at[:, : nx // 4].set(0.0)
        v = jnp.ones((ny, nx)) * 5.0
        result = _restrict(v, mask_f, mask_c)
        outside = np.array(mask_c) == 0
        np.testing.assert_allclose(np.array(result)[outside], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestJacobiSmoother
# ---------------------------------------------------------------------------


class TestJacobiSmoother:
    def test_reduces_residual(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy)
        level = solver.levels[0]
        rng = np.random.default_rng(1)
        rhs = jnp.array(rng.standard_normal((ny, nx)))
        u0 = jnp.zeros((ny, nx))

        r_before = jnp.linalg.norm(rhs - _apply_operator(u0, level))
        u1 = _weighted_jacobi(u0, rhs, level, omega=0.95, n_iters=10)
        r_after = jnp.linalg.norm(rhs - _apply_operator(u1, level))
        assert float(r_after) < float(r_before)


# ---------------------------------------------------------------------------
# TestVCycleSolver
# ---------------------------------------------------------------------------


class TestVCycleSolver:
    def test_poisson_convergence(self, mg_grid_32):
        """Constant-coeff Poisson on a full rectangle: residual decreases.

        Note: λ=0 Neumann Laplacian is singular (constant null mode),
        so convergence is slower than Helmholtz.
        """
        ny, nx, dx, dy = mg_grid_32
        solver = _all_ones_solver(ny, nx, dx, dy, n_cycles=20)
        rng = np.random.default_rng(10)
        rhs = jnp.array(rng.standard_normal((ny, nx)))
        u = solver(rhs)
        residual = jnp.linalg.norm(rhs - _apply_operator(u, solver.levels[0]))
        rhs_norm = jnp.linalg.norm(rhs)
        assert float(residual / rhs_norm) < 0.1

    def test_helmholtz_convergence(self, mg_grid_32):
        """Helmholtz with lambda > 0 converges well."""
        ny, nx, dx, dy = mg_grid_32
        lam = 10.0
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=lam, n_cycles=20)
        rng = np.random.default_rng(11)
        rhs = jnp.array(rng.standard_normal((ny, nx)))
        u = solver(rhs)
        residual = jnp.linalg.norm(rhs - _apply_operator(u, solver.levels[0]))
        rhs_norm = jnp.linalg.norm(rhs)
        assert float(residual / rhs_norm) < 1e-2

    def test_variable_coefficient(self, mg_grid_32):
        """Variable-coefficient problem: residual decreases."""
        ny, nx, dx, dy = mg_grid_32
        mask = np.ones((ny, nx))
        rng = np.random.default_rng(42)
        coeff = rng.uniform(0.5, 2.0, (ny, nx))
        solver = build_multigrid_solver(mask, dx, dy, coeff=coeff, n_cycles=20)
        rhs = jnp.array(rng.standard_normal((ny, nx)))
        u = solver(rhs)
        residual = jnp.linalg.norm(rhs - _apply_operator(u, solver.levels[0]))
        rhs_norm = jnp.linalg.norm(rhs)
        assert float(residual / rhs_norm) < 0.1

    def test_masked_domain_zero_outside(self, mg_grid_16):
        """Masked domain: solution is zero outside mask."""
        ny, nx, dx, dy = mg_grid_16
        mask = _circular_mask(ny, nx)
        solver = build_multigrid_solver(mask, dx, dy, n_levels=1, n_cycles=10)
        rng = np.random.default_rng(7)
        rhs = jnp.array(rng.standard_normal((ny, nx))) * jnp.array(mask)
        u = solver(rhs)
        outside = np.array(mask) == 0
        np.testing.assert_allclose(np.array(u)[outside], 0.0, atol=1e-15)

    def test_rectangular_grid(self, mg_grid_rect):
        """Non-square grid works correctly (Helmholtz, λ > 0)."""
        ny, nx, dx, dy = mg_grid_rect
        lam = 10.0
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=lam, n_cycles=20)
        rng = np.random.default_rng(12)
        rhs = jnp.array(rng.standard_normal((ny, nx)))
        u = solver(rhs)
        residual = jnp.linalg.norm(rhs - _apply_operator(u, solver.levels[0]))
        rhs_norm = jnp.linalg.norm(rhs)
        assert float(residual / rhs_norm) < 1e-2

    def test_single_vcycle_reduces_residual(self, mg_grid_16):
        """A single V-cycle reduces the residual."""
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy)
        rng = np.random.default_rng(3)
        rhs = jnp.array(rng.standard_normal((ny, nx)))
        u0 = jnp.zeros((ny, nx))

        r_before = jnp.linalg.norm(rhs - _apply_operator(u0, solver.levels[0]))
        u1 = solver.v_cycle(u0, rhs)
        r_after = jnp.linalg.norm(rhs - _apply_operator(u1, solver.levels[0]))
        assert float(r_after) < float(r_before)


# ---------------------------------------------------------------------------
# TestMultigridPreconditioner
# ---------------------------------------------------------------------------


class TestMultigridPreconditioner:
    def test_preconditioner_output_shape(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy)
        pc = make_multigrid_preconditioner(solver)
        r = jnp.ones((ny, nx))
        out = pc(r)
        assert out.shape == (ny, nx)

    def test_preconditioner_with_cg(self, mg_grid_16):
        """Multigrid-preconditioned CG converges for Helmholtz (lambda > 0)."""
        from finitevolx._src.solvers.iterative import solve_cg

        ny, nx, dx, dy = mg_grid_16
        lam = 10.0
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=lam)
        pc = make_multigrid_preconditioner(solver)

        rng = np.random.default_rng(13)
        rhs = jnp.array(rng.standard_normal((ny, nx)))

        def matvec(u):
            return _apply_operator(u, solver.levels[0])

        _x, info = solve_cg(matvec, rhs, preconditioner=pc, max_steps=100)
        assert info.converged


# ---------------------------------------------------------------------------
# TestJAXCompat
# ---------------------------------------------------------------------------


class TestJAXCompat:
    def test_jit_solve(self, mg_grid_16):
        """eqx.filter_jit produces the same result as eager."""
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy, n_cycles=3)
        rng = np.random.default_rng(5)
        rhs = jnp.array(rng.standard_normal((ny, nx)))

        u_eager = solver(rhs)

        @eqx.filter_jit
        def _solve(s, r):
            return s(r)

        u_jit = _solve(solver, rhs)
        np.testing.assert_allclose(np.array(u_eager), np.array(u_jit), atol=1e-12)

    def test_vmap_solve(self, mg_grid_16):
        """jax.vmap(solver) over a batch dimension."""
        ny, nx, dx, dy = mg_grid_16
        solver = _all_ones_solver(ny, nx, dx, dy, n_cycles=3)
        rng = np.random.default_rng(6)
        batch_rhs = jnp.array(rng.standard_normal((3, ny, nx)))

        u_batch = jax.vmap(solver)(batch_rhs)
        assert u_batch.shape == (3, ny, nx)

        for i in range(3):
            u_single = solver(batch_rhs[i])
            np.testing.assert_allclose(
                np.array(u_batch[i]), np.array(u_single), atol=1e-12
            )


# ---------------------------------------------------------------------------
# TestImplicitDifferentiation
# ---------------------------------------------------------------------------


class TestImplicitDifferentiation:
    def test_grad_through_implicit_solve(self, mg_grid_16):
        """jax.grad works through __call__ (implicit differentiation).

        Verifies that the backward pass produces finite, non-zero gradients
        via the adjoint equation rather than unrolling through V-cycles.
        """
        ny, nx, dx, dy = mg_grid_16
        lam = 10.0
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=lam, n_cycles=5)

        def loss_fn(rhs):
            u = solver(rhs)
            return jnp.sum(u**2)

        rng = np.random.default_rng(20)
        rhs = jnp.array(rng.standard_normal((ny, nx)))
        grad_rhs = jax.grad(loss_fn)(rhs)

        assert grad_rhs.shape == rhs.shape
        assert jnp.all(jnp.isfinite(grad_rhs))
        assert float(jnp.linalg.norm(grad_rhs)) > 0

    def test_all_three_forward_match(self, mg_grid_16):
        """All three solve modes produce the same forward solution."""
        ny, nx, dx, dy = mg_grid_16
        lam = 10.0
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=lam, n_cycles=5)
        rng = np.random.default_rng(21)
        rhs = jnp.array(rng.standard_normal((ny, nx)))

        u_implicit = solver(rhs)
        u_onestep = solver.solve_onestep(rhs)
        u_unrolled = solver.solve_unrolled(rhs)
        np.testing.assert_allclose(
            np.array(u_implicit), np.array(u_unrolled), atol=1e-12
        )
        np.testing.assert_allclose(np.array(u_implicit), np.array(u_onestep), atol=1e-6)

    def test_grad_onestep(self, mg_grid_16):
        """One-step differentiation (Bolte et al. 2023) produces finite grads."""
        ny, nx, dx, dy = mg_grid_16
        lam = 10.0
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=lam, n_cycles=5)

        def loss_fn(rhs):
            u = solver.solve_onestep(rhs)
            return jnp.sum(u**2)

        rng = np.random.default_rng(22)
        rhs = jnp.array(rng.standard_normal((ny, nx)))
        grad_rhs = jax.grad(loss_fn)(rhs)

        assert grad_rhs.shape == rhs.shape
        assert jnp.all(jnp.isfinite(grad_rhs))
        assert float(jnp.linalg.norm(grad_rhs)) > 0

    def test_onestep_grad_close_to_implicit(self, mg_grid_16):
        """One-step gradient approximates the implicit (IFT) gradient.

        For a well-converged solver the error should be O(ρ) where ρ is
        the V-cycle convergence rate.
        """
        ny, nx, dx, dy = mg_grid_16
        lam = 10.0
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=lam, n_cycles=10)

        rng = np.random.default_rng(23)
        rhs = jnp.array(rng.standard_normal((ny, nx)))

        def loss_implicit(r):
            return jnp.sum(solver(r) ** 2)

        def loss_onestep(r):
            return jnp.sum(solver.solve_onestep(r) ** 2)

        g_implicit = jax.grad(loss_implicit)(rhs)
        g_onestep = jax.grad(loss_onestep)(rhs)

        # One-step should be in the same ballpark (relative error < 1)
        rel_error = float(
            jnp.linalg.norm(g_onestep - g_implicit) / jnp.linalg.norm(g_implicit)
        )
        assert rel_error < 1.0

    def test_grad_implicit_is_finite(self, mg_grid_16):
        """Gradient w.r.t. a scalar parameter via implicit diff is finite."""
        ny, nx, dx, dy = mg_grid_16
        lam = 10.0
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=lam, n_cycles=10)

        def loss_fn(rhs_scale):
            rhs = rhs_scale * jnp.ones((ny, nx))
            u = solver(rhs)
            return jnp.mean(u)

        grad_val = jax.grad(loss_fn)(1.0)
        assert jnp.isfinite(grad_val)
        assert float(jnp.abs(grad_val)) > 0
