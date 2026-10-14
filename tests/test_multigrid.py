"""Tests for the geometric multigrid Helmholtz solver."""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from finitevolx._src.solvers.multigrid import (
    MultigridSolver,
    _apply_operator,
    _compute_n_levels,
    _prolongate,
    _restrict,
    _weighted_jacobi,
    build_multigrid_solver,
)
from finitevolx._src.solvers.preconditioners import (
    make_gaussx_multigrid_preconditioner,
    make_multigrid_preconditioner,
)

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
    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    def test_is_gaussx_preconditioner(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        pc = make_gaussx_multigrid_preconditioner(_all_ones_solver(ny, nx, dx, dy))
        assert isinstance(pc, gaussx.OperatorPreconditioner)

    @staticmethod
    def _dense(pc, n, dtype=jnp.float64):
        """Dense M^-1 of the preconditioner through its lineax operator."""
        system = lx.FunctionLinearOperator(
            lambda v: v, jax.ShapeDtypeStruct((n,), dtype), tags=[lx.symmetric_tag]
        )
        op = pc.as_operator(system)
        return np.asarray(jax.vmap(op.mv)(jnp.eye(n, dtype=dtype)).T)

    @staticmethod
    def _dense_raw_and_k(solver, ny, nx):
        """Dense V = -Vcycle(0, .) and K = -A on the full [Ny*Nx] grid."""
        level0 = solver.levels[0]
        fvx_pc = make_multigrid_preconditioner(solver)
        eye = jnp.eye(ny * nx)
        V = -np.asarray(jax.vmap(lambda e: fvx_pc(e.reshape(ny, nx)).ravel())(eye)).T
        K = -np.asarray(
            jax.vmap(lambda e: _apply_operator(e.reshape(ny, nx), level0).ravel())(eye)
        ).T
        return V, K

    @pytest.mark.parametrize("masked", [False, True])
    def test_symmetric_positive_definite(self, masked):
        """CG needs a symmetric positive definite M^-1. One V-cycle V is not
        symmetric (restriction is not the transpose of prolongation); the
        preconditioner is the symmetrised two-step iteration
        M^-1 = V + V^T - V^T K V, symmetric to round-off and, for a
        convergent multigrid (default settings), positive definite on the
        wet points."""
        ny, nx = 16, 16
        dx, dy = 1.0 / nx, 1.0 / ny
        mask = _circular_mask(ny, nx) if masked else np.ones((ny, nx))
        solver = build_multigrid_solver(mask, dx, dy, lambda_=1.0)
        wet = np.asarray(solver.levels[0].mask).ravel().astype(bool)
        V, K = self._dense_raw_and_k(solver, ny, nx)
        Vw = V[np.ix_(wet, wet)]
        # the raw V-cycle is measurably non-symmetric (why this exists)
        assert np.linalg.norm(Vw - Vw.T) > 1e-3 * np.linalg.norm(Vw)

        M = self._dense(make_gaussx_multigrid_preconditioner(solver), ny * nx)
        np.testing.assert_allclose(
            M, V + V.T - V.T @ K @ V, rtol=1e-9, atol=1e-12 * np.abs(M).max()
        )
        M = M[np.ix_(wet, wet)]
        np.testing.assert_allclose(M, M.T, rtol=0, atol=1e-12 * np.abs(M).max())
        assert np.linalg.eigvalsh(M).min() > 0.0

    def test_gaussx_cg_matches_solve_cg(self):
        """gaussx.CGSolver with the multigrid preconditioner solves the
        positive system (−A) x = −b on a masked basin to the same answer as
        finitevolX's solve_cg."""
        from finitevolx._src.solvers.iterative import solve_cg

        ny, nx = 32, 32
        dx, dy = 1.0 / nx, 1.0 / ny
        lam = 10.0
        mask = _circular_mask(ny, nx)
        solver = build_multigrid_solver(mask, dx, dy, lambda_=lam)
        pc = make_gaussx_multigrid_preconditioner(solver)
        level0 = solver.levels[0]

        def matvec(u):
            return _apply_operator(u, level0)

        rng = np.random.default_rng(13)
        rhs = jnp.array(rng.standard_normal((ny, nx))) * level0.mask

        x_fvx, info = solve_cg(
            matvec,
            rhs,
            preconditioner=make_multigrid_preconditioner(solver),
            rtol=1e-10,
            atol=1e-10,
        )
        assert info.converged

        neg_A = gaussx.as_linear_operator(
            lambda v: -matvec(v.reshape(ny, nx)).ravel(),
            shape=(ny * nx, ny * nx),
            positive_semidefinite=True,
        )
        cg = gaussx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=200, preconditioner=pc)
        x_gx = cg.solve(neg_A, -rhs.ravel()).reshape(ny, nx)
        np.testing.assert_allclose(
            np.asarray(x_gx * level0.mask),
            np.asarray(x_fvx * level0.mask),
            rtol=1e-7,
            atol=1e-9,
        )

    def test_preconditioner_reduces_gaussx_cg_steps(self):
        """The symmetrised V-cycle cuts lineax CG steps on the positive system,
        relative to no preconditioner and to the raw (non-symmetric) V-cycle."""
        ny, nx = 32, 32
        dx, dy = 1.0 / nx, 1.0 / ny
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=1.0)
        pc = make_gaussx_multigrid_preconditioner(solver)
        level0 = solver.levels[0]
        structure = jax.ShapeDtypeStruct((ny * nx,), jnp.float64)
        neg_A = lx.FunctionLinearOperator(
            lambda v: -_apply_operator(v.reshape(ny, nx), level0).ravel(),
            structure,
            tags=[lx.positive_semidefinite_tag],
        )
        fvx_pc = make_multigrid_preconditioner(solver)
        raw = lx.FunctionLinearOperator(
            lambda v: -fvx_pc(v.reshape(ny, nx)).ravel(),
            structure,
            tags=[lx.positive_semidefinite_tag],
        )
        b = jnp.array(np.random.default_rng(5).standard_normal(ny * nx))
        cg = lx.CG(rtol=1e-8, atol=1e-8, max_steps=2000)

        def steps(precond):
            opts = {} if precond is None else {"preconditioner": precond}
            sol = lx.linear_solve(neg_A, b, cg, options=opts)
            return int(sol.stats["num_steps"])

        n_plain, n_raw, n_sym = steps(None), steps(raw), steps(pc.as_operator(neg_A))
        assert n_sym < n_plain // 4
        assert n_sym < n_raw

    def test_follows_vector_dtype(self):
        """The V-cycle runs in the float64 hierarchy, but a float32 system gets
        a float32 operator and a float32 solve, and a direct call returns the
        vector's dtype (codex review on #340)."""
        ny, nx = 32, 32
        dx, dy = 1.0 / nx, 1.0 / ny
        solver = _all_ones_solver(ny, nx, dx, dy, lambda_=1.0)
        level0 = solver.levels[0]
        neg_A = lx.FunctionLinearOperator(
            lambda v: (
                -_apply_operator(v.reshape(ny, nx), level0).ravel().astype(jnp.float32)
            ),
            jax.ShapeDtypeStruct((ny * nx,), jnp.float32),
            tags=[lx.positive_semidefinite_tag],
        )
        pc = make_gaussx_multigrid_preconditioner(solver)
        pc_op = pc.as_operator(neg_A)
        assert pc_op.in_structure() == neg_A.in_structure()
        b = jnp.asarray(
            np.random.default_rng(7).standard_normal(ny * nx), dtype=jnp.float32
        )
        assert pc(b, neg_A).dtype == jnp.float32
        sol = lx.linear_solve(
            neg_A,
            b,
            lx.CG(rtol=1e-4, atol=1e-4, max_steps=200),
            options={"preconditioner": pc_op},
        )
        assert sol.value.dtype == jnp.float32
        assert sol.result == lx.RESULTS.successful


# ---------------------------------------------------------------------------
# TestJAXCompat
# ---------------------------------------------------------------------------


class TestJAXCompat:
    @pytest.mark.slow
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

    @pytest.mark.slow
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
    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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
