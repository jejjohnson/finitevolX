"""Tests for the geometric multigrid Helmholtz solver."""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from finitevolx import Mask2D, masked_laplacian, solve_cg
from finitevolx._src.solvers.iterative import _masked_neumann_laplacian
from finitevolx._src.solvers.multigrid import (
    MultigridSolver,
    _apply_operator,
    _compute_n_levels,
    _prolongate,
    _restrict,
    _transfer_operators,
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
    """32x32 square grid (4 multigrid levels: 32 -> 16 -> 8 -> 4)."""
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


def _basin_mask2d(n_interior):
    """Mask2D of a circular basin on an (n+2) grid; its ring is dry."""
    n = n_interior + 2
    jj, ii = np.mgrid[:n, :n]
    c = (n - 1) / 2
    h = ((jj - c) ** 2 + (ii - c) ** 2) < (0.45 * n_interior) ** 2
    return Mask2D.from_mask(h.astype(float))


def _dense_wet_operator(level):
    """Dense matrix of the level operator restricted to its wet cells."""
    ny, nx = level.mask.shape
    wet = np.asarray(level.mask).ravel() > 0
    eye = jnp.eye(ny * nx)
    a = np.asarray(
        jax.vmap(lambda e: _apply_operator(e.reshape(ny, nx), level).ravel())(eye)
    ).T
    return a[np.ix_(wet, wet)]


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

    @pytest.mark.parametrize(
        ("n", "minimum"), [(34, 4), (64, 5), (130, 6), (258, 7), (16, 3)]
    )
    def test_n_levels_odd_sizes_coarsen(self, n, minimum):
        """(N+2) grids coarsen as deep as N (#305): (n + 1) // 2 per level."""
        assert _compute_n_levels(n, n) >= minimum

    def test_odd_shape_any_n_levels(self):
        """Any shape works with any n_levels (no divisibility constraint)."""
        solver = build_multigrid_solver(np.ones((15, 15)), 1.0, 1.0, n_levels=3)
        assert [lv.mask.shape for lv in solver.levels] == [(15, 15), (8, 8), (4, 4)]

    def test_bad_n_levels_raises(self):
        with pytest.raises(ValueError, match="n_levels"):
            build_multigrid_solver(np.ones((8, 8)), 1.0, 1.0, n_levels=0)

    def test_too_large_bottom_raises(self):
        with pytest.raises(ValueError, match="dense bottom"):
            build_multigrid_solver(np.ones((80, 80)), 1.0, 1.0, n_levels=1)

    def test_variable_coefficient(self, mg_grid_16):
        ny, nx, dx, dy = mg_grid_16
        mask = np.ones((ny, nx))
        coeff = np.random.default_rng(42).uniform(0.5, 2.0, (ny, nx))
        solver = build_multigrid_solver(mask, dx, dy, coeff=coeff)
        assert isinstance(solver, MultigridSolver)

    def test_level_shapes(self):
        solver = _all_ones_solver(34, 18, 1.0, 1.0)
        shapes = [level.mask.shape for level in solver.levels]
        assert shapes == [(34, 18), (17, 9), (9, 5)]

    def test_coarse_mask_keeps_narrow_channel(self):
        """Mask-coarsening rule: a coarse point is wet iff any wet fine point
        lies in its interpolation support, so a one-cell channel survives."""
        mask = np.zeros((17, 17))
        mask[8, 1:16] = 1.0  # one-cell-wide channel on an odd row
        mask_c, _, _ = _transfer_operators(mask, "dirichlet")
        assert mask_c[4, 1:8].all() and mask_c[4, 0] == 1.0
        # nothing outside the channel's support
        assert mask_c[:4].sum() == 0 and mask_c[5:].sum() == 0

    def test_bad_bc_raises(self):
        with pytest.raises(ValueError, match="bc must be"):
            build_multigrid_solver(np.ones((8, 8)), 1.0, 1.0, bc="robin")


# ---------------------------------------------------------------------------
# TestBoundaryCondition (#304)
# ---------------------------------------------------------------------------


class TestBoundaryCondition:
    """bc="dirichlet" (default) is masked_laplacian's psi = 0 wall on the dry
    X nodes; bc="neumann" is the zero-flux wall of a T-point problem."""

    @pytest.mark.parametrize("kind", ["basin", "rectangle", "all_wet"])
    def test_dirichlet_matches_masked_laplacian(self, kind):
        n = 14
        dx, dy = 1.0 / n, 1.3 / n
        if kind == "basin":
            mask = _basin_mask2d(n)
        elif kind == "rectangle":
            mask = Mask2D.from_mask(np.ones((n + 2, n + 2)))
        else:
            mask = np.ones((n + 2, n + 2))
        solver = build_multigrid_solver(mask, dx, dy, lambda_=4.0)
        assert solver.bc == "dirichlet"
        level = solver.levels[0]
        m = np.asarray(level.mask)
        psi = jnp.asarray(np.random.default_rng(0).standard_normal(m.shape))
        out = np.asarray(_apply_operator(psi, level))
        # the documented contract: level 0 (c = 1) is masked_laplacian (dry
        # neighbours zero, zero-padded at the array edge); the same 5-point
        # arithmetic in a different order, so round-off only
        ref = np.asarray(masked_laplacian(psi, jnp.asarray(m), dx, dy, 4.0))
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * np.abs(ref).max())

    def test_neumann_matches_masked_neumann_laplacian(self):
        """Level 0 with bc="neumann", c = 1, is the solvers' Neumann operator
        on a basin with an island (the copies must not drift)."""
        n = 14
        dx, dy = 1.0 / n, 1.3 / n
        h = np.asarray(_basin_mask2d(n).h, dtype=float)
        h[6:9, 5:8] = 0.0  # island
        solver = build_multigrid_solver(h, dx, dy, lambda_=4.0, bc="neumann")
        psi = jnp.asarray(np.random.default_rng(0).standard_normal(h.shape))
        out = np.asarray(_apply_operator(psi, solver.levels[0]))
        ref = np.asarray(_masked_neumann_laplacian(psi, jnp.asarray(h), dx, dy, 4.0))
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * np.abs(ref).max())

    def test_mask2d_uses_psi_mask(self):
        mask = _basin_mask2d(14)
        solver = build_multigrid_solver(mask, 1.0, 1.0)
        np.testing.assert_array_equal(
            np.asarray(solver.levels[0].mask), np.asarray(mask.xy_corner_strict)
        )

    def test_neumann_annihilates_constants(self):
        mask = np.asarray(_basin_mask2d(14).h, dtype=float)
        solver = build_multigrid_solver(mask, 0.1, 0.1, bc="neumann")
        out = _apply_operator(jnp.ones(mask.shape), solver.levels[0])
        np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-10)

    @pytest.mark.slow
    def test_dirichlet_lambda0_nonsingular(self):
        """The Dirichlet operator is negative definite at lambda = 0 (the
        Neumann one has a zero eigenvalue per wet component)."""
        mask = _basin_mask2d(10)
        solver = build_multigrid_solver(mask, 1.0, 1.0, lambda_=0.0)
        eig = np.linalg.eigvalsh(_dense_wet_operator(solver.levels[0]))
        assert eig.max() < -1e-2  # smallest |eig| ~ 2 pi^2 / L^2 >> round-off
        neu = build_multigrid_solver(
            np.asarray(mask.xy_corner_strict), 1.0, 1.0, bc="neumann"
        )
        eig_n = np.linalg.eigvalsh(_dense_wet_operator(neu.levels[0]))
        assert abs(eig_n.max()) < 1e-10

    @pytest.mark.slow
    @pytest.mark.parametrize("lambda_", [0.0, 4.0])
    def test_dirichlet_solve_matches_cg(self, lambda_):
        """MG on a masked basin solves masked_laplacian's problem: it agrees
        with CG on the same operator to the solve tolerance."""
        n = 30
        dx = dy = 1.0 / n
        mask = _basin_mask2d(n)
        m = jnp.asarray(mask.xy_corner_strict, dtype=float)
        rhs = jnp.asarray(np.random.default_rng(1).standard_normal(m.shape)) * m
        x_cg, info = solve_cg(
            lambda p: masked_laplacian(p, m, dx, dy, lambda_),
            rhs,
            rtol=1e-12,
            atol=1e-12,
            max_steps=2000,
        )
        assert info.converged
        solver = build_multigrid_solver(mask, dx, dy, lambda_=lambda_, n_cycles=40)
        u = solver(rhs)
        # 40 V-cycles drive the MG error far below the 1e-8 asserted here; CG
        # is converged to 1e-12
        rel = float(jnp.abs(u - x_cg).max() / jnp.abs(x_cg).max())
        assert rel < 1e-8

    @pytest.mark.slow
    def test_neumann_lambda0_zero_mean(self):
        """bc="neumann" at lambda = 0 solves the projected (solvable) problem
        and returns the zero-mean solution on each wet component."""
        n = 30
        h = np.asarray(_basin_mask2d(n).h, dtype=float)
        h[n // 2, :] = 0.0  # split the basin into two components
        solver = build_multigrid_solver(h, 1.0 / n, 1.0 / n, bc="neumann", n_cycles=30)
        assert int(solver.null_counts.shape[0]) == 3  # dry + two basins
        rhs = jnp.asarray(np.random.default_rng(2).standard_normal(h.shape)) * h
        u = solver(rhs)
        rhs_p = solver._project(rhs)
        res = rhs_p - _apply_operator(u, solver.levels[0])
        assert float(jnp.linalg.norm(res) / jnp.linalg.norm(rhs_p)) < 1e-6
        top, bottom = h.copy(), h.copy()
        top[: n // 2] = 0.0
        bottom[n // 2 :] = 0.0
        for part in (top, bottom):
            assert abs(float(jnp.sum(u * part))) < 1e-10 * float(jnp.abs(u).sum())


# ---------------------------------------------------------------------------
# TestConvergence (#305, #344)
# ---------------------------------------------------------------------------


class TestConvergence:
    """Full-depth coarsening, exact bottom solve, R = P^T / 4: the V-cycle
    contracts at a grid-independent rate and is a symmetric operator."""

    @staticmethod
    def _contraction(solver, n_cycles=8, rhs=None):
        """Per-cycle residual ratios ||r_k|| / ||r_{k-1}|| from u = 0."""
        level = solver.levels[0]
        if rhs is None:
            rng = np.random.default_rng(0)
            rhs = jnp.asarray(rng.standard_normal(level.mask.shape)) * level.mask
        cycle = eqx.filter_jit(lambda s, u, f: s.v_cycle(u, f))
        u = jnp.zeros_like(rhs)
        norms = [float(jnp.linalg.norm(rhs))]
        for _ in range(n_cycles):
            u = cycle(solver, u, rhs)
            norms.append(float(jnp.linalg.norm(rhs - _apply_operator(u, level))))
        norms = np.asarray(norms)
        return norms[1:] / norms[:-1]

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [32, 128, 256])
    def test_asymptotic_contraction(self, n):
        """34^2, 130^2, 258^2 basins: <= 0.3 per cycle on cycles 5-8 (measured
        0.06-0.07, grid independent; before #305: 0.2 / 0.89 / 0.91)."""
        solver = build_multigrid_solver(_basin_mask2d(n), 1.0 / n, 1.0 / n, 4.0)
        assert solver.n_levels >= {32: 4, 128: 6, 256: 7}[n]
        rates = self._contraction(solver)
        assert rates[4:8].max() <= 0.3

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("bc", "lambda_"), [("dirichlet", 4.0), ("dirichlet", 0.0), ("neumann", 0.0)]
    )
    def test_v_cycle_symmetric(self, bc, lambda_):
        """<M r1, r2> = <r1, M r2> to round-off for M = v_cycle(0, .) (#344)."""
        n = 30
        mask2d = _basin_mask2d(n)
        mask = mask2d if bc == "dirichlet" else np.asarray(mask2d.h, dtype=float)
        solver = build_multigrid_solver(mask, 1.0 / n, 1.0 / n, lambda_, bc=bc)
        m = solver.levels[0].mask
        rng = np.random.default_rng(4)
        r1, r2 = (jnp.asarray(rng.standard_normal(m.shape)) * m for _ in range(2))
        pc = make_multigrid_preconditioner(solver)
        a, b = float(jnp.sum(pc(r1) * r2)), float(jnp.sum(r1 * pc(r2)))
        assert abs(a - b) <= 1e-12 * abs(a)

    def test_unequal_smoothing_rejected_as_preconditioner(self):
        solver = build_multigrid_solver(np.ones((16, 16)), 1.0, 1.0, n_pre=2, n_post=1)
        with pytest.raises(ValueError, match="n_pre == n_post"):
            make_multigrid_preconditioner(solver)

    def test_single_level_is_exact(self):
        """n_levels=1 is the exact bottom solve: one cycle solves A u = f."""
        mask = _basin_mask2d(14)
        solver = build_multigrid_solver(mask, 0.1, 0.1, lambda_=1.0, n_levels=1)
        level = solver.levels[0]
        rhs = jnp.asarray(np.random.default_rng(5).standard_normal(level.mask.shape))
        rhs = rhs * level.mask
        u = solver.v_cycle(jnp.zeros_like(rhs), rhs)
        res = jnp.linalg.norm(rhs - _apply_operator(u, level)) / jnp.linalg.norm(rhs)
        assert float(res) < 1e-12

    @pytest.mark.slow
    def test_cg_iterations_circular_basin_128(self):
        """CG + one V-cycle: <= 15 iterations at 128^2 (measured 6; 146 before
        #304/#305)."""
        n = 128
        dx = 1.0 / n
        mask = _basin_mask2d(n)
        m = jnp.asarray(mask.xy_corner_strict, dtype=float)
        solver = build_multigrid_solver(mask, dx, dx, lambda_=4.0)
        rhs = jnp.asarray(np.random.default_rng(1).standard_normal(m.shape)) * m
        _, info = solve_cg(
            lambda p: masked_laplacian(p, m, dx, dx, 4.0),
            rhs,
            preconditioner=make_multigrid_preconditioner(solver),
            rtol=1e-8,
            atol=1e-8,
        )
        assert info.converged
        assert int(info.iterations) <= 15

    @pytest.mark.slow
    def test_onestep_gradient_34_basin(self):
        """solve_onestep's gradient is within 5% of the implicit gradient on a
        34^2 basin (measured 3.4%; 68% off before #305)."""
        n = 32
        solver = build_multigrid_solver(_basin_mask2d(n), 1.0 / n, 1.0 / n, 4.0)
        rhs = solver.levels[0].mask

        def grad(solve):
            return jax.grad(lambda s: jnp.mean(solve(s * rhs)))(1.0)

        g_implicit, g_onestep = grad(solver), grad(solver.solve_onestep)
        assert abs(float(g_onestep - g_implicit)) <= 0.05 * abs(float(g_implicit))


# ---------------------------------------------------------------------------
# TestNarrowFeatures: one-cell lakes and channels at odd indices
# ---------------------------------------------------------------------------


def _narrow_mask(kind):
    """psi-style masks whose one-cell features sit at odd fine indices."""
    if kind == "lake_10":
        w = np.zeros((10, 10))
        w[1:-2, 1:-2] = 1.0
        w[3:8, 3:8] = 0.0
        w[5, 5] = 1.0  # isolated one-cell lake
    elif kind == "lake_12x66":
        w = np.zeros((12, 66))
        w[1:-2, 1:-2] = 1.0
        w[3:8, 29:34] = 0.0
        w[5, 31] = 1.0
    else:  # two basins joined by a one-cell channel in an odd column
        w = np.zeros((34, 34))
        w[1:-2, 1:-2] = 1.0
        w[10:24, :] = 0.0
        w[10:24, 15 if kind == "odd_channel" else 16] = 1.0
    return w


class TestNarrowFeatures:
    """The coarsening keeps the prolongation at full column rank, so the
    Galerkin coarse operators and the bottom inverse stay well defined (a
    one-cell lake at an odd index used to give four identical columns)."""

    @pytest.mark.parametrize(
        ("kind", "n_levels"),
        [
            ("lake_10", None),
            ("lake_12x66", None),
            ("odd_channel", None),
            ("odd_channel", 2),
            ("odd_channel", 3),
        ],
    )
    @pytest.mark.parametrize(
        ("bc", "lambda_"),
        [("dirichlet", 0.0), ("dirichlet", 4.0), ("neumann", 0.0), ("neumann", 4.0)],
    )
    def test_builds_full_rank_bounded_bottom(self, kind, n_levels, bc, lambda_):
        w = _narrow_mask(kind)
        solver = build_multigrid_solver(
            w, 0.1, 0.1, lambda_=lambda_, bc=bc, n_levels=n_levels
        )
        for level in solver.levels[:-1]:
            mask_f = np.asarray(level.mask)
            mask_c, _, p = _transfer_operators(mask_f, bc)
            p_wet = p.toarray()[np.ix_(mask_f.ravel() > 0, mask_c.ravel() > 0)]
            assert np.linalg.matrix_rank(p_wet) == p_wet.shape[1]
        # |A_c^-1| |A_c| is a condition-number bound: O(1e2) here, ~1e13 (or a
        # singular-matrix error) with rank-deficient coarse operators
        a_max = float(jnp.abs(solver.levels[-1].stencil).max())
        b_max = float(jnp.abs(solver.bottom_inverse).max())
        assert np.isfinite(b_max) and b_max * a_max < 1e6

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["odd_channel", "even_channel", "island"])
    @pytest.mark.parametrize(
        ("bc", "lambda_"),
        [("dirichlet", 0.0), ("dirichlet", 4.0), ("neumann", 0.0), ("neumann", 4.0)],
    )
    def test_contraction_and_cg(self, kind, bc, lambda_):
        """Measured per-cycle contraction (cycles 5-8) on these 34^2 masks:
        Dirichlet 0.05-0.09; Neumann 0.09-0.23 (lambda = 4) and 0.15-0.48
        (lambda = 0, slowest on the odd channel). CG with the V-cycle as
        preconditioner takes 5-7 iterations on all of them."""
        if kind == "island":
            w = np.zeros((34, 34))
            w[1:-2, 1:-2] = 1.0
            w[12:20, 12:20] = 0.0
        else:
            w = _narrow_mask(kind)
        solver = build_multigrid_solver(w, 0.1, 0.1, lambda_=lambda_, bc=bc)
        level = solver.levels[0]
        m = level.mask
        rhs = jnp.asarray(np.random.default_rng(0).standard_normal(w.shape)) * m
        if bc == "neumann" and lambda_ == 0.0:
            rhs = solver._project(rhs)  # the solvable part
        rates = TestConvergence._contraction(solver, rhs=rhs)
        bound = 0.3 if (bc == "dirichlet" or lambda_ > 0) else 0.6
        assert rates[4:8].max() <= bound
        _, info = solve_cg(
            lambda x: _apply_operator(x, level),
            rhs,
            preconditioner=make_multigrid_preconditioner(solver),
            rtol=1e-8,
            atol=1e-8 * float(jnp.abs(rhs).max()),
        )
        assert info.converged and int(info.iterations) <= 15


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
    @staticmethod
    def _ops(mask, bc="dirichlet"):
        mask_c, scale, p = _transfer_operators(np.asarray(mask, dtype=float), bc)
        return jnp.asarray(mask_c), jnp.asarray(scale), p

    def test_restrict_constant_interior(self):
        """Full weighting (4, 2, 2, 1)/16 averages a constant exactly."""
        mask_c, scale, _ = self._ops(np.ones((16, 16)))
        result = _restrict(jnp.full((16, 16), 3.14), scale, mask_c)
        assert result.shape == (8, 8)
        np.testing.assert_allclose(np.array(result)[1:-1, 1:-1], 3.14, atol=1e-12)

    def test_prolong_constant(self):
        """Neumann interpolation is renormalised over wet coarse points, so a
        constant is reproduced up to the walls (Dirichlet tapers to the
        zero wall value instead)."""
        mask = _circular_mask(17, 17)
        mask_c, scale, _ = self._ops(mask, "neumann")
        v = jnp.full(mask_c.shape, 2.71)
        result = _prolongate(v, mask_c, scale)
        assert result.shape == (17, 17)
        np.testing.assert_allclose(np.array(result), 2.71 * mask, atol=1e-12)

    @pytest.mark.parametrize("bc", ["dirichlet", "neumann"])
    @pytest.mark.parametrize("shape", [(16, 16), (17, 15)])
    def test_matches_sparse_and_is_transpose(self, bc, shape):
        """_prolongate is the build-time P and _restrict is exactly P^T / 4,
        on even and odd shapes and an irregular mask (#305, #344)."""
        mask = _circular_mask(*shape, radius_frac=0.45)
        mask_c, scale, p = self._ops(mask, bc)
        rng = np.random.default_rng(0)
        e_c = rng.standard_normal(mask_c.shape)
        r_f = rng.standard_normal(shape)
        pe = np.asarray(_prolongate(jnp.asarray(e_c), mask_c, scale))
        np.testing.assert_allclose(
            pe.ravel(), p @ (e_c.ravel() * np.asarray(mask_c).ravel()), atol=1e-14
        )
        rr = np.asarray(_restrict(jnp.asarray(r_f), scale, mask_c))
        np.testing.assert_allclose(rr.ravel(), 0.25 * (p.T @ r_f.ravel()), atol=1e-14)
        # <R r, e>_c = <r, P e>_f / 4
        np.testing.assert_allclose(
            np.sum(rr * e_c), 0.25 * np.sum(r_f * pe), rtol=1e-12
        )

    def test_masked_restrict_no_contamination(self):
        mask_f = np.ones((16, 16))
        mask_f[:, :8] = 0.0
        mask_c, scale, _ = self._ops(mask_f)
        result = _restrict(jnp.ones((16, 16)) * 5.0, scale, mask_c)
        outside = np.array(mask_c) == 0
        assert outside.any()
        np.testing.assert_allclose(np.array(result)[outside], 0.0, atol=1e-15)

    def test_galerkin_coarse_operator(self):
        """Every coarse level applies A_c = R A P (9-point stencil)."""
        mask = _basin_mask2d(30)
        solver = build_multigrid_solver(mask, 1.0 / 30, 1.0 / 30, lambda_=4.0)
        rng = np.random.default_rng(3)
        for fine, coarse in zip(solver.levels[:-1], solver.levels[1:], strict=True):
            e_c = jnp.asarray(rng.standard_normal(coarse.mask.shape)) * coarse.mask
            pe = _prolongate(e_c, coarse.mask, fine.prolong_scale)
            rap = _restrict(_apply_operator(pe, fine), fine.prolong_scale, coarse.mask)
            ac = _apply_operator(e_c, coarse)
            np.testing.assert_allclose(
                np.asarray(ac), np.asarray(rap), atol=1e-10 * float(jnp.abs(ac).max())
            )


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
        """Constant-coeff Poisson (Dirichlet walls) on a full rectangle:
        residual decreases."""
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

    @pytest.mark.slow
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

    @pytest.mark.slow
    @pytest.mark.parametrize("masked", [False, True])
    def test_symmetric_positive_definite(self, masked):
        """CG needs a symmetric positive definite M^-1. The raw V-cycle V is
        symmetric (R = P^T / 4, equal pre/post smoothing, #344) and positive
        definite; the gaussx preconditioner is the two-step iteration
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
        # the raw V-cycle is symmetric positive definite on its own
        np.testing.assert_allclose(Vw, Vw.T, rtol=0, atol=1e-12 * np.abs(Vw).max())
        assert np.linalg.eigvalsh(Vw).min() > 0.0

        M = self._dense(make_gaussx_multigrid_preconditioner(solver), ny * nx)
        np.testing.assert_allclose(
            M, V + V.T - V.T @ K @ V, rtol=1e-9, atol=1e-12 * np.abs(M).max()
        )
        M = M[np.ix_(wet, wet)]
        np.testing.assert_allclose(M, M.T, rtol=0, atol=1e-12 * np.abs(M).max())
        assert np.linalg.eigvalsh(M).min() > 0.0

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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


class TestFloat32:
    """The hierarchy is float64, but a float32 right-hand side gets a float32
    solve (contract 7): the level arrays are cast where they are used."""

    @pytest.mark.slow
    @pytest.mark.parametrize(("bc", "lambda_"), [("dirichlet", 4.0), ("neumann", 0.0)])
    def test_solver_follows_rhs_dtype(self, bc, lambda_):
        w = np.zeros((18, 18))
        w[1:-2, 1:-2] = 1.0
        solver = build_multigrid_solver(w, 1.0 / 16, 1.0 / 16, lambda_, bc=bc)
        rhs = jnp.asarray(np.random.default_rng(0).standard_normal(w.shape) * w)
        u64 = solver(rhs)
        rhs32 = rhs.astype(jnp.float32)
        for solve in (solver, solver.solve_onestep, solver.solve_unrolled):
            u32 = solve(rhs32)
            assert u32.dtype == jnp.float32
            # float32 round-off on an O(1e-2) solution
            scale = float(jnp.abs(u64).max())
            assert float(jnp.abs(u32 - u64).max()) < 1e-4 * scale

    @pytest.mark.slow
    def test_preconditioner_in_float32_cg(self):
        w = np.zeros((18, 18))
        w[1:-2, 1:-2] = 1.0
        dx = 1.0 / 16
        solver = build_multigrid_solver(w, dx, dx, lambda_=4.0)
        m32 = jnp.asarray(w, dtype=jnp.float32)
        rhs = jnp.asarray(np.random.default_rng(1).standard_normal(w.shape) * w)
        rhs = rhs.astype(jnp.float32)
        pc = make_multigrid_preconditioner(solver)
        assert pc(rhs).dtype == jnp.float32
        x, info = solve_cg(
            lambda p: masked_laplacian(p, m32, dx, dx, 4.0), rhs, preconditioner=pc
        )
        assert x.dtype == jnp.float32 and info.converged


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
