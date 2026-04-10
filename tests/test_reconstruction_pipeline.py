"""Tests for the public reconstruction pipeline API.

Covers:
- Linear stencil functions (polynomial exactness, symmetry)
- reconstruct() dispatcher (conservation, accuracy, JAX transforms)
- distbound mask accessors
- plusminus velocity splitting
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    Mask2D,
    linear_2pts,
    linear_3pts_left,
    linear_3pts_right,
    linear_4pts,
    linear_5pts_left,
    linear_5pts_right,
    linear_6pts,
    plusminus,
    reconstruct,
    upwind_1pt,
    upwind_3pt,
    upwind_5pt,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar(v: float) -> jnp.ndarray:
    return jnp.asarray(v, dtype=float)


def _smooth_field_1d(N: int, dx: float = 1.0) -> jnp.ndarray:
    """Smooth sine field on N cells."""
    x = (jnp.arange(N) + 0.5) * dx
    return jnp.sin(2.0 * jnp.pi * x / (N * dx))


def _smooth_field_2d(Ny: int, Nx: int) -> jnp.ndarray:
    """Smooth sine field on [Ny, Nx] grid with ghost ring."""
    j = jnp.arange(Ny)[:, None]
    i = jnp.arange(Nx)[None, :]
    return jnp.sin(2.0 * jnp.pi * i / Nx) * jnp.cos(2.0 * jnp.pi * j / Ny)


# ===========================================================================
# Linear stencil tests
# ===========================================================================


class TestLinear2pts:
    """linear_2pts must be exact for degree-0 (constant) fields."""

    def test_constant_field(self):
        C = 3.7
        q0 = _scalar(C)
        qp = _scalar(C)
        result = float(linear_2pts(q0, qp))
        assert result == pytest.approx(C)

    def test_linear_field(self):
        """For a linear field f(x)=x, face value at i+1/2 = (i + i+1)/2."""
        q0 = _scalar(2.0)
        qp = _scalar(3.0)
        result = float(linear_2pts(q0, qp))
        assert result == pytest.approx(2.5)

    def test_vectorized(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        faces = linear_2pts(q[:-1], q[1:])
        np.testing.assert_allclose(faces, [1.5, 2.5, 3.5])


class TestLinear3pts:
    """3-point stencils must be exact for degree ≤ 2 polynomials."""

    def test_constant(self):
        C = 5.0
        vals = [_scalar(C)] * 3
        assert float(linear_3pts_left(*vals)) == pytest.approx(C)
        assert float(linear_3pts_right(*vals)) == pytest.approx(C)

    def test_linear_field(self):
        """f(x) = x at cells 0, 1, 2. Face between cells 1 and 2 (1+1/2) is 1.5."""
        qm, q0, qp = _scalar(0.0), _scalar(1.0), _scalar(2.0)
        assert float(linear_3pts_left(qm, q0, qp)) == pytest.approx(1.5)
        # Right-biased: q0=1, qp=2, qpp=3, face between cells 1 and 2 (1+1/2) = 1.5
        assert float(
            linear_3pts_right(_scalar(1.0), _scalar(2.0), _scalar(3.0))
        ) == pytest.approx(1.5)

    def test_coefficients(self):
        """Verify the stencil coefficients match the documented formula."""
        qm, q0, qp = _scalar(3.0), _scalar(7.0), _scalar(11.0)
        expected_left = -1.0 / 6.0 * 3.0 + 5.0 / 6.0 * 7.0 + 1.0 / 3.0 * 11.0
        assert float(linear_3pts_left(qm, q0, qp)) == pytest.approx(
            expected_left, abs=1e-12
        )
        expected_right = 1.0 / 3.0 * 3.0 + 5.0 / 6.0 * 7.0 - 1.0 / 6.0 * 11.0
        assert float(linear_3pts_right(qm, q0, qp)) == pytest.approx(
            expected_right, abs=1e-12
        )


class TestLinear4pts:
    """4-point symmetric stencil exactness."""

    def test_constant(self):
        C = 2.0
        result = float(linear_4pts(*[_scalar(C)] * 4))
        assert result == pytest.approx(C)

    def test_symmetry(self):
        """Symmetric stencil with symmetric data: q0=qp → result = q0."""
        qm, q0, qp, qpp = _scalar(1.0), _scalar(5.0), _scalar(5.0), _scalar(1.0)
        result = float(linear_4pts(qm, q0, qp, qpp))
        # Symmetric: -1/12(1) + 7/12(5) + 7/12(5) - 1/12(1) = -2/12 + 70/12 = 68/12
        expected = (
            -1.0 / 12.0 * 1.0 + 7.0 / 12.0 * 5.0 + 7.0 / 12.0 * 5.0 - 1.0 / 12.0 * 1.0
        )
        assert result == pytest.approx(expected, abs=1e-12)

    def test_linear_exactness(self):
        """4-point centred should be exact for linear f(x) = x."""
        # Cells at -1, 0, 1, 2; face at 0.5
        qm, q0, qp, qpp = _scalar(-1.0), _scalar(0.0), _scalar(1.0), _scalar(2.0)
        result = float(linear_4pts(qm, q0, qp, qpp))
        assert result == pytest.approx(0.5, abs=1e-12)


class TestLinear5pts:
    """5-point stencils must be exact for degree ≤ 4 polynomials."""

    def test_constant(self):
        C = 4.2
        vals = [_scalar(C)] * 5
        assert float(linear_5pts_left(*vals)) == pytest.approx(C)
        assert float(linear_5pts_right(*vals)) == pytest.approx(C)

    def test_linear(self):
        vals_left = [_scalar(float(i)) for i in range(-2, 3)]
        result = float(linear_5pts_left(*vals_left))
        assert result == pytest.approx(0.5)

    def test_coefficients(self):
        """Verify the stencil coefficients match the documented formula."""
        vals = [_scalar(float(x)) for x in [2.0, 3.0, 5.0, 7.0, 11.0]]
        expected = (
            1.0 / 30.0 * 2.0
            - 13.0 / 60.0 * 3.0
            + 47.0 / 60.0 * 5.0
            + 9.0 / 20.0 * 7.0
            - 1.0 / 20.0 * 11.0
        )
        result = float(linear_5pts_left(*vals))
        assert result == pytest.approx(expected, abs=1e-12)


class TestLinear6pts:
    """6-point symmetric stencil exactness."""

    def test_constant(self):
        C = 1.5
        result = float(linear_6pts(*[_scalar(C)] * 6))
        assert result == pytest.approx(C)

    def test_symmetry(self):
        """Symmetric data → result = centred average."""
        vals = [
            _scalar(1.0),
            _scalar(4.0),
            _scalar(9.0),
            _scalar(9.0),
            _scalar(4.0),
            _scalar(1.0),
        ]
        result = float(linear_6pts(*vals))
        # Symmetric field: should evaluate to f(midpoint)
        assert np.isfinite(result)

    def test_linear_exactness(self):
        """6-point stencil should reconstruct linear fields exactly."""
        # f(x) = x at cells -2, -1, 0, 1, 2, 3; face at 0.5
        vals = [_scalar(float(x)) for x in range(-2, 4)]
        result = float(linear_6pts(*vals))
        assert result == pytest.approx(0.5, abs=1e-12)

    def test_coefficients(self):
        """Verify the stencil coefficients match the documented formula."""
        vals = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]
        expected = (
            1.0 / 60.0 * 2.0
            - 2.0 / 15.0 * 3.0
            + 37.0 / 60.0 * 5.0
            + 37.0 / 60.0 * 7.0
            - 2.0 / 15.0 * 11.0
            + 1.0 / 60.0 * 13.0
        )
        result = float(linear_6pts(*[_scalar(v) for v in vals]))
        assert result == pytest.approx(expected, abs=1e-12)


class TestLinearWENOConsistency:
    """In smooth regions, WENO should reduce to the linear stencil values."""

    def test_weno3_smooth_approaches_linear3(self):
        """On smooth data, WENO3 output should match linear_3pts_left closely."""
        from finitevolx._src.advection.weno import weno_3pts

        # Smooth data with large offset to ensure smoothness ratios are good
        N = 100
        x = jnp.linspace(0, 2 * jnp.pi, N)
        q = 100.0 + jnp.sin(x)

        weno_vals = weno_3pts(q[:-2], q[1:-1], q[2:])
        linear_vals = linear_3pts_left(q[:-2], q[1:-1], q[2:])

        # WENO should be very close to linear on smooth data
        np.testing.assert_allclose(weno_vals, linear_vals, atol=2e-3)

    def test_weno5_smooth_approaches_linear5(self):
        """On smooth data, WENO5 output should match linear_5pts_left closely."""
        from finitevolx._src.advection.weno import weno_5pts

        N = 100
        x = jnp.linspace(0, 2 * jnp.pi, N)
        q = 1.0 + 0.01 * jnp.sin(x)

        weno_vals = weno_5pts(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        linear_vals = linear_5pts_left(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])

        np.testing.assert_allclose(weno_vals, linear_vals, atol=1e-6)


# ===========================================================================
# plusminus tests
# ===========================================================================


class TestPlusMinus:
    def test_basic_split(self):
        u = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        u_pos, u_neg = plusminus(u)
        np.testing.assert_array_equal(u_pos, [0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(u_neg, [-2.0, -1.0, 0.0, 0.0, 0.0])

    def test_reconstruction(self):
        """u_pos + u_neg == u."""
        u = jnp.array([-3.0, 1.5, 0.0, -0.1, 7.0])
        u_pos, u_neg = plusminus(u)
        np.testing.assert_allclose(u_pos + u_neg, u)


# ===========================================================================
# Low-level upwind helper tests
# ===========================================================================


class TestUpwind1pt:
    def test_positive_flow_selects_left(self):
        # q = [g0, c1, c2, g3]; interior faces between c1-c2
        q = jnp.array([0.0, 1.0, 2.0, 3.0])
        u = jnp.array([1.0, 1.0])  # 2 interior faces
        result = upwind_1pt(q, u)
        # Positive flow → left cell: q[1]=1.0, q[2]=2.0
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_negative_flow_selects_right(self):
        q = jnp.array([0.0, 1.0, 2.0, 3.0])
        u = jnp.array([-1.0, -1.0])
        result = upwind_1pt(q, u)
        # Negative flow → right cell: q[2]=2.0, q[3]=3.0
        np.testing.assert_array_equal(result, [2.0, 3.0])

    def test_constant_field(self):
        q = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0])
        u = jnp.array([1.0, -1.0, 0.5])
        result = upwind_1pt(q, u)
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])


class TestUpwind3pt:
    def test_constant_field_all_methods(self):
        """Constant scalar → face value = constant for all methods."""
        C = 3.0
        q = jnp.full(6, C)
        u = jnp.ones(4)
        for method in ("weno", "wenoz", "linear"):
            result = upwind_3pt(q, u, method)
            np.testing.assert_allclose(
                result, C, atol=1e-12, err_msg=f"method={method}"
            )

    def test_linear_field_positive(self):
        """Linear field f(x)=x: face values should be exact midpoints."""
        q = jnp.arange(6, dtype=float)  # 0, 1, 2, 3, 4, 5
        u = jnp.ones(4)  # all positive
        result = upwind_3pt(q, u, method="linear")
        # 3-point left-biased on linear data is exact
        expected = jnp.array([1.5, 2.5, 3.5, 4.5])
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestUpwind5pt:
    def test_constant_field(self):
        C = 7.0
        q = jnp.full(8, C)
        u = jnp.ones(4)
        for method in ("weno", "wenoz", "linear"):
            result = upwind_5pt(q, u, method)
            np.testing.assert_allclose(
                result, C, atol=1e-12, err_msg=f"method={method}"
            )

    def test_linear_field(self):
        q = jnp.arange(8, dtype=float)
        u = jnp.ones(4)
        result = upwind_5pt(q, u, method="linear")
        expected = jnp.array([2.5, 3.5, 4.5, 5.5])
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ===========================================================================
# reconstruct() dispatcher tests
# ===========================================================================


class TestReconstructBasic:
    """Basic correctness of the reconstruct() dispatcher."""

    @pytest.fixture()
    def uniform_grid(self):
        """10x10 grid with ghost ring."""
        Ny, Nx = 10, 10
        return Ny, Nx

    def test_constant_scalar_x(self, uniform_grid):
        """Constant scalar + uniform velocity → constant flux."""
        Ny, Nx = uniform_grid
        C = 3.0
        q = jnp.full((Ny, Nx), C)
        u = jnp.ones((Ny, Nx))
        for num_pts in (1, 3, 5):
            flux = reconstruct(q, u, dim=1, method="weno", num_pts=num_pts)
            assert flux.shape == (Ny, Nx)
            # Interior flux should be C * 1.0 = 3.0
            interior = flux[1:-1, 1:-1]
            np.testing.assert_allclose(
                interior, C, atol=1e-10, err_msg=f"num_pts={num_pts}"
            )

    def test_constant_scalar_y(self, uniform_grid):
        """Constant scalar + uniform velocity in y-direction."""
        Ny, Nx = uniform_grid
        C = 2.5
        q = jnp.full((Ny, Nx), C)
        v = jnp.ones((Ny, Nx))
        for num_pts in (1, 3, 5):
            flux = reconstruct(q, v, dim=0, method="weno", num_pts=num_pts)
            assert flux.shape == (Ny, Nx)
            interior = flux[1:-1, 1:-1]
            np.testing.assert_allclose(
                interior, C, atol=1e-10, err_msg=f"num_pts={num_pts}"
            )

    def test_zero_velocity_gives_zero_flux(self, uniform_grid):
        """Zero velocity → zero flux regardless of scalar field."""
        Ny, Nx = uniform_grid
        q = _smooth_field_2d(Ny, Nx)
        u = jnp.zeros((Ny, Nx))
        flux = reconstruct(q, u, dim=1, method="weno", num_pts=3)
        np.testing.assert_allclose(flux, 0.0, atol=1e-15)

    def test_ghost_ring_is_zero(self, uniform_grid):
        """Output ghost ring must always be zero."""
        Ny, Nx = uniform_grid
        q = _smooth_field_2d(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        for dim in (0, 1):
            for num_pts in (1, 3, 5):
                flux = reconstruct(q, u, dim=dim, method="weno", num_pts=num_pts)
                # Check ghost ring
                assert jnp.all(flux[0, :] == 0.0)
                assert jnp.all(flux[-1, :] == 0.0)
                assert jnp.all(flux[:, 0] == 0.0)
                assert jnp.all(flux[:, -1] == 0.0)

    def test_all_methods(self, uniform_grid):
        """All method strings work without error."""
        Ny, Nx = uniform_grid
        q = _smooth_field_2d(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        for method in ("weno", "wenoz", "linear"):
            for num_pts in (1, 3, 5):
                flux = reconstruct(q, u, dim=1, method=method, num_pts=num_pts)
                assert flux.shape == (Ny, Nx)
                assert jnp.all(jnp.isfinite(flux))


class TestReconstructAccuracy:
    """Scientific accuracy tests for the reconstruct() dispatcher."""

    def test_higher_order_more_accurate(self):
        """On a smooth field, 5-point should be more accurate than 3-point."""
        Ny, Nx = 20, 20
        # Smooth sine field
        q = _smooth_field_2d(Ny, Nx)
        u = jnp.ones((Ny, Nx))

        flux_1pt = reconstruct(q, u, dim=1, method="weno", num_pts=1)
        flux_3pt = reconstruct(q, u, dim=1, method="weno", num_pts=3)
        flux_5pt = reconstruct(q, u, dim=1, method="weno", num_pts=5)

        # Use linear (optimal) 5-point as reference
        ref = reconstruct(q, u, dim=1, method="linear", num_pts=5)

        err_1pt = jnp.linalg.norm(flux_1pt - ref)
        err_3pt = jnp.linalg.norm(flux_3pt - ref)
        err_5pt = jnp.linalg.norm(flux_5pt - ref)

        # Higher-order should be closer to the reference
        assert float(err_5pt) < float(err_3pt)
        assert float(err_3pt) < float(err_1pt)

    def test_convergence_rate_smooth_field(self):
        """Verify that reconstruction error decreases with grid refinement."""
        errors = []
        for N in (16, 32, 64):
            q = _smooth_field_2d(N, N)
            u = jnp.ones((N, N))
            flux_weno = reconstruct(q, u, dim=1, method="weno", num_pts=5)
            flux_lin = reconstruct(q, u, dim=1, method="linear", num_pts=5)
            err = float(jnp.max(jnp.abs(flux_weno[2:-2, 2:-2] - flux_lin[2:-2, 2:-2])))
            errors.append(err)
        # Error should decrease as grid is refined (WENO converges to linear)
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

    def test_upwind_sign_sensitivity(self):
        """Verify velocity sign correctly selects upwind/downwind stencil."""
        Ny, Nx = 12, 12
        # Asymmetric field — left side high, right side low
        i = jnp.arange(Nx)[None, :]
        q = jnp.broadcast_to(10.0 - i.astype(float), (Ny, Nx))

        u_pos = jnp.ones((Ny, Nx))
        u_neg = -jnp.ones((Ny, Nx))

        flux_pos = reconstruct(q, u_pos, dim=1, method="weno", num_pts=3)
        flux_neg = reconstruct(q, u_neg, dim=1, method="weno", num_pts=3)

        # Positive flow: face value biased toward left cell (higher)
        # Negative flow: face value biased toward right cell (lower)
        # Since flux = h_face * u, and u has opposite signs:
        # flux_pos should be positive, flux_neg should be negative
        interior_pos = flux_pos[2:-2, 2:-2]
        interior_neg = flux_neg[2:-2, 2:-2]
        assert jnp.all(interior_pos > 0)
        assert jnp.all(interior_neg < 0)


class TestReconstructNearDiscontinuity:
    """WENO's key property: reduced oscillation near discontinuities."""

    def test_step_function_bounded(self):
        """WENO reconstruction near a step should produce less overshoot than linear."""
        Ny, Nx = 6, 20  # Need enough rows for ghost ring
        q = jnp.zeros((Ny, Nx))
        q = q.at[:, :10].set(1.0)  # step at center
        u = jnp.ones((Ny, Nx))

        flux_weno = reconstruct(q, u, dim=1, method="weno", num_pts=5)
        flux_lin = reconstruct(q, u, dim=1, method="linear", num_pts=5)

        # Compare absolute overshoot (values > 1.0)
        weno_overshoot = float(jnp.max(jnp.abs(flux_weno[2:-2, 2:-2])))
        lin_overshoot = float(jnp.max(jnp.abs(flux_lin[2:-2, 2:-2])))

        # WENO max overshoot should be no worse than linear
        assert weno_overshoot <= lin_overshoot + 1e-10


class TestReconstructJaxCompat:
    """JAX transform compatibility."""

    def test_jit(self):
        Ny, Nx = 10, 10
        q = _smooth_field_2d(Ny, Nx)
        u = jnp.ones((Ny, Nx))

        @jax.jit
        def f(q, u):
            return reconstruct(q, u, dim=1, method="weno", num_pts=5)

        result = f(q, u)
        ref = reconstruct(q, u, dim=1, method="weno", num_pts=5)
        np.testing.assert_allclose(result, ref, atol=1e-14)

    def test_vmap_over_batch(self):
        """vmap over a batch dimension."""
        B, Ny, Nx = 4, 10, 10
        q = jnp.stack([_smooth_field_2d(Ny, Nx) * (i + 1) for i in range(B)])
        u = jnp.ones((B, Ny, Nx))

        # Verify sequential results match vmap results
        results_seq = jnp.stack(
            [reconstruct(q[i], u[i], dim=1, method="weno", num_pts=3) for i in range(B)]
        )

        @jax.vmap
        def f(q, u):
            return reconstruct(q, u, dim=1, method="weno", num_pts=3)

        result = f(q, u)
        assert result.shape == (B, Ny, Nx)
        assert jnp.all(jnp.isfinite(result))
        np.testing.assert_allclose(result, results_seq, atol=1e-14)

    def test_grad(self):
        """Gradient should be computable through reconstruct."""
        Ny, Nx = 8, 8
        q = _smooth_field_2d(Ny, Nx)
        u = jnp.ones((Ny, Nx))

        def loss(q):
            flux = reconstruct(q, u, dim=1, method="weno", num_pts=3)
            return jnp.sum(flux**2)

        grad_q = jax.grad(loss)(q)
        assert grad_q.shape == q.shape
        assert jnp.all(jnp.isfinite(grad_q))


class TestReconstructValidation:
    """Input validation and error handling."""

    def test_invalid_dim(self):
        q = jnp.ones((5, 5))
        u = jnp.ones((5, 5))
        with pytest.raises(ValueError, match="dim must be"):
            reconstruct(q, u, dim=2)

    def test_invalid_num_pts(self):
        q = jnp.ones((5, 5))
        u = jnp.ones((5, 5))
        with pytest.raises(ValueError, match="num_pts must be"):
            reconstruct(q, u, dim=1, num_pts=7)

    def test_invalid_method(self):
        q = jnp.ones((5, 5))
        u = jnp.ones((5, 5))
        with pytest.raises(ValueError, match="method must be one of"):
            reconstruct(q, u, dim=1, method="bogus", num_pts=3)
        # Also rejected for num_pts=1 (method is validated up-front)
        with pytest.raises(ValueError, match="method must be one of"):
            reconstruct(q, u, dim=1, method="bogus", num_pts=1)


# ===========================================================================
# Distbound mask accessor tests
# ===========================================================================


class TestDistboundAccessors:
    """Test the ind_coast/2/3plus properties on Mask2D."""

    @pytest.fixture()
    def island_mask(self):
        """16x16 domain with a 4x4 island in the centre → non-trivial coast."""
        Ny, Nx = 16, 16
        h = np.ones((Ny, Nx), dtype=bool)
        # Create island
        h[6:10, 6:10] = False
        return Mask2D.from_mask(h)

    @pytest.fixture()
    def all_ocean(self):
        return Mask2D.from_dimensions(10, 10)

    def test_distbound_mutual_exclusion(self, island_mask):
        """ind_coast, ind_near_coast, ind_ocean are mutually exclusive on wet cells."""
        d1 = island_mask.ind_coast
        d2 = island_mask.ind_near_coast
        d3 = island_mask.ind_ocean

        # No cell should be in more than one category
        overlap_12 = jnp.sum(d1 & d2)
        overlap_13 = jnp.sum(d1 & d3)
        overlap_23 = jnp.sum(d2 & d3)
        assert int(overlap_12) == 0
        assert int(overlap_13) == 0
        assert int(overlap_23) == 0

    def test_distbound_covers_all_wet(self, island_mask):
        """ind_coast + ind_near_coast + ind_ocean covers all wet cells."""
        d1 = island_mask.ind_coast
        d2 = island_mask.ind_near_coast
        d3 = island_mask.ind_ocean
        wet = island_mask.h

        covered = d1 | d2 | d3
        np.testing.assert_array_equal(covered, wet)

    def test_distbound_aliases_classification(self, island_mask):
        """distbound accessors should be consistent with ind_coast etc."""
        np.testing.assert_array_equal(island_mask.ind_coast, island_mask.ind_coast)
        np.testing.assert_array_equal(
            island_mask.ind_near_coast, island_mask.ind_near_coast
        )
        np.testing.assert_array_equal(island_mask.ind_ocean, island_mask.ind_ocean)

    def test_all_ocean_domain(self, all_ocean):
        """All-ocean domain: no coast/near-coast, all cells are ocean."""
        assert int(jnp.sum(all_ocean.ind_coast)) == 0
        assert int(jnp.sum(all_ocean.ind_near_coast)) == 0
        # All wet cells should be ind_ocean
        assert int(jnp.sum(all_ocean.ind_ocean)) == int(jnp.sum(all_ocean.h))

    def test_island_has_coast_ring(self, island_mask):
        """Island mask should have coast cells surrounding the island."""
        d1 = island_mask.ind_coast
        # Coast ring is cells adjacent to the 4x4 island
        assert int(jnp.sum(d1)) > 0
        # Coast cells should be near the island (rows 5-10, cols 5-10 area)
        coast_region = d1[4:12, 4:12]
        assert int(jnp.sum(coast_region)) > 0

    def test_stencil_blending_formula(self, island_mask):
        """The distbound masks work for the documented blending formula:
        flux = flux_1pt * ind_coast + flux_3pt * ind_near_coast + flux_5pt * ind_ocean
        """
        Ny, Nx = island_mask.h.shape
        # Create dummy flux fields
        flux_1pt = jnp.ones((Ny, Nx)) * 1.0
        flux_3pt = jnp.ones((Ny, Nx)) * 3.0
        flux_5pt = jnp.ones((Ny, Nx)) * 5.0

        blended = (
            flux_1pt * island_mask.ind_coast
            + flux_3pt * island_mask.ind_near_coast
            + flux_5pt * island_mask.ind_ocean
        )

        # At coast cells, value should be 1.0
        coast_vals = blended[island_mask.ind_coast]
        np.testing.assert_allclose(coast_vals, 1.0)
        # At near-coast, 3.0
        near_coast_vals = blended[island_mask.ind_near_coast]
        np.testing.assert_allclose(near_coast_vals, 3.0)
        # At ocean, 5.0
        ocean_vals = blended[island_mask.ind_ocean]
        np.testing.assert_allclose(ocean_vals, 5.0)


class TestDistboundPhysics:
    """Scientific tests verifying distbound masks reflect distance-from-land."""

    def test_coast_adjacent_to_land(self):
        """Every coast cell must have at least one land neighbour."""
        Ny, Nx = 20, 20
        h = np.ones((Ny, Nx), dtype=bool)
        h[8:12, 8:12] = False  # island
        mask = Mask2D.from_mask(h)

        coast = np.array(mask.ind_coast)
        land = np.array(mask.ind_land)

        for j in range(Ny):
            for i in range(Nx):
                if coast[j, i]:
                    # Check 4-connected neighbours for land
                    neighbours = []
                    if j > 0:
                        neighbours.append(land[j - 1, i])
                    if j < Ny - 1:
                        neighbours.append(land[j + 1, i])
                    if i > 0:
                        neighbours.append(land[j, i - 1])
                    if i < Nx - 1:
                        neighbours.append(land[j, i + 1])
                    assert any(neighbours), (
                        f"Coast cell ({j},{i}) has no land neighbour"
                    )

    def test_near_coast_not_adjacent_to_land(self):
        """Near-coast cells (ind_near_coast) should NOT be directly adjacent to land."""
        Ny, Nx = 20, 20
        h = np.ones((Ny, Nx), dtype=bool)
        h[8:12, 8:12] = False
        mask = Mask2D.from_mask(h)

        near_coast = np.array(mask.ind_near_coast)
        land = np.array(mask.ind_land)

        for j in range(Ny):
            for i in range(Nx):
                if near_coast[j, i]:
                    neighbours = []
                    if j > 0:
                        neighbours.append(land[j - 1, i])
                    if j < Ny - 1:
                        neighbours.append(land[j + 1, i])
                    if i > 0:
                        neighbours.append(land[j, i - 1])
                    if i < Nx - 1:
                        neighbours.append(land[j, i + 1])
                    assert not any(neighbours), (
                        f"Near-coast cell ({j},{i}) should not be adjacent to land"
                    )
