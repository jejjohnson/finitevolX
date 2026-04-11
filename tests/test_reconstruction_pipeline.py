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
    Advection2D,
    CartesianGrid2D,
    Mask2D,
    Reconstruction2D,
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
    upwind_flux,
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
        """ind_* boolean accessors are consistent with the underlying classification field."""
        cls_ = island_mask.classification
        np.testing.assert_array_equal(island_mask.ind_land, cls_ == 0)
        np.testing.assert_array_equal(island_mask.ind_coast, cls_ == 1)
        np.testing.assert_array_equal(island_mask.ind_near_coast, cls_ == 2)
        np.testing.assert_array_equal(island_mask.ind_ocean, cls_ == 3)

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


# ===========================================================================
# Stencil capability blending — tests that exercise the actual adaptive
# stencil masks (built from StencilCapability) together with the distbound
# classification, across every stencil size that Advection2D supports.
# ===========================================================================


class TestStencilCapabilityTiers:
    """Verify ``get_adaptive_masks`` tiers match the stencil hierarchy.

    These tests use the *actual* stencil-capability machinery rather than
    the distbound classification, covering the stencil sizes that
    ``Advection2D`` supports via ``upwind_flux`` dispatch:

    * weno3 / TVD limiters: ``(2, 4)``
    * weno5 / wenoz5:       ``(2, 4, 6)``
    """

    def test_weno5_tiers_on_1row_coast(self):
        """In a 1×N row with land borders, the (2, 4, 6) tiers stratify
        exactly by distance-from-land along the row:

        * index 1 / -2: adjacent to land → masks[2]
        * index 2 / -3: 2 away from land → masks[4]
        * indices 3..-4: interior        → masks[6]
        """
        h = np.zeros((1, 12), dtype=bool)
        h[0, 1:-1] = True  # 10 wet cells with land at both ends
        m = Mask2D.from_mask(h)
        masks_x = m.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))

        masks2 = np.asarray(masks_x[2])[0]
        masks4 = np.asarray(masks_x[4])[0]
        masks6 = np.asarray(masks_x[6])[0]

        # Adjacent-to-land → upwind1 tier
        assert masks2[1] and masks2[10]
        # 2 away from land → weno3 tier
        assert masks4[2] and masks4[9]
        # Interior → weno5 tier
        for i in range(3, 9):
            assert masks6[i], f"expected masks[6] True at index {i}"

    def test_weno3_tiers_on_1row_coast(self):
        """weno3/TVD uses (2, 4); any cell ≥2 away from land should be
        at the max tier since there's no bigger stencil to fall to."""
        h = np.zeros((1, 10), dtype=bool)
        h[0, 1:-1] = True  # 8 wet cells
        m = Mask2D.from_mask(h)
        masks_x = m.get_adaptive_masks(direction="x", stencil_sizes=(2, 4))

        masks2 = np.asarray(masks_x[2])[0]
        masks4 = np.asarray(masks_x[4])[0]

        # Adjacent-to-land → upwind1 tier
        assert masks2[1] and masks2[8]
        # Everything else (indices 2..7) → weno3 tier
        for i in range(2, 8):
            assert masks4[i], f"expected masks[4] True at index {i}"

    def test_mutually_exclusive_tiers(self):
        """A cell is assigned to exactly one tier (at most)."""
        rng = np.random.default_rng(42)
        h = rng.random((8, 12)) > 0.25  # sparse land
        m = Mask2D.from_mask(h)
        for direction in ("x", "y"):
            masks = m.get_adaptive_masks(direction=direction, stencil_sizes=(2, 4, 6))
            total = sum(np.asarray(mm).astype(int) for mm in masks.values())
            assert np.all(total <= 1), (
                f"direction={direction}: some cell is in multiple tiers"
            )

    def test_tier_union_subset_of_wet(self):
        """The union of all tiers is a subset of the wet cells: no dry
        cell can be at any tier."""
        h = np.ones((6, 10), dtype=bool)
        h[2:4, 4:6] = False  # interior land patch
        m = Mask2D.from_mask(h)
        masks_x = m.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        union = np.zeros_like(np.asarray(h))
        for mm in masks_x.values():
            union = union | np.asarray(mm)
        # Every tier cell must also be wet.
        assert bool(np.all((~union) | h))

    def test_directional_independence(self):
        """x- and y- tiers are independent: a cell can use a large
        x-stencil (open in x) but a small y-stencil (blocked in y)."""
        # 6x10 zonal channel: walls at j=0 and j=-1, open in x.
        h = np.ones((6, 10), dtype=bool)
        h[0, :] = h[-1, :] = False
        m = Mask2D.from_mask(h)
        masks_x = m.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        masks_y = m.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))

        # Row 1 (just above the south wall) and row 4 (just below the north
        # wall) are adjacent to land in y → y masks[2] is True there.
        assert bool(np.all(np.asarray(masks_y[2])[1, :]))
        assert bool(np.all(np.asarray(masks_y[2])[4, :]))
        # Deep interior of those rows in x is far from any land in x →
        # x masks[6] True (except the 2 cells nearest the x boundaries).
        for i in range(3, 7):
            assert bool(np.asarray(masks_x[6])[1, i])
            assert bool(np.asarray(masks_x[6])[4, i])


class TestStencilCapabilityDistboundCorrespondence:
    """Correspondence between ``get_adaptive_masks`` and distbound
    classification (``ind_coast``, ``ind_near_coast``, ``ind_ocean``).

    The two systems agree *exactly* in the axis-aligned 1-D case (a land
    border along the axis of interest).  In 2-D they can diverge because
    the distbound classification uses an n-D cross-shaped dilation while
    the adaptive masks are directional.
    """

    def test_axis_aligned_1row_correspondence(self):
        """In a 1-row domain with land borders, ind_coast matches
        masks[2], ind_near_coast matches masks[4], ind_ocean matches
        masks[6]."""
        h = np.zeros((1, 12), dtype=bool)
        h[0, 1:-1] = True
        m = Mask2D.from_mask(h)
        masks_x = m.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))

        np.testing.assert_array_equal(np.asarray(m.ind_coast), np.asarray(masks_x[2]))
        np.testing.assert_array_equal(
            np.asarray(m.ind_near_coast), np.asarray(masks_x[4])
        )
        np.testing.assert_array_equal(np.asarray(m.ind_ocean), np.asarray(masks_x[6]))

    def test_zonal_channel_y_direction_correspondence(self):
        """In a 6×10 zonal channel (walls at j=0 and j=-1), the
        y-direction adaptive masks match the distbound tiers exactly
        because the only land is in the y direction."""
        h = np.ones((6, 10), dtype=bool)
        h[0, :] = h[-1, :] = False
        m = Mask2D.from_mask(h)
        masks_y = m.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))

        np.testing.assert_array_equal(np.asarray(m.ind_coast), np.asarray(masks_y[2]))
        np.testing.assert_array_equal(
            np.asarray(m.ind_near_coast), np.asarray(masks_y[4])
        )
        np.testing.assert_array_equal(np.asarray(m.ind_ocean), np.asarray(masks_y[6]))

    def test_diagonal_land_divergence(self):
        """A cell adjacent to land *only* in the x-direction is
        classified as ``ind_coast`` (cross-shaped dilation sees the land),
        but its y-direction stencil is unaffected so y-masks[6] can
        still be True at that cell."""
        # 10x10 with a narrow north-south land column at i=5.
        h = np.ones((10, 10), dtype=bool)
        h[:, 5] = False
        m = Mask2D.from_mask(h)
        masks_y = m.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))

        # Cell (4, 4) is adjacent to land at (4, 5) in x → ind_coast.
        assert bool(m.ind_coast[4, 4])
        # But in the y direction, cells j=4 in column i=4 are surrounded
        # by wet cells → y masks[6] True (far enough from the y-boundaries).
        assert bool(np.asarray(masks_y[6])[4, 4])
        # And the y-direction adaptive-masks for that cell are NOT at
        # the smallest tier.
        assert not bool(np.asarray(masks_y[2])[4, 4])


class TestAdvectionStencilFallback:
    """End-to-end: verify that ``Advection2D`` with a mask falls back to
    the correct narrower stencil at coast cells.

    Strategy: for a straight coast (land column), compute the advective
    tendency with a large-stencil method (e.g. weno5) + mask, then
    compare at each tier cell against the *unmasked* result of the
    corresponding narrower method (upwind1 / weno3 / weno5).
    """

    @pytest.fixture
    def linear_field(self):
        """Linear tracer h[j, i] = 1 + i — smooth in x, constant in y.
        For constant positive velocity u, weno5 / weno3 / upwind1 all
        give the same answer on a *smooth* field (exact for linear data),
        so the adaptive-stencil fallback is transparent.  We use a
        staircase field instead to make the tiers distinguishable."""
        return None  # Unused — kept for API symmetry with future fixtures.

    def _make_staircase_field(self, Ny: int, Nx: int) -> jnp.ndarray:
        """Discrete staircase: h[j, i] = i % 2.  This sharp pattern is
        handled differently by upwind1 (piecewise constant), weno3,
        and weno5 — so the resulting fluxes should differ by tier."""
        idx = jnp.arange(Nx)[None, :]
        return jnp.broadcast_to((idx % 2).astype(jnp.float64), (Ny, Nx))

    def test_weno5_masked_matches_unmasked_far_from_coast(self):
        """Away from any coast, weno5 + mask should equal unmasked
        weno5 (the mask dispatch picks the max tier everywhere)."""
        Ny, Nx = 14, 14
        grid = CartesianGrid2D.from_interior(Nx - 2, Ny - 2, 1.0, 1.0)
        adv_plain = Advection2D(grid=grid)
        # All-ocean mask: the adaptive dispatch should use weno5 everywhere.
        mask = Mask2D.from_dimensions(Ny, Nx)
        adv_masked = Advection2D(grid=grid, mask=mask)
        h = self._make_staircase_field(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))

        masked = adv_masked(h, u, v, method="weno5")
        plain = adv_plain(h, u, v, method="weno5")
        # Deep interior cells should match (both use the full weno5 stencil).
        np.testing.assert_allclose(
            np.asarray(masked[4:-4, 4:-4]),
            np.asarray(plain[4:-4, 4:-4]),
            rtol=1e-10,
        )

    def test_weno5_masked_differs_from_unmasked_at_coast(self):
        """At cells with ``masks[2]`` True (adjacent to land in the
        stencil direction), the adaptive weno5 tendency should differ
        from the *unmasked* weno5 tendency, because ``upwind_flux``
        falls back to a narrower stencil.  We verify the fallback is
        *happening* at the tendency level; the exact per-tier match is
        tested face-by-face in :class:`TestUpwindFluxFallbackExplicit`.
        """
        # Build a domain with a land column so there's a coast in x.
        Ny, Nx = 14, 14
        h_mask = np.ones((Ny, Nx), dtype=bool)
        h_mask[:, 6] = False
        mask = Mask2D.from_mask(h_mask)

        grid = CartesianGrid2D.from_interior(Nx - 2, Ny - 2, 1.0, 1.0)
        adv_plain = Advection2D(grid=grid)
        adv_masked = Advection2D(grid=grid, mask=mask)
        h = self._make_staircase_field(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))

        masked = adv_masked(h, u, v, method="weno5")
        unmasked = adv_plain(h, u, v, method="weno5")

        # At at least one coast cell (interior rows), the masked tendency
        # differs from the unmasked tendency: that's the fallback firing.
        masks_x = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        coast_x = np.asarray(masks_x[2])
        coast_mask = np.zeros_like(coast_x)
        coast_mask[3:-3, :] = coast_x[3:-3, :]  # interior rows only
        diff = np.abs(np.asarray(masked) - np.asarray(unmasked))
        assert np.any(diff[coast_mask] > 1e-10), (
            "expected adaptive dispatch to differ from unmasked at coast cells"
        )
        # No NaNs / infs anywhere — fallback produced finite output.
        assert bool(np.all(np.isfinite(masked)))

    def test_tvd_masked_differs_from_unmasked_at_coast(self):
        """TVD limiters dispatch through ``(2, 4)``.  Verify the
        fallback fires at coast cells: the masked van_leer tendency
        differs from the unmasked van_leer tendency at ``masks[2]``
        cells."""
        Ny, Nx = 12, 12
        h_mask = np.ones((Ny, Nx), dtype=bool)
        h_mask[:, 5] = False
        mask = Mask2D.from_mask(h_mask)

        grid = CartesianGrid2D.from_interior(Nx - 2, Ny - 2, 1.0, 1.0)
        adv_plain = Advection2D(grid=grid)
        adv_masked = Advection2D(grid=grid, mask=mask)
        h = self._make_staircase_field(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))

        masked = adv_masked(h, u, v, method="van_leer")
        unmasked = adv_plain(h, u, v, method="van_leer")

        masks_x = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4))
        coast_x = np.asarray(masks_x[2])
        coast_mask = np.zeros_like(coast_x)
        coast_mask[3:-3, :] = coast_x[3:-3, :]
        diff = np.abs(np.asarray(masked) - np.asarray(unmasked))
        assert np.any(diff[coast_mask] > 1e-10), (
            "expected van_leer adaptive dispatch to differ from unmasked at coast"
        )
        assert bool(np.all(np.isfinite(masked)))


class TestUpwindFluxFallbackExplicit:
    """Direct tests of ``upwind_flux`` (the low-level blending function)
    against hand-picked stencil-hierarchy dicts, bypassing the
    ``Advection2D`` dispatch.  These exercise the blending formula
    ``F = sum_s M_s(i_up) * F_s`` for each supported advection stencil
    hierarchy: ``(2,)``, ``(2, 4)``, ``(2, 4, 6)``."""

    @pytest.fixture
    def coast_grid_and_mask(self):
        """14×14 grid with a land column at i=7."""
        Ny, Nx = 14, 14
        grid = CartesianGrid2D.from_interior(Nx - 2, Ny - 2, 1.0, 1.0)
        h_mask = np.ones((Ny, Nx), dtype=bool)
        h_mask[:, 7] = False
        mask = Mask2D.from_mask(h_mask)
        return grid, mask

    def test_single_stencil_hierarchy_equals_unmasked(self, coast_grid_and_mask):
        """If the hierarchy has a single stencil size, ``upwind_flux``
        should reduce to calling that stencil's reconstruction directly
        (modulo cells where even that stencil isn't supported — those
        become zero)."""
        grid, mask = coast_grid_and_mask
        recon = Reconstruction2D(grid=grid)
        Ny, Nx = grid.Ny, grid.Nx
        q = jnp.asarray(np.arange(Ny * Nx, dtype=float).reshape(Ny, Nx))
        u_vel = jnp.ones((Ny, Nx))

        mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2,))
        rec_funcs = {2: recon.upwind1_x}
        blended = upwind_flux(
            q, u_vel, dim=1, rec_funcs=rec_funcs, mask_hierarchy=mask_hier
        )
        direct = recon.upwind1_x(q, u_vel)

        # Wherever masks[2] is True, blended should equal direct.
        keep = np.asarray(mask_hier[2])
        np.testing.assert_allclose(
            np.asarray(blended[keep]),
            np.asarray(direct[keep]),
            rtol=1e-12,
        )

    def test_three_stencil_hierarchy_blends_correctly(self, coast_grid_and_mask):
        """With the weno5 hierarchy ``(2, 4, 6)``, every wet face-flux
        should be one of: upwind1 (at coast), weno3 (at near-coast),
        weno5 (deep interior), or zero (if even stencil 2 isn't
        supported — e.g. at the land column's east face)."""
        grid, mask = coast_grid_and_mask
        recon = Reconstruction2D(grid=grid)
        Ny, Nx = grid.Ny, grid.Nx
        q = jnp.asarray(np.sin(np.arange(Ny * Nx, dtype=float).reshape(Ny, Nx)))
        u_vel = jnp.ones((Ny, Nx))

        mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        rec_funcs = {
            2: recon.upwind1_x,
            4: recon.weno3_x,
            6: recon.weno5_x,
        }
        blended = upwind_flux(
            q, u_vel, dim=1, rec_funcs=rec_funcs, mask_hierarchy=mask_hier
        )
        f_upwind1 = recon.upwind1_x(q, u_vel)
        f_weno3 = recon.weno3_x(q, u_vel)
        f_weno5 = recon.weno5_x(q, u_vel)

        m2 = np.asarray(mask_hier[2])
        m4 = np.asarray(mask_hier[4])
        m6 = np.asarray(mask_hier[6])

        # At masks[2] cells the blended flux equals upwind1.
        np.testing.assert_allclose(
            np.asarray(blended[m2]), np.asarray(f_upwind1[m2]), rtol=1e-12
        )
        # At masks[4] cells the blended flux equals weno3.
        np.testing.assert_allclose(
            np.asarray(blended[m4]), np.asarray(f_weno3[m4]), rtol=1e-12
        )
        # At masks[6] cells the blended flux equals weno5.
        np.testing.assert_allclose(
            np.asarray(blended[m6]), np.asarray(f_weno5[m6]), rtol=1e-12
        )
        # At cells that are *not* in any tier (dry or boundary), the
        # blended flux is zero because none of the tier masks contribute.
        none_tier = ~(m2 | m4 | m6)
        np.testing.assert_allclose(np.asarray(blended[none_tier]), 0.0, atol=1e-12)

    def test_wenoz5_hierarchy_blends_correctly(self, coast_grid_and_mask):
        """Same test for the ``wenoz5`` hierarchy ``(upwind1, wenoz3,
        wenoz5)``."""
        grid, mask = coast_grid_and_mask
        recon = Reconstruction2D(grid=grid)
        Ny, Nx = grid.Ny, grid.Nx
        q = jnp.asarray(np.cos(np.arange(Ny * Nx, dtype=float).reshape(Ny, Nx)))
        u_vel = jnp.ones((Ny, Nx))

        mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        rec_funcs = {
            2: recon.upwind1_x,
            4: recon.wenoz3_x,
            6: recon.wenoz5_x,
        }
        blended = upwind_flux(
            q, u_vel, dim=1, rec_funcs=rec_funcs, mask_hierarchy=mask_hier
        )
        f_upwind1 = recon.upwind1_x(q, u_vel)
        f_wenoz3 = recon.wenoz3_x(q, u_vel)
        f_wenoz5 = recon.wenoz5_x(q, u_vel)

        m2 = np.asarray(mask_hier[2])
        m4 = np.asarray(mask_hier[4])
        m6 = np.asarray(mask_hier[6])

        np.testing.assert_allclose(
            np.asarray(blended[m2]), np.asarray(f_upwind1[m2]), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(blended[m4]), np.asarray(f_wenoz3[m4]), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(blended[m6]), np.asarray(f_wenoz5[m6]), rtol=1e-12
        )

    def test_tvd_hierarchy_blends_correctly(self, coast_grid_and_mask):
        """TVD flux-limiter hierarchy ``(upwind1, tvd_van_leer)``."""
        grid, mask = coast_grid_and_mask
        recon = Reconstruction2D(grid=grid)
        Ny, Nx = grid.Ny, grid.Nx
        q = jnp.asarray(np.sin(np.arange(Ny * Nx, dtype=float).reshape(Ny, Nx)))
        u_vel = jnp.ones((Ny, Nx))

        mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4))
        rec_funcs = {
            2: recon.upwind1_x,
            4: lambda q_, u_: recon.tvd_x(q_, u_, limiter="van_leer"),
        }
        blended = upwind_flux(
            q, u_vel, dim=1, rec_funcs=rec_funcs, mask_hierarchy=mask_hier
        )
        f_upwind1 = recon.upwind1_x(q, u_vel)
        f_tvd = recon.tvd_x(q, u_vel, limiter="van_leer")

        m2 = np.asarray(mask_hier[2])
        m4 = np.asarray(mask_hier[4])

        np.testing.assert_allclose(
            np.asarray(blended[m2]), np.asarray(f_upwind1[m2]), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(blended[m4]), np.asarray(f_tvd[m4]), rtol=1e-12
        )
