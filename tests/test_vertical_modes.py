from __future__ import annotations

"""Tests for vertical coupling matrix and layer/mode transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.vertical.vertical_modes import (
    build_coupling_matrix,
    decompose_vertical_modes,
    layer_to_mode,
    mode_to_layer,
)

# ---------------------------------------------------------------------------
# build_coupling_matrix
# ---------------------------------------------------------------------------


class TestBuildCouplingMatrix:
    def test_single_layer_value(self):
        """A[0,0] = 1 / (H * g') for a single layer."""
        H = jnp.array([1000.0])
        g_prime = jnp.array([9.81])
        A = build_coupling_matrix(H, g_prime)
        expected = 1.0 / (1000.0 * 9.81)
        np.testing.assert_allclose(float(A[0, 0]), expected, rtol=1e-6)

    def test_single_layer_shape(self):
        H = jnp.array([1000.0])
        g_prime = jnp.array([9.81])
        A = build_coupling_matrix(H, g_prime)
        assert A.shape == (1, 1)

    def test_two_layer_shape(self):
        H = jnp.array([500.0, 500.0])
        g_prime = jnp.array([0.02, 0.02])
        A = build_coupling_matrix(H, g_prime)
        assert A.shape == (2, 2)

    def test_two_layer_symmetry(self):
        """A is symmetric when both layers have the same thickness."""
        H = jnp.array([500.0, 500.0])
        g_prime = jnp.array([0.01, 0.01])
        A = build_coupling_matrix(H, g_prime)
        np.testing.assert_allclose(np.array(A), np.array(A).T, atol=1e-12)

    def test_two_layer_values(self):
        """Check each entry of the 2-layer A matrix analytically."""
        H0, H1 = 500.0, 500.0
        g0, g1 = 0.02, 0.02
        H = jnp.array([H0, H1])
        g_prime = jnp.array([g0, g1])
        A = build_coupling_matrix(H, g_prime)

        # Top row
        np.testing.assert_allclose(
            float(A[0, 0]), 1.0 / (H0 * g0) + 1.0 / (H0 * g1), rtol=1e-6
        )
        np.testing.assert_allclose(float(A[0, 1]), -1.0 / (H0 * g1), rtol=1e-6)
        # Bottom row
        np.testing.assert_allclose(float(A[1, 1]), 1.0 / (H1 * g1), rtol=1e-6)
        np.testing.assert_allclose(float(A[1, 0]), -1.0 / (H1 * g1), rtol=1e-6)

    def test_three_layer_shape(self):
        H = jnp.array([300.0, 400.0, 300.0])
        g_prime = jnp.array([0.02, 0.015, 0.015])
        A = build_coupling_matrix(H, g_prime)
        assert A.shape == (3, 3)

    def test_three_layer_tridiagonal(self):
        """Off-tridiagonal elements must be zero."""
        H = jnp.array([300.0, 400.0, 300.0])
        g_prime = jnp.array([0.02, 0.015, 0.015])
        A = build_coupling_matrix(H, g_prime)
        A_np = np.array(A)
        # Check that only diagonal and first off-diagonals are non-zero
        for i in range(3):
            for j in range(3):
                if abs(i - j) > 1:
                    assert A_np[i, j] == 0.0, f"A[{i},{j}] should be 0"

    def test_three_layer_interior_row(self):
        """Check the interior row (i=1) of the 3-layer matrix."""
        H = jnp.array([300.0, 400.0, 300.0])
        g0, g1, g2 = 0.02, 0.015, 0.015
        g_prime = jnp.array([g0, g1, g2])
        A = build_coupling_matrix(H, g_prime)
        H1 = 400.0
        np.testing.assert_allclose(float(A[1, 0]), -1.0 / (H1 * g1), rtol=1e-6)
        np.testing.assert_allclose(
            float(A[1, 1]), 1.0 / H1 * (1.0 / g2 + 1.0 / g1), rtol=1e-6
        )
        np.testing.assert_allclose(float(A[1, 2]), -1.0 / (H1 * g2), rtol=1e-6)

    def test_positive_diagonal(self):
        """Diagonal elements must be positive (A is positive-semi-definite)."""
        H = jnp.array([300.0, 400.0, 300.0])
        g_prime = jnp.array([0.02, 0.015, 0.015])
        A = build_coupling_matrix(H, g_prime)
        assert jnp.all(jnp.diag(A) > 0)


# ---------------------------------------------------------------------------
# decompose_vertical_modes
# ---------------------------------------------------------------------------


class TestDecomposeVerticalModes:
    @pytest.fixture
    def two_layer_setup(self):
        H = jnp.array([500.0, 500.0])
        g_prime = jnp.array([0.02, 0.02])
        A = build_coupling_matrix(H, g_prime)
        f0 = 1e-4
        return A, f0

    @pytest.fixture
    def three_layer_setup(self):
        H = jnp.array([300.0, 400.0, 300.0])
        g_prime = jnp.array([0.02, 0.015, 0.015])
        A = build_coupling_matrix(H, g_prime)
        f0 = 1e-4
        return A, f0

    def test_single_layer_shapes(self):
        H = jnp.array([1000.0])
        g_prime = jnp.array([9.81])
        A = build_coupling_matrix(H, g_prime)
        radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0=1e-4)
        assert radii.shape == (1,)
        assert Cl2m.shape == (1, 1)
        assert Cm2l.shape == (1, 1)

    def test_two_layer_shapes(self, two_layer_setup):
        A, f0 = two_layer_setup
        radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0)
        assert radii.shape == (2,)
        assert Cl2m.shape == (2, 2)
        assert Cm2l.shape == (2, 2)

    def test_three_layer_shapes(self, three_layer_setup):
        A, f0 = three_layer_setup
        radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0)
        assert radii.shape == (3,)
        assert Cl2m.shape == (3, 3)
        assert Cm2l.shape == (3, 3)

    def test_round_trip_1d(self, two_layer_setup):
        """mode_to_layer(layer_to_mode(field)) ≈ field for a 1-D layer vector."""
        A, f0 = two_layer_setup
        _radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0)
        field = jnp.array([1.0, 3.0])
        reconstructed = mode_to_layer(layer_to_mode(field, Cl2m), Cm2l)
        np.testing.assert_allclose(np.array(reconstructed), np.array(field), atol=1e-5)

    def test_round_trip_3layer_1d(self, three_layer_setup):
        """Round-trip for 3-layer 1-D field."""
        A, f0 = three_layer_setup
        _radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0)
        field = jnp.array([1.0, 2.0, 3.0])
        reconstructed = mode_to_layer(layer_to_mode(field, Cl2m), Cm2l)
        np.testing.assert_allclose(np.array(reconstructed), np.array(field), atol=1e-5)

    def test_round_trip_2d_field(self, two_layer_setup):
        """Round-trip for a 2-layer, 2-D spatial field [nl, Ny, Nx]."""
        A, f0 = two_layer_setup
        _radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0)
        field = jnp.ones((2, 4, 4))
        reconstructed = mode_to_layer(layer_to_mode(field, Cl2m), Cm2l)
        np.testing.assert_allclose(np.array(reconstructed), np.array(field), atol=1e-5)

    def test_rossby_radii_positive(self, two_layer_setup):
        """All finite Rossby radii must be positive."""
        A, f0 = two_layer_setup
        radii, _Cl2m, _Cm2l = decompose_vertical_modes(A, f0)
        finite_radii = radii[np.isfinite(np.array(radii))]
        assert jnp.all(finite_radii > 0)

    def test_rossby_radius_single_layer_known_value(self):
        """For a 1-layer system, Rd = sqrt(g' H) / f0.

        The single-layer eigenvalue is λ = 1/(H*g'), so:
            Rd = 1 / (|f0| * sqrt(λ)) = sqrt(H*g') / |f0|
        """
        H = jnp.array([1000.0])
        g_prime_val = 0.02
        g_prime = jnp.array([g_prime_val])
        f0 = 1e-4
        A = build_coupling_matrix(H, g_prime)
        radii, _Cl2m, _Cm2l = decompose_vertical_modes(A, f0)
        # Expected: Rd = sqrt(H * g') / f0
        expected_rd = np.sqrt(float(H[0]) * g_prime_val) / abs(f0)
        np.testing.assert_allclose(float(radii[0]), expected_rd, rtol=1e-5)

    def test_rossby_radius_two_layer_physical_range(self):
        """Two-layer Rossby radii should be in a physically reasonable range.

        H = [500 m, 500 m], g' = [0.02 m/s², 0.02 m/s²], f0 = 1e-4 s⁻¹.
        Both deformation radii are O(10–100 km) for typical ocean parameters.

        The expected range [1 km, 1000 km] is deliberately broad; for this
        specific parameter set (H=500 m, g'=0.02 m/s²) the analytic values
        are approximately 20 km and 51 km (computed from the eigenvalues of A).
        """
        H = jnp.array([500.0, 500.0])
        g_prime = jnp.array([0.02, 0.02])
        f0 = 1e-4
        A = build_coupling_matrix(H, g_prime)
        radii, _Cl2m, _Cm2l = decompose_vertical_modes(A, f0)
        finite_radii = np.array(radii[np.isfinite(np.array(radii))])
        assert len(finite_radii) > 0, "Expected at least one finite Rossby radius"
        # All finite Rossby radii should be in [1 km, 1000 km]
        assert np.all(finite_radii > 1e3) and np.all(finite_radii < 1e6), (
            f"Rossby radii {finite_radii / 1e3} km outside expected range [1, 1000] km"
        )

    def test_cl2m_cm2l_inverse(self, two_layer_setup):
        """Cl2m @ Cm2l should be close to the identity matrix."""
        A, f0 = two_layer_setup
        _radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0)
        product = np.array(Cl2m) @ np.array(Cm2l)
        np.testing.assert_allclose(product, np.eye(2), atol=1e-5)

    def test_cl2m_cm2l_inverse_three_layer(self, three_layer_setup):
        """Cl2m @ Cm2l ≈ I for 3-layer case."""
        A, f0 = three_layer_setup
        _radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0)
        product = np.array(Cl2m) @ np.array(Cm2l)
        np.testing.assert_allclose(product, np.eye(3), atol=1e-5)


# ---------------------------------------------------------------------------
# layer_to_mode / mode_to_layer
# ---------------------------------------------------------------------------


class TestLayerModeTransforms:
    @pytest.fixture
    def transform_mats_2layer(self):
        H = jnp.array([500.0, 500.0])
        g_prime = jnp.array([0.02, 0.02])
        A = build_coupling_matrix(H, g_prime)
        _radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0=1e-4)
        return Cl2m, Cm2l

    def test_layer_to_mode_shape_1d(self, transform_mats_2layer):
        Cl2m, _ = transform_mats_2layer
        field = jnp.ones((2,))
        result = layer_to_mode(field, Cl2m)
        assert result.shape == (2,)

    def test_layer_to_mode_shape_2d(self, transform_mats_2layer):
        Cl2m, _ = transform_mats_2layer
        field = jnp.ones((2, 6, 6))
        result = layer_to_mode(field, Cl2m)
        assert result.shape == (2, 6, 6)

    def test_mode_to_layer_shape_1d(self, transform_mats_2layer):
        _, Cm2l = transform_mats_2layer
        field = jnp.ones((2,))
        result = mode_to_layer(field, Cm2l)
        assert result.shape == (2,)

    def test_mode_to_layer_shape_2d(self, transform_mats_2layer):
        _, Cm2l = transform_mats_2layer
        field = jnp.ones((2, 6, 6))
        result = mode_to_layer(field, Cm2l)
        assert result.shape == (2, 6, 6)

    def test_identity_transform(self):
        """With identity matrices, transforms return the input unchanged."""
        field = jnp.array([1.0, 2.0, 3.0])
        I = jnp.eye(3)
        np.testing.assert_allclose(
            np.array(layer_to_mode(field, I)), np.array(field), atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(mode_to_layer(field, I)), np.array(field), atol=1e-6
        )

    def test_round_trip_preserves_spatial_structure(self, transform_mats_2layer):
        """Round-trip should preserve spatially varying 2-D layer fields."""
        Cl2m, Cm2l = transform_mats_2layer
        ny, nx = 8, 8
        rng = np.random.default_rng(42)
        field = jnp.array(rng.standard_normal((2, ny, nx)))
        reconstructed = mode_to_layer(layer_to_mode(field, Cl2m), Cm2l)
        np.testing.assert_allclose(np.array(reconstructed), np.array(field), atol=1e-5)


# ---------------------------------------------------------------------------
# JAX-transform compatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Verify that all four public functions are compatible with jax.jit."""

    def test_jit_build_coupling_matrix(self):
        """jax.jit(build_coupling_matrix) matches eager output."""
        H = jnp.array([500.0, 500.0])
        g_prime = jnp.array([0.02, 0.02])
        A_eager = build_coupling_matrix(H, g_prime)
        A_jit = jax.jit(build_coupling_matrix)(H, g_prime)
        np.testing.assert_allclose(np.array(A_jit), np.array(A_eager), rtol=1e-6)

    def test_jit_build_coupling_matrix_three_layer(self):
        """jax.jit(build_coupling_matrix) works for the 3-layer case."""
        H = jnp.array([300.0, 400.0, 300.0])
        g_prime = jnp.array([0.02, 0.015, 0.015])
        A_eager = build_coupling_matrix(H, g_prime)
        A_jit = jax.jit(build_coupling_matrix)(H, g_prime)
        np.testing.assert_allclose(np.array(A_jit), np.array(A_eager), rtol=1e-6)

    def test_jit_decompose_vertical_modes(self):
        """jax.jit(decompose_vertical_modes) matches eager output."""
        H = jnp.array([500.0, 500.0])
        g_prime = jnp.array([0.02, 0.02])
        A = build_coupling_matrix(H, g_prime)
        f0 = jnp.array(1e-4)
        radii_eager, Cl2m_eager, Cm2l_eager = decompose_vertical_modes(A, f0)
        radii_jit, Cl2m_jit, Cm2l_jit = jax.jit(decompose_vertical_modes)(A, f0)
        np.testing.assert_allclose(
            np.array(radii_jit), np.array(radii_eager), rtol=1e-5
        )
        np.testing.assert_allclose(np.array(Cl2m_jit), np.array(Cl2m_eager), rtol=1e-5)
        np.testing.assert_allclose(np.array(Cm2l_jit), np.array(Cm2l_eager), rtol=1e-5)

    def test_jit_layer_to_mode(self):
        """jax.jit(layer_to_mode) matches eager output."""
        Cl2m = jnp.eye(2)
        field = jnp.array([1.0, 2.0])
        result_eager = layer_to_mode(field, Cl2m)
        result_jit = jax.jit(layer_to_mode)(field, Cl2m)
        np.testing.assert_allclose(
            np.array(result_jit), np.array(result_eager), rtol=1e-6
        )

    def test_jit_mode_to_layer(self):
        """jax.jit(mode_to_layer) matches eager output."""
        Cm2l = jnp.eye(2)
        field = jnp.array([1.0, 2.0])
        result_eager = mode_to_layer(field, Cm2l)
        result_jit = jax.jit(mode_to_layer)(field, Cm2l)
        np.testing.assert_allclose(
            np.array(result_jit), np.array(result_eager), rtol=1e-6
        )

    def test_jit_round_trip(self):
        """JIT-compiled round-trip mode_to_layer(layer_to_mode(field)) ≈ field."""
        H = jnp.array([500.0, 500.0])
        g_prime = jnp.array([0.02, 0.02])
        A = build_coupling_matrix(H, g_prime)
        _radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0=1e-4)
        field = jnp.array([1.0, 3.0])

        def round_trip(f: jnp.ndarray) -> jnp.ndarray:
            return mode_to_layer(layer_to_mode(f, Cl2m), Cm2l)

        result = jax.jit(round_trip)(field)
        np.testing.assert_allclose(np.array(result), np.array(field), atol=1e-5)
