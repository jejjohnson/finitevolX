"""Tests for mask-aware sample statistics over a leading sample axis.

The contract these back is per-gridpoint standardisation on a masked
domain: wet cells get the ordinary mean/std, land never contributes,
land sentinels never leak, and the returned ``(loc, scale)`` pair is
always safe to divide by.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.mask.cartesian import Mask2D, Mask3D
from finitevolx._src.operators.reductions import (
    masked_mean,
    masked_moments,
    masked_std,
)

NY, NX = 6, 7
NSAMPLE = 9


@pytest.fixture
def samples():
    rng = np.random.RandomState(0)
    return jnp.asarray(rng.randn(NSAMPLE, NY, NX), dtype=jnp.float32)


@pytest.fixture
def land():
    """Boolean wet-cell layout with a land block and a full land row."""
    wet = np.ones((NY, NX), dtype=bool)
    wet[1:3, 2:4] = False
    wet[4, :] = False
    return wet


@pytest.fixture
def mask2d(land):
    return Mask2D.from_mask(jnp.asarray(land))


class TestUnmaskedMatchesNumpy:
    def test_mean_matches_jnp_mean(self, samples):
        np.testing.assert_allclose(
            masked_mean(samples), jnp.mean(samples, axis=0), rtol=1e-6
        )

    def test_std_matches_jnp_std(self, samples):
        np.testing.assert_allclose(
            masked_std(samples), jnp.std(samples, axis=0), rtol=1e-5
        )

    @pytest.mark.parametrize("ddof", [0, 1, 2])
    def test_std_honours_ddof(self, samples, ddof):
        np.testing.assert_allclose(
            masked_std(samples, ddof=ddof),
            np.std(np.asarray(samples), axis=0, ddof=ddof),
            rtol=1e-5,
        )

    def test_scalar_reduction_over_all_axes(self, samples):
        got = masked_mean(samples, axis=(0, -2, -1))
        assert got.shape == ()
        np.testing.assert_allclose(got, np.asarray(samples).mean(), rtol=1e-6)

    def test_moments_agree_with_the_separate_calls(self, samples, mask2d):
        mean, std = masked_moments(samples, mask2d)
        np.testing.assert_allclose(mean, masked_mean(samples, mask2d), rtol=1e-6)
        np.testing.assert_allclose(std, masked_std(samples, mask2d), rtol=1e-6)


class TestMaskedPerGridpoint:
    def test_wet_cells_match_the_unmasked_statistics(self, samples, mask2d, land):
        mean = np.asarray(masked_mean(samples, mask2d))
        std = np.asarray(masked_std(samples, mask2d))
        ref_mean = np.asarray(samples).mean(axis=0)
        ref_std = np.asarray(samples).std(axis=0)
        wet = np.asarray(mask2d.h)
        np.testing.assert_allclose(mean[wet], ref_mean[wet], rtol=1e-6)
        np.testing.assert_allclose(std[wet], ref_std[wet], rtol=1e-5)

    def test_dry_cells_are_identity_loc_scale(self, samples, mask2d):
        mean = np.asarray(masked_mean(samples, mask2d))
        std = np.asarray(masked_std(samples, mask2d))
        dry = ~np.asarray(mask2d.h)
        assert dry.any()
        np.testing.assert_array_equal(mean[dry], 0.0)
        np.testing.assert_array_equal(std[dry], 1.0)

    def test_standardising_leaves_land_untouched(self, samples, mask2d):
        """``(x - loc) / scale`` must round-trip, land included."""
        mean, std = masked_moments(samples, mask2d)
        standardised = (samples - mean) / std
        recovered = standardised * std + mean
        np.testing.assert_allclose(recovered, samples, rtol=1e-5, atol=1e-6)

    def test_land_nan_does_not_leak_into_wet_cells(self, samples, mask2d):
        poisoned = jnp.where(mask2d.h, samples, jnp.nan)
        mean = np.asarray(masked_mean(poisoned, mask2d))
        std = np.asarray(masked_std(poisoned, mask2d))
        assert np.isfinite(mean).all()
        assert np.isfinite(std).all()
        np.testing.assert_allclose(mean, masked_mean(samples, mask2d), rtol=1e-6)

    def test_land_inf_does_not_leak_into_wet_cells(self, samples, mask2d):
        poisoned = jnp.where(mask2d.h, samples, jnp.inf)
        assert np.isfinite(np.asarray(masked_mean(poisoned, mask2d))).all()
        assert np.isfinite(np.asarray(masked_std(poisoned, mask2d))).all()


class TestMaskedScalarReduction:
    def test_divides_by_the_wet_count_not_the_cell_count(self, samples, mask2d):
        got = float(masked_mean(samples, mask2d, axis=(0, -2, -1)))
        wet = np.asarray(mask2d.h)
        expected = np.asarray(samples)[:, wet].mean()
        assert got == pytest.approx(expected, rel=1e-5)
        # The naive all-cells mean is a genuinely different number.
        assert got != pytest.approx(np.asarray(samples).mean(), rel=1e-3)

    def test_std_excludes_land_from_the_deviation_sum(self, samples, mask2d):
        """Land must not contribute ``(0 - mean)**2`` to the variance."""
        got = float(masked_std(samples, mask2d, axis=(0, -2, -1)))
        wet = np.asarray(mask2d.h)
        expected = np.asarray(samples)[:, wet].std()
        assert got == pytest.approx(expected, rel=1e-5)

    def test_land_sentinels_do_not_leak_into_the_scalar(self, samples, mask2d):
        poisoned = jnp.where(mask2d.h, samples, jnp.nan)
        got = float(masked_mean(poisoned, mask2d, axis=(0, -2, -1)))
        assert np.isfinite(got)
        assert got == pytest.approx(
            float(masked_mean(samples, mask2d, axis=(0, -2, -1))), rel=1e-6
        )


class TestVarianceFloor:
    def test_constant_field_returns_eps_not_zero(self):
        constant = jnp.ones((NSAMPLE, NY, NX))
        std = masked_std(constant, eps=1e-6)
        np.testing.assert_allclose(std, 1e-6)
        assert np.all(np.asarray(std) > 0.0)

    def test_floor_does_not_disturb_a_well_resolved_std(self, samples):
        loose = masked_std(samples, eps=1e-30)
        tight = masked_std(samples, eps=1e-8)
        np.testing.assert_allclose(loose, tight, rtol=1e-6)

    def test_single_sample_with_ddof_one_is_degenerate(self):
        """``count - ddof <= 0`` yields the identity scale, not a divide-by-zero."""
        one = jnp.ones((1, NY, NX))
        std = masked_std(one, ddof=1)
        np.testing.assert_array_equal(np.asarray(std), 1.0)

    def test_dividing_by_the_scale_is_always_finite(self, samples, mask2d):
        constant = jnp.where(mask2d.h, 3.0, jnp.nan) * jnp.ones((NSAMPLE, 1, 1))
        mean, std = masked_moments(constant, mask2d)
        assert np.isfinite(
            np.asarray((constant - mean) / std)[:, np.asarray(mask2d.h)]
        ).all()


class TestStaggeringLocations:
    @pytest.mark.parametrize("location", ["h", "u", "v", "xy_corner"])
    def test_uses_the_named_c_grid_mask(self, samples, mask2d, location):
        mean = np.asarray(masked_mean(samples, mask2d, location=location))
        expected_dry = ~np.asarray(getattr(mask2d, location))
        np.testing.assert_array_equal(mean[expected_dry], 0.0)

    def test_u_and_v_masks_differ_from_h(self, samples, mask2d):
        """Otherwise the location kwarg would be silently doing nothing."""
        assert not np.array_equal(np.asarray(mask2d.u), np.asarray(mask2d.h))
        mean_h = np.asarray(masked_mean(samples, mask2d, location="h"))
        mean_u = np.asarray(masked_mean(samples, mask2d, location="u"))
        assert not np.allclose(mean_h, mean_u)

    def test_unknown_location_is_rejected(self, samples, mask2d):
        with pytest.raises(ValueError, match="unknown location"):
            masked_mean(samples, mask2d, location="nope")

    def test_w_location_requires_a_3d_mask(self, samples, mask2d):
        with pytest.raises(ValueError, match="no 'w' mask"):
            masked_mean(samples, mask2d, location="w")


class TestThreeDimensional:
    @pytest.fixture
    def mask3d(self):
        wet = np.ones((3, NY, NX), dtype=bool)
        wet[0, 2:4, 1:3] = False
        return Mask3D.from_mask(jnp.asarray(wet))

    @pytest.fixture
    def samples3d(self):
        rng = np.random.RandomState(1)
        return jnp.asarray(rng.randn(NSAMPLE, 3, NY, NX), dtype=jnp.float32)

    def test_per_gridpoint_statistics_on_a_3d_mask(self, samples3d, mask3d):
        mean = np.asarray(masked_mean(samples3d, mask3d))
        assert mean.shape == (3, NY, NX)
        wet = np.asarray(mask3d.h)
        np.testing.assert_allclose(
            mean[wet], np.asarray(samples3d).mean(axis=0)[wet], rtol=1e-6
        )
        np.testing.assert_array_equal(mean[~wet], 0.0)

    def test_per_level_scalar_reduction(self, samples3d, mask3d):
        """``axis=(0, -2, -1)`` leaves one number per vertical level."""
        got = masked_mean(samples3d, mask3d, axis=(0, -2, -1))
        assert got.shape == (3,)
        wet = np.asarray(mask3d.h)
        for k in range(3):
            expected = np.asarray(samples3d)[:, k][:, wet[k]].mean()
            assert float(got[k]) == pytest.approx(expected, rel=1e-5)

    def test_w_location_available_on_3d_masks(self, samples3d, mask3d):
        mean = np.asarray(masked_mean(samples3d, mask3d, location="w"))
        np.testing.assert_array_equal(mean[~np.asarray(mask3d.w)], 0.0)


class TestTransformsAndGradients:
    def test_jit_matches_eager(self, samples, mask2d):
        eager = masked_mean(samples, mask2d)
        jitted = jax.jit(masked_mean)(samples, mask2d)
        np.testing.assert_allclose(jitted, eager, rtol=1e-6)

    def test_std_under_jit(self, samples, mask2d):
        np.testing.assert_allclose(
            jax.jit(masked_std)(samples, mask2d), masked_std(samples, mask2d), rtol=1e-6
        )

    def test_grad_of_mean_is_mask_over_count(self, samples, mask2d):
        def total(x):
            return jnp.sum(masked_mean(x, mask2d))

        g = np.asarray(jax.grad(total)(samples))
        wet = np.asarray(mask2d.h)
        expected = np.broadcast_to(wet / NSAMPLE, g.shape)
        np.testing.assert_allclose(g, expected, rtol=1e-6, atol=1e-7)

    def test_grad_is_finite_with_land_sentinels(self, samples, mask2d):
        """A NaN on land must not poison the gradient of the wet cells."""
        poisoned = jnp.where(mask2d.h, samples, jnp.nan)

        def total(x):
            return jnp.sum(masked_mean(x, mask2d))

        assert np.isfinite(np.asarray(jax.grad(total)(poisoned))).all()

    def test_grad_of_std_is_finite_on_a_constant_field(self):
        """The ``eps`` floor keeps ``d/dx sqrt(var)`` finite at zero variance."""
        constant = jnp.ones((NSAMPLE, NY, NX))

        def total(x):
            return jnp.sum(masked_std(x))

        assert np.isfinite(np.asarray(jax.grad(total)(constant))).all()


class TestHalfPrecision:
    """float16 saturates at 65504, which breaks both statistics.

    The count of wet cells overflows to ``inf`` and the ``eps**2``
    variance floor underflows to zero, so accumulation is promoted to
    at least float32.
    """

    def test_a_large_wet_count_does_not_overflow(self):
        samples = jnp.full((2, 256, 256), 3.0, dtype=jnp.float16)
        got = masked_mean(samples, axis=(0, -2, -1))
        assert float(got) == pytest.approx(3.0, rel=1e-3)

    def test_the_count_really_would_overflow_in_float16(self):
        """Otherwise the test above would pass for the wrong reason."""
        naive = jnp.sum(jnp.ones((2, 256, 256), dtype=jnp.float16))
        assert not np.isfinite(float(naive))

    def test_a_constant_half_precision_field_gets_a_usable_scale(self):
        samples = jnp.full((8, 4, 4), 2.0, dtype=jnp.float16)
        std = masked_std(samples, axis=0)
        assert float(jnp.min(std)) > 0.0

    def test_half_precision_results_are_promoted(self):
        samples = jnp.full((4, 4, 4), 1.0, dtype=jnp.float16)
        assert masked_mean(samples).dtype == jnp.float32
        assert masked_std(samples).dtype == jnp.float32

    def test_single_precision_is_left_alone(self):
        samples = jnp.ones((4, 4, 4), dtype=jnp.float32)
        assert masked_mean(samples).dtype == jnp.float32
        assert masked_std(samples).dtype == jnp.float32
