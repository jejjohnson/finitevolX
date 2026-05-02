"""Tests for the equation of state module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx._src.operators.eos import (
    buoyancy,
    linear_density,
    linear_density_anomaly,
    linear_drho_dS,
    linear_drho_dT,
    reduced_gravity,
)

jax.config.update("jax_enable_x64", True)

RHO_0 = 1025.0
ALPHA = 2e-4
BETA = 7e-4
T_REF = 10.0
S_REF = 35.0


class TestLinearDensity:
    def test_reference_gives_rho0(self):
        T = jnp.full((8, 8), T_REF)
        S = jnp.full((8, 8), S_REF)
        rho = linear_density(T, S)
        np.testing.assert_allclose(rho, RHO_0, rtol=1e-12)

    def test_warm_anomaly_decreases_density(self):
        T = jnp.full((8, 8), T_REF + 1.0)
        S = jnp.full((8, 8), S_REF)
        rho = linear_density(T, S)
        assert float(jnp.mean(rho)) < RHO_0

    def test_salt_anomaly_increases_density(self):
        T = jnp.full((8, 8), T_REF)
        S = jnp.full((8, 8), S_REF + 1.0)
        rho = linear_density(T, S)
        assert float(jnp.mean(rho)) > RHO_0

    def test_known_value(self):
        T = jnp.full((4, 4), T_REF + 5.0)
        S = jnp.full((4, 4), S_REF + 2.0)
        rho = linear_density(T, S)
        expected = RHO_0 * (1.0 - ALPHA * 5.0 + BETA * 2.0)
        np.testing.assert_allclose(rho, expected, rtol=1e-12)

    def test_batch_dims(self):
        T = jnp.ones((3, 8, 8)) * T_REF
        S = jnp.ones((3, 8, 8)) * S_REF
        rho = linear_density(T, S)
        assert rho.shape == (3, 8, 8)
        np.testing.assert_allclose(rho, RHO_0, rtol=1e-12)

    def test_jit_compatible(self):
        T = jnp.full((8, 8), T_REF + 1.0)
        S = jnp.full((8, 8), S_REF)
        rho_eager = linear_density(T, S)
        rho_jit = jax.jit(linear_density)(T, S)
        np.testing.assert_allclose(rho_eager, rho_jit, rtol=1e-12)


class TestLinearDensityAnomaly:
    def test_reference_gives_zero(self):
        T = jnp.full((8, 8), T_REF)
        S = jnp.full((8, 8), S_REF)
        rho_prime = linear_density_anomaly(T, S)
        np.testing.assert_allclose(rho_prime, 0.0, atol=1e-12)

    def test_warm_anomaly_is_negative(self):
        T = jnp.full((8, 8), T_REF + 1.0)
        S = jnp.full((8, 8), S_REF)
        rho_prime = linear_density_anomaly(T, S)
        assert float(jnp.mean(rho_prime)) < 0.0

    def test_salt_anomaly_is_positive(self):
        T = jnp.full((8, 8), T_REF)
        S = jnp.full((8, 8), S_REF + 1.0)
        rho_prime = linear_density_anomaly(T, S)
        assert float(jnp.mean(rho_prime)) > 0.0

    def test_equals_density_minus_rho0(self):
        T = jnp.full((8, 8), T_REF + 3.0)
        S = jnp.full((8, 8), S_REF - 1.0)
        rho = linear_density(T, S)
        rho_prime = linear_density_anomaly(T, S)
        np.testing.assert_allclose(rho_prime, rho - RHO_0, rtol=1e-12)


class TestLinearDrhoDT:
    def test_sign_is_negative(self):
        assert linear_drho_dT() < 0.0

    def test_known_value(self):
        expected = -RHO_0 * ALPHA
        np.testing.assert_allclose(linear_drho_dT(), expected, rtol=1e-12)

    def test_custom_params(self):
        np.testing.assert_allclose(
            linear_drho_dT(rho_0=1000.0, alpha=1e-4), -0.1, rtol=1e-12
        )


class TestLinearDrhoDS:
    def test_sign_is_positive(self):
        assert linear_drho_dS() > 0.0

    def test_known_value(self):
        expected = RHO_0 * BETA
        np.testing.assert_allclose(linear_drho_dS(), expected, rtol=1e-12)

    def test_custom_params(self):
        np.testing.assert_allclose(
            linear_drho_dS(rho_0=1000.0, beta=1e-3), 1.0, rtol=1e-12
        )


class TestBuoyancy:
    def test_reference_density_gives_zero(self):
        rho = jnp.full((8, 8), RHO_0)
        b = buoyancy(rho)
        np.testing.assert_allclose(b, 0.0, atol=1e-12)

    def test_lighter_gives_positive(self):
        rho = jnp.full((8, 8), RHO_0 - 1.0)
        b = buoyancy(rho)
        assert float(jnp.mean(b)) > 0.0

    def test_heavier_gives_negative(self):
        rho = jnp.full((8, 8), RHO_0 + 1.0)
        b = buoyancy(rho)
        assert float(jnp.mean(b)) < 0.0

    def test_known_value(self):
        g = 9.80665
        rho = jnp.full((4, 4), RHO_0 + 0.5)
        b = buoyancy(rho)
        expected = -g * 0.5 / RHO_0
        np.testing.assert_allclose(b, expected, rtol=1e-12)

    def test_jit_compatible(self):
        rho = jnp.full((8, 8), RHO_0 + 1.0)
        b_eager = buoyancy(rho)
        b_jit = jax.jit(buoyancy)(rho)
        np.testing.assert_allclose(b_eager, b_jit, rtol=1e-12)


class TestReducedGravity:
    def test_equal_layers_gives_zero(self):
        rho = jnp.full((8, 8), RHO_0)
        gp = reduced_gravity(rho, rho)
        np.testing.assert_allclose(gp, 0.0, atol=1e-12)

    def test_stable_stratification_positive(self):
        rho_up = jnp.full((8, 8), RHO_0)
        rho_dn = jnp.full((8, 8), RHO_0 + 1.0)
        gp = reduced_gravity(rho_up, rho_dn)
        assert float(jnp.mean(gp)) > 0.0

    def test_unstable_stratification_negative(self):
        rho_up = jnp.full((8, 8), RHO_0 + 1.0)
        rho_dn = jnp.full((8, 8), RHO_0)
        gp = reduced_gravity(rho_up, rho_dn)
        assert float(jnp.mean(gp)) < 0.0

    def test_known_value(self):
        g = 9.80665
        rho_up = jnp.full((4, 4), 1024.0)
        rho_dn = jnp.full((4, 4), 1026.0)
        gp = reduced_gravity(rho_up, rho_dn)
        expected = g * 2.0 / RHO_0
        np.testing.assert_allclose(gp, expected, rtol=1e-12)

    def test_batch_dims(self):
        rho_up = jnp.full((3, 8, 8), RHO_0)
        rho_dn = jnp.full((3, 8, 8), RHO_0 + 1.0)
        gp = reduced_gravity(rho_up, rho_dn)
        assert gp.shape == (3, 8, 8)

    def test_jit_compatible(self):
        rho_up = jnp.full((8, 8), RHO_0)
        rho_dn = jnp.full((8, 8), RHO_0 + 1.0)
        gp_eager = reduced_gravity(rho_up, rho_dn)
        gp_jit = jax.jit(reduced_gravity)(rho_up, rho_dn)
        np.testing.assert_allclose(gp_eager, gp_jit, rtol=1e-12)
