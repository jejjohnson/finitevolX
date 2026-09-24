"""Tests for AbstractForcing and the forcing composition wrappers."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    AbstractForcing,
    ForcingProduct,
    ForcingSum,
    TimeVaryingForcing,
)

jax.config.update("jax_enable_x64", True)


class Damp(AbstractForcing):
    """Scalar forcing ``dq = -r * q``."""

    r: float

    def __call__(self, q):
        return -self.r * q


class Push(AbstractForcing):
    """Momentum-style forcing returning ``(du, dv) = (a * tx, a * ty)``."""

    a: float

    def __call__(self, tx, ty, scale=1.0):
        return self.a * scale * tx, self.a * scale * ty


class TestAbstractForcing:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractForcing()  # type: ignore[abstract]

    def test_subclass_without_call_is_abstract(self):
        class Incomplete(AbstractForcing):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_is_pytree(self):
        leaves = jax.tree.leaves(Damp(r=jnp.array(0.5)))
        assert len(leaves) == 1


class TestForcingSum:
    def test_scalar_sum(self):
        q = jnp.arange(4.0)
        total = ForcingSum(Damp(0.1), Damp(0.2), Damp(0.3))
        np.testing.assert_allclose(total(q), -0.6 * q, rtol=1e-12)

    def test_tuple_outputs_sum_leafwise(self):
        tx, ty = jnp.ones(3), 2.0 * jnp.ones(3)
        du, dv = ForcingSum(Push(1.0), Push(3.0))(tx, ty)
        np.testing.assert_allclose(du, 4.0)
        np.testing.assert_allclose(dv, 8.0)

    def test_kwargs_forwarded(self):
        tx = ty = jnp.ones(2)
        du, _ = ForcingSum(Push(1.0), Push(1.0))(tx, ty, scale=5.0)
        np.testing.assert_allclose(du, 10.0)

    def test_single_member_is_identity(self):
        q = jnp.ones(3)
        np.testing.assert_allclose(ForcingSum(Damp(2.0))(q), Damp(2.0)(q))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ForcingSum()

    def test_nested(self):
        q = jnp.ones(2)
        nested = ForcingSum(ForcingSum(Damp(1.0), Damp(2.0)), Damp(3.0))
        np.testing.assert_allclose(nested(q), -6.0)

    def test_filter_jit(self):
        q = jnp.ones(3)
        total = ForcingSum(Damp(1.0), Damp(1.0))
        np.testing.assert_allclose(eqx.filter_jit(total)(q), -2.0)


class TestForcingProduct:
    def test_scalar_modulator(self):
        q = jnp.ones(3)
        np.testing.assert_allclose(ForcingProduct(Damp(1.0), 0.5)(q), -0.5)

    def test_array_modulator_sponge(self):
        q = jnp.ones(4)
        sponge = jnp.array([0.0, 0.25, 0.75, 1.0])
        np.testing.assert_allclose(ForcingProduct(Damp(2.0), sponge)(q), -2.0 * sponge)

    def test_callable_modulator_receives_same_args(self):
        tx, ty = jnp.ones(2), jnp.ones(2)

        def alpha(tx, ty, scale=1.0):
            return scale

        du, dv = ForcingProduct(Push(1.0), alpha)(tx, ty, scale=3.0)
        # inner gives 3 * tx, modulator gives 3 -> 9
        np.testing.assert_allclose(du, 9.0)
        np.testing.assert_allclose(dv, 9.0)

    def test_tuple_output_every_leaf_scaled(self):
        du, dv = ForcingProduct(Push(1.0), 2.0)(jnp.ones(2), jnp.ones(2))
        np.testing.assert_allclose(du, 2.0)
        np.testing.assert_allclose(dv, 2.0)

    def test_grad_through_array_modulator(self):
        """A trainable modulator field is differentiable via eqx.filter_grad."""
        q = jnp.array([1.0, 2.0])
        op = ForcingProduct(Damp(1.0), jnp.array([1.0, 1.0]))

        @eqx.filter_grad
        def loss(op):
            return jnp.sum(op(q))

        grads = loss(op)
        np.testing.assert_allclose(grads.modulator, -q)

    def test_composes_with_sum(self):
        q = jnp.ones(2)
        total = ForcingSum(ForcingProduct(Damp(1.0), 0.5), Damp(1.0))
        np.testing.assert_allclose(total(q), -1.5)


class TestTimeVaryingForcing:
    def test_single_field(self):
        tv = TimeVaryingForcing(Damp(1.0), lambda t: t * jnp.ones(3))
        np.testing.assert_allclose(tv(2.0), -2.0)

    def test_tuple_fields_unpacked_and_kwargs_forwarded(self):
        tv = TimeVaryingForcing(
            Push(1.0), lambda t: (t * jnp.ones(2), 2.0 * t * jnp.ones(2))
        )
        du, dv = tv(1.5, scale=2.0)
        np.testing.assert_allclose(du, 3.0)
        np.testing.assert_allclose(dv, 6.0)

    def test_analytic_seasonal_cycle(self):
        base = jnp.ones(4)
        period = 365.0
        tv = TimeVaryingForcing(
            Damp(1.0),
            lambda t: base * (1.0 + 0.3 * jnp.sin(2 * jnp.pi * t / period)),
        )
        np.testing.assert_allclose(tv(0.0), -1.0, atol=1e-12)
        np.testing.assert_allclose(tv(period / 4), -1.3, atol=1e-12)

    def test_diffrax_linear_interpolation(self):
        ts = jnp.array([0.0, 1.0, 2.0])
        ys = jnp.stack([jnp.zeros((3, 3)), jnp.ones((3, 3)), 3.0 * jnp.ones((3, 3))])
        path = dfx.LinearInterpolation(ts=ts, ys=ys)
        tv = TimeVaryingForcing(Damp(1.0), path.evaluate)
        np.testing.assert_allclose(tv(0.5), -0.5, atol=1e-12)
        np.testing.assert_allclose(tv(1.5), -2.0, atol=1e-12)

    def test_jit_and_grad_wrt_time(self):
        ts = jnp.array([0.0, 1.0])
        ys = jnp.stack([jnp.zeros(2), 2.0 * jnp.ones(2)])
        path = dfx.LinearInterpolation(ts=ts, ys=ys)
        tv = TimeVaryingForcing(Damp(1.0), path.evaluate)

        @eqx.filter_jit
        def total(t):
            return jnp.sum(tv(t))

        np.testing.assert_allclose(total(0.5), -2.0, atol=1e-12)
        # d/dt sum(-(2 t) * ones(2)) = -4
        np.testing.assert_allclose(jax.grad(total)(0.5), -4.0, atol=1e-12)

    def test_sum_of_time_varying(self):
        tv1 = TimeVaryingForcing(Damp(1.0), lambda t: t * jnp.ones(2))
        tv2 = TimeVaryingForcing(Damp(2.0), lambda t: jnp.ones(2))
        np.testing.assert_allclose(ForcingSum(tv1, tv2)(3.0), -5.0)
