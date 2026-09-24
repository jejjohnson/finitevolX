"""Forcing base class (Layer 1) and composition wrappers (Layer 3).

:class:`AbstractForcing` is the minimal contract every forcing operator
satisfies: an :class:`equinox.Module` whose ``__call__`` returns a tendency
contribution (a single array or a tuple such as ``(du, dv)``).  The call
signature is deliberately **not** fixed — each forcing takes its own
physics-specific inputs.

The composition wrappers build new forcings out of existing ones:

* :class:`ForcingSum` — ``F = F_1 + F_2 + ... + F_n`` (independent processes)
* :class:`ForcingProduct` — ``F = alpha * F_inner`` (sponges, modulation)
* :class:`TimeVaryingForcing` — evaluates time-dependent input fields
  ``fields = field_fn(t)`` and feeds them to a time-agnostic forcing.

Forcings compose **inside** the right-hand-side function of a model; they
are parts of the vector field, not separate ``diffrax`` terms.  Core forcing
operators stay time-agnostic: time-dependence is resolved either by the
user in ``rhs_fn(t, ...)`` or by :class:`TimeVaryingForcing`.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
import functools
import operator
from typing import Any, cast

import equinox as eqx
import jax
from jaxtyping import Array, Float, PyTree


class AbstractForcing(eqx.Module):
    """Base class for forcing operators.

    Subclasses implement ``__call__`` to return a tendency contribution —
    a single array for scalar forcings (e.g. ``dq``) or a tuple of arrays for
    momentum forcings (e.g. ``(du, dv)``).  The call signature is left to
    each subclass.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import AbstractForcing
    >>> class ConstantForcing(AbstractForcing):
    ...     value: float
    ...
    ...     def __call__(self, q):
    ...         return self.value * jnp.ones_like(q)
    >>> ConstantForcing(value=2.0)(jnp.zeros(3)).tolist()
    [2.0, 2.0, 2.0]
    """

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> PyTree[Array]:
        """Return the tendency contribution of this forcing."""
        raise NotImplementedError


class ForcingSum(AbstractForcing):
    """Additive composition of forcings: ``F = F_1 + F_2 + ... + F_n``.

    Every member is called with the same arguments and the resulting
    tendencies are summed leaf-by-leaf, so all members must return pytrees
    of the same structure (e.g. all ``(du, dv)`` tuples).  Analogous to
    ``diffrax.MultiTerm`` for the pieces of a single vector field.

    Parameters
    ----------
    *forcings : AbstractForcing
        One or more forcings sharing a call signature and output structure.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import AbstractForcing, ForcingSum
    >>> class Scale(AbstractForcing):
    ...     a: float
    ...
    ...     def __call__(self, q):
    ...         return self.a * q
    >>> total = ForcingSum(Scale(1.0), Scale(2.0))
    >>> total(jnp.ones(2)).tolist()
    [3.0, 3.0]
    """

    forcings: tuple[AbstractForcing, ...]

    def __init__(self, *forcings: AbstractForcing) -> None:
        if not forcings:
            raise ValueError("ForcingSum requires at least one forcing.")
        self.forcings = tuple(forcings)

    def __call__(self, *args: Any, **kwargs: Any) -> PyTree[Array]:
        """Sum of every member's tendency, computed leaf-by-leaf.

        Parameters
        ----------
        *args, **kwargs
            Forwarded unchanged to every member forcing.

        Returns
        -------
        PyTree[Array]
            Leaf-wise sum of the member tendencies.
        """
        results = [f(*args, **kwargs) for f in self.forcings]
        # F[leaf] = F_1[leaf] + F_2[leaf] + ... + F_n[leaf]
        return jax.tree.map(lambda *xs: functools.reduce(operator.add, xs), *results)


class ForcingProduct(AbstractForcing):
    """Modulated forcing: ``F = alpha * F_inner``.

    Scales every leaf of the inner forcing's tendency by ``alpha``.  Typical
    uses are sponge layers (``alpha(x)`` a smooth 0 -> 1 ramp near a
    boundary), seasonal modulation (``alpha(t)``), or a learned efficiency
    field (``alpha`` a trainable array).

    Parameters
    ----------
    forcing : AbstractForcing
        The inner forcing.
    modulator : Callable or float or Float[Array, "..."]
        Either a fixed factor (scalar or array broadcastable to each output
        leaf) or a callable.  A callable modulator is called with the **same
        arguments** as the forcing and must return the factor.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import AbstractForcing, ForcingProduct
    >>> class Identity(AbstractForcing):
    ...     def __call__(self, q):
    ...         return q
    >>> sponge = jnp.array([0.0, 0.5, 1.0])
    >>> ForcingProduct(Identity(), sponge)(jnp.full(3, 2.0)).tolist()
    [0.0, 1.0, 2.0]
    """

    forcing: AbstractForcing
    modulator: Callable[..., Any] | float | Float[Array, "..."]

    def __call__(self, *args: Any, **kwargs: Any) -> PyTree[Array]:
        """Modulated tendency ``alpha * F_inner(*args, **kwargs)``.

        Parameters
        ----------
        *args, **kwargs
            Forwarded unchanged to the inner forcing (and to the modulator
            when it is callable).

        Returns
        -------
        PyTree[Array]
            Inner tendency with every leaf multiplied by ``alpha``.
        """
        tendency = self.forcing(*args, **kwargs)
        modulator = self.modulator
        alpha = (
            cast(Callable[..., Any], modulator)(*args, **kwargs)
            if callable(modulator)
            else modulator
        )
        # F[leaf] = alpha * F_inner[leaf]
        return jax.tree.map(lambda t: alpha * t, tendency)


class TimeVaryingForcing(AbstractForcing):
    """Bundle a time-agnostic forcing with time-dependent input fields.

    At call time the input fields are evaluated as ``field_fn(t)`` and passed
    to ``forcing`` as its leading positional arguments; a tuple is unpacked
    into several arguments, anything else is passed as a single argument.
    Any further ``*args`` / ``**kwargs`` follow the evaluated fields.

    ``field_fn`` may be an analytic formula or a data-driven interpolant such
    as ``diffrax.LinearInterpolation(ts, ys).evaluate`` (ERA5, JRA-55, ...),
    both of which are compatible with ``jax.jit`` / ``jax.grad``.

    Parameters
    ----------
    forcing : Callable
        Any forcing operator (typically a Layer 2 operator).
    field_fn : Callable
        ``t -> fields`` — a single array or a tuple of arrays.

    Examples
    --------
    Analytic seasonal modulation of a scalar forcing:

    >>> import jax.numpy as jnp
    >>> from finitevolx import AbstractForcing, TimeVaryingForcing
    >>> class Scale(AbstractForcing):
    ...     def __call__(self, q, a):
    ...         return a * q
    >>> tv = TimeVaryingForcing(Scale(), lambda t: jnp.full(2, 1.0 + t))
    >>> tv(1.0, a=3.0).tolist()
    [6.0, 6.0]

    Data-driven wind from gridded reanalysis (sketch)::

        import diffrax as dfx

        tau_x_path = dfx.LinearInterpolation(ts=era5_times, ys=era5_tau_x)
        tau_y_path = dfx.LinearInterpolation(ts=era5_times, ys=era5_tau_y)
        tv_wind = TimeVaryingForcing(
            forcing=wind_op,
            field_fn=lambda t: (tau_x_path.evaluate(t), tau_y_path.evaluate(t)),
        )
        du, dv = tv_wind(t, dz_top=50.0)
    """

    forcing: Callable[..., PyTree[Array]]
    field_fn: Callable[[Any], Any]

    def __call__(self, t: Any, *args: Any, **kwargs: Any) -> PyTree[Array]:
        """Evaluate the fields at ``t`` and apply the wrapped forcing.

        Parameters
        ----------
        t : scalar
            Time at which to evaluate ``field_fn``.
        *args, **kwargs
            Extra arguments forwarded to the forcing after the fields.

        Returns
        -------
        PyTree[Array]
            The wrapped forcing's tendency.
        """
        fields = self.field_fn(t)
        if isinstance(fields, tuple):
            return self.forcing(*fields, *args, **kwargs)
        return self.forcing(fields, *args, **kwargs)
