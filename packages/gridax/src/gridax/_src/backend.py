"""Backend hooks: the few operations the array API does not cover.

The stencils and operators of gridax are written against the array API, so
they run on NumPy, JAX, CuPy or anything ``array-api-compat`` knows. Four
operations need more than the standard offers. Each hook keeps JAX's own
call when the input is a JAX array, so jitted code lowers exactly as it did
when it was written in ``jax.numpy`` (``tests/test_stencil_lowering.py``
pins that), and uses a plain equivalent everywhere else.

- :func:`set_at`: a functional ``x[index] = values`` (``x.at[index].set`` on
  JAX, a copy and an assignment elsewhere).
- :func:`pad`: constant zero padding (``jnp.pad`` / ``np.pad``, or a
  concatenation of zero blocks on a namespace without ``pad``).
- :func:`optimization_barrier` and :func:`stop_gradient`: identities on
  values that steer XLA and JAX's autodiff; plain identities elsewhere.
"""

from __future__ import annotations

from typing import Any

from gridax._src.pytree import array_namespace


def is_jax(xp: Any) -> bool:
    """Whether ``xp`` is ``jax.numpy``."""
    return getattr(xp, "__name__", "") == "jax.numpy"


def set_at(x: Any, index: Any, values: Any) -> Any:
    """``x`` with ``x[index]`` replaced by ``values``, without mutating ``x``.

    Parameters
    ----------
    x : array
        The array to update.
    index : index expression
        Anything ``x[index]`` accepts (slices, tuples of slices, ellipsis).
    values : array or scalar
        Broadcastable to ``x[index]``.

    Returns
    -------
    array
        A new array of ``x``'s namespace, shape and dtype.
    """
    xp = array_namespace(x, values)
    if is_jax(xp):
        return xp.asarray(x).at[index].set(values)
    out = xp.asarray(x, copy=True)
    out[index] = values
    return out


def pad(x: Any, widths: tuple[tuple[int, int], ...]) -> Any:
    """``x`` padded with zeros, ``widths[k] = (before, after)`` along axis ``k``.

    Parameters
    ----------
    x : array
        The array to pad.
    widths : tuple of (int, int)
        One pair per axis of ``x``.

    Returns
    -------
    array
        The padded array, in ``x``'s namespace and dtype.
    """
    xp = array_namespace(x)
    native = getattr(xp, "pad", None)
    if native is not None:
        return native(x, widths)
    out = x
    for axis, (before, after) in enumerate(widths):
        parts = []
        for width in (before, after):
            shape = list(out.shape)
            shape[axis] = width
            parts.append(xp.zeros(tuple(shape), dtype=out.dtype))
        out = xp.concat([parts[0], out, parts[1]], axis=axis)
    return out


def optimization_barrier(x: Any) -> Any:
    """``x`` unchanged; on JAX, ``jax.lax.optimization_barrier(x)``."""
    if is_jax(array_namespace(x)):
        import jax

        return jax.lax.optimization_barrier(x)
    return x


def stop_gradient(x: Any) -> Any:
    """``x`` unchanged; on JAX, ``jax.lax.stop_gradient(x)``."""
    if is_jax(array_namespace(x)):
        import jax

        return jax.lax.stop_gradient(x)
    return x
