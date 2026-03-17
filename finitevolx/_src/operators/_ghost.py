"""Ghost-ring zeroing utilities for Arakawa C-grid operators."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def interior(values: Array, like: Array, ghost: int = 1) -> Array:
    """Create an array shaped like ``like``, zero everywhere except the interior.

    Parameters
    ----------
    values : Array
        Interior values to embed.
    like : Array
        Reference array whose shape defines the output.
    ghost : int, optional
        Width of the ghost ring (default 1).

    Returns
    -------
    Array
        Zero-padded array with ``values`` written to the interior.
    """
    slices = tuple(slice(ghost, -ghost) for _ in range(like.ndim))
    out = jnp.zeros(like.shape, dtype=jnp.result_type(like, values))
    return out.at[slices].set(values)


def zero_z_ghosts(arr: Array) -> Array:
    """Zero the first and last slices along axis 0.

    Parameters
    ----------
    arr : Array
        3-D (or higher) array whose first and last z-slices should be zeroed.

    Returns
    -------
    Array
        Array with ``arr[0]`` and ``arr[-1]`` set to zero.
    """
    return arr.at[0].set(0.0).at[-1].set(0.0)
