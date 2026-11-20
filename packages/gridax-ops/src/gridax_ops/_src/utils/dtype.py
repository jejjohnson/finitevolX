"""The dtype rule: the caller decides, and a silent downcast is an error.

An operation returns the floating dtype of its input. JAX breaks that for
one case only: with ``jax_enable_x64`` off (JAX's default), a float64 array
that enters JAX becomes float32 without a warning, so a float64 reanalysis
cube would come back at float32 precision. :func:`check_dtype` turns that
case into an error at the point where host data enters a kernel; the caller
then chooses, by enabling x64 or by passing float32 on purpose.

Grid metrics are not covered: they are properties of the grid, not data, and
take the field's dtype (``operators._utils._metric_like``).
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np


def check_dtype(x: Any, *, name: str = "input") -> None:
    """Raise if JAX would silently downcast ``x`` from float64 to float32.

    ``x`` fails when its dtype is float64 (or complex128) and
    ``jax_enable_x64`` is off. Everything else passes: float32 and lower
    precisions, integers and booleans, and float64 with x64 on. A JAX array
    already exists at its final dtype, so it always passes; tracers do too.

    Parameters
    ----------
    x : array-like
        The value about to enter a JAX kernel: a NumPy array, a NumPy or
        Python scalar, or anything with a ``dtype``.
    name : str, optional
        What ``x`` is, for the error message. Default ``"input"``.

    Raises
    ------
    TypeError
        If ``x`` is float64 or complex128 and ``jax_enable_x64`` is off.

    Examples
    --------
    >>> import numpy as np
    >>> check_dtype(np.zeros(3, dtype=np.float32))  # passes, x64 on or off
    >>> jax.config.update("jax_enable_x64", False)
    >>> check_dtype(np.zeros(3), name="sst")
    Traceback (most recent call last):
        ...
    TypeError: sst is float64, but jax_enable_x64 is off, ...
    """
    if isinstance(x, jax.Array):
        return
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        # Python floats are weakly typed in JAX: they take the other
        # operand's dtype instead of forcing one, so they never downcast.
        return
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        return
    if jax.config.read("jax_enable_x64"):
        return
    narrow = "complex64" if dtype.kind == "c" else "float32"
    raise TypeError(
        f"{name} is {dtype}, but jax_enable_x64 is off, so JAX would compute"
        f" in {narrow} without saying so. Enable it"
        " (jax.config.update('jax_enable_x64', True)) to keep float64, or pass"
        f" {narrow} to choose the lower precision."
    )
