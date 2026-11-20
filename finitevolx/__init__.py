"""Deprecated: finitevolX is now gridax-jax (``import gridax_jax``) on gridax.

``import finitevolx`` warns once and re-exports every public name of
:mod:`gridax_jax`, which includes the grid and mask classes of :mod:`gridax`;
``finitevolx.X is gridax_jax.X`` for every ``X``. The shim is removed after
one 0.X release: replace ``import finitevolx as fvx`` with
``import gridax_jax as fvx`` (and ``finitevolx.pipekit`` with
``gridax_jax.pipekit``).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "finitevolx is renamed gridax_jax (pip install gridax-jax); the grids and"
    " masks are in gridax. `import finitevolx` re-exports both and will be"
    " removed after one 0.X release: use `import gridax_jax`.",
    DeprecationWarning,
    stacklevel=2,
)

import gridax_jax as _gridax_jax

__all__ = list(_gridax_jax.__all__)

# Copy the names gridax_jax defines eagerly; the deprecated steppers it serves
# through its own ``__getattr__`` stay lazy, so they warn only when used.
globals().update(
    {
        name: _gridax_jax.__dict__[name]
        for name in __all__
        if name in _gridax_jax.__dict__
    }
)


def __getattr__(name: str) -> object:
    """Names served lazily by :mod:`gridax_jax` (the deprecated steppers)."""
    return getattr(_gridax_jax, name)
