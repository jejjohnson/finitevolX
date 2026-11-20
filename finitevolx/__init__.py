"""Deprecated: finitevolX is now gridax-ops (``import gridax_ops``) on gridax.

``import finitevolx`` warns once and re-exports every public name of
:mod:`gridax_ops`, which includes the grid and mask classes of :mod:`gridax`;
``finitevolx.X is gridax_ops.X`` for every ``X``. The shim is removed after
one 0.X release: replace ``import finitevolx as fvx`` with
``import gridax_ops as fvx`` (and ``finitevolx.pipekit`` with
``gridax_ops.pipekit``).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "finitevolx is renamed gridax_ops (pip install gridax-ops); the grids and"
    " masks are in gridax. `import finitevolx` re-exports both and will be"
    " removed after one 0.X release: use `import gridax_ops`.",
    DeprecationWarning,
    stacklevel=2,
)

import gridax_ops as _gridax_ops

__all__ = list(_gridax_ops.__all__)

# Copy the names gridax_ops defines eagerly; the deprecated steppers it serves
# through its own ``__getattr__`` stay lazy, so they warn only when used.
globals().update(
    {
        name: _gridax_ops.__dict__[name]
        for name in __all__
        if name in _gridax_ops.__dict__
    }
)


def __getattr__(name: str) -> object:
    """Names served lazily by :mod:`gridax_ops` (the deprecated steppers)."""
    return getattr(_gridax_ops, name)
