"""Frozen dataclasses that become JAX pytrees when JAX can be imported.

The grid containers hold NumPy arrays and Python scalars and are built
without JAX, so a NumPy-only install can describe the grid a model ran on.
When JAX is importable, :func:`pytree_dataclass` also registers the class as
a pytree, so a grid passes through ``jax.jit``, ``jax.vmap`` and
``eqx.filter_jit`` like the ``eqx.Module`` it replaces.

Fields declared with :func:`static_field` are pytree metadata: they must be
hashable, they are compared by value when JAX caches a trace, and changing
one retraces. Every other field is a leaf.
"""

from __future__ import annotations

import dataclasses
from typing import Any, dataclass_transform

import numpy as np

_STATIC = "static"


def static_field(**kwargs: Any) -> Any:
    """A dataclass field that is pytree metadata, not a leaf.

    Use it for sizes, strings and flags that pick a code path or a shape.

    Parameters
    ----------
    **kwargs
        Passed to :func:`dataclasses.field`.

    Returns
    -------
    dataclasses.Field
    """
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_STATIC] = True
    return dataclasses.field(metadata=metadata, **kwargs)


def array_namespace(*values: Any) -> Any:
    """The array namespace of the first non-NumPy array among ``values``.

    Lets a property written once run on the NumPy arrays a grid is built
    with and on the JAX arrays or tracers that replace them under a
    transform. Python scalars and NumPy arrays resolve to NumPy.

    Parameters
    ----------
    *values
        Arrays or scalars.

    Returns
    -------
    module
        ``numpy``, or the namespace a value reports through
        ``__array_namespace__`` (``jax.numpy`` for JAX arrays and tracers).
    """
    for value in values:
        if isinstance(value, np.ndarray | np.generic):
            continue
        get_namespace = getattr(value, "__array_namespace__", None)
        if get_namespace is not None:
            return get_namespace()
    return np


def _register_with_jax(cls: type) -> None:
    """Register ``cls`` as a JAX pytree, if JAX can be imported.

    Unflattening bypasses ``__init__`` and ``__post_init__``: JAX rebuilds
    the object with tracers, ``None`` or sentinel leaves, which a
    constructor's checks must not see.
    """
    try:
        import jax
    except ImportError:
        return

    fields = dataclasses.fields(cls)
    meta_names = tuple(f.name for f in fields if f.metadata.get(_STATIC, False))
    data_names = tuple(f.name for f in fields if not f.metadata.get(_STATIC, False))
    keys = tuple(jax.tree_util.GetAttrKey(name) for name in data_names)

    def flatten_with_keys(obj):
        children = [
            (key, getattr(obj, name))
            for key, name in zip(keys, data_names, strict=True)
        ]
        return children, tuple(getattr(obj, name) for name in meta_names)

    def flatten(obj):
        children = [getattr(obj, name) for name in data_names]
        return children, tuple(getattr(obj, name) for name in meta_names)

    def unflatten(meta, children):
        obj = object.__new__(cls)
        for name, value in zip(meta_names, meta, strict=True):
            object.__setattr__(obj, name, value)
        for name, value in zip(data_names, children, strict=True):
            object.__setattr__(obj, name, value)
        return obj

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float | int | str | bool | type(None)) or isinstance(
        b, float | int | str | bool | type(None)
    ):
        return type(a) is type(b) and a == b
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except Exception:  # tracers and other values with no concrete value
        return a is b


def _pytree_eq(self: Any, other: Any) -> bool:
    """Equal when of the same class with equal fields (arrays by value)."""
    if type(self) is not type(other):
        return NotImplemented
    return all(
        _values_equal(getattr(self, f.name), getattr(other, f.name))
        for f in dataclasses.fields(self)
    )


def _pytree_hash(self: Any) -> int:
    """Hash of the class and the scalar fields (array fields are skipped)."""
    scalars = tuple(
        getattr(self, f.name)
        for f in dataclasses.fields(self)
        if isinstance(getattr(self, f.name), int | float | str | bool)
    )
    return hash((type(self), scalars))


@dataclass_transform(
    frozen_default=True, field_specifiers=(dataclasses.field, static_field)
)
def pytree_dataclass[T: type](cls: T) -> T:
    """Make ``cls`` a frozen dataclass, and a JAX pytree when JAX is importable.

    Equality compares the class and every field, arrays by value, so two
    grids built the same way are equal. The hash uses the class and the
    scalar fields only, which keeps it consistent with equality.

    Parameters
    ----------
    cls : type
        The class body to decorate. Subclasses of a decorated class must be
        decorated too.

    Returns
    -------
    type
        The same class, now a frozen dataclass.

    Examples
    --------
    >>> @pytree_dataclass
    ... class Box:
    ...     n: int = static_field()
    ...     width: float
    >>> Box(n=3, width=1.5)
    Box(n=3, width=1.5)
    """
    cls = dataclasses.dataclass(frozen=True, eq=False)(cls)
    cls.__eq__ = _pytree_eq
    cls.__hash__ = _pytree_hash
    _register_with_jax(cls)
    return cls
