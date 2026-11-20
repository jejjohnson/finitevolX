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
import functools
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
    """The array namespace shared by ``values``, NumPy for plain values.

    Lets a function written once run on the NumPy arrays a grid is built
    with and on the JAX arrays, tracers, CuPy arrays (or anything else that
    follows the array API) a caller passes instead. With
    ``array-api-compat`` installed it decides, so backends without a native
    ``__array_namespace__`` (PyTorch) work too; otherwise the first value
    that reports a namespace other than NumPy's wins. Python scalars,
    ``None`` and NumPy arrays resolve to NumPy.

    Parameters
    ----------
    *values
        Arrays, scalars or ``None``.

    Returns
    -------
    module
        ``numpy``, or the namespace of the arrays (``jax.numpy`` for JAX
        arrays and tracers).
    """
    arrays = [
        v for v in values if v is not None and not isinstance(v, int | float | complex)
    ]
    try:
        import array_api_compat
    except ImportError:
        array_api_compat = None
    if array_api_compat is not None and arrays:
        namespace = array_api_compat.array_namespace(*arrays)
        if namespace.__name__.endswith("numpy") and "jax" not in namespace.__name__:
            return np
        return namespace
    for value in arrays:
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


# Not frozen_default: a hand-written __init__ assigns its fields (frozen at
# run time once it returns), which a frozen hint would flag.
@dataclass_transform(field_specifiers=(dataclasses.field, static_field))
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
    custom_init = "__init__" in cls.__dict__
    if (
        not custom_init
        and hasattr(cls, "__check_init__")
        and "__post_init__" not in cls.__dict__
    ):
        # As on an eqx.Module, __check_init__ runs after construction (and
        # never on unflattening, which bypasses __init__).
        def __post_init__(self: Any) -> None:
            self.__check_init__()

        setattr(cls, "__post_init__", __post_init__)  # noqa: B010
    cls = dataclasses.dataclass(frozen=True, eq=False, init=not custom_init)(cls)
    if custom_init:
        _allow_assignment_in(cls)
    cls.__eq__ = _pytree_eq
    cls.__hash__ = _pytree_hash
    _register_with_jax(cls)
    return cls


_INITIALISING = "_pytree_dataclass_initialising"


def _allow_assignment_in(cls: type) -> None:
    """Let a hand-written ``__init__`` assign fields; freeze the object after it.

    An operator that precomputes in ``__init__`` (an inner operator, an
    adaptive-mask hierarchy) assigns its fields there, as on an
    ``eqx.Module``. Outside ``__init__`` the object is frozen as usual, and
    JAX's unflattening still bypasses ``__init__``.
    """
    user_init = cls.__init__

    @functools.wraps(user_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, _INITIALISING, True)
        try:
            user_init(self, *args, **kwargs)
        finally:
            object.__delattr__(self, _INITIALISING)
        missing = [
            f.name for f in dataclasses.fields(self) if f.name not in self.__dict__
        ]
        if missing:
            raise TypeError(f"{type(self).__name__}.__init__ did not set {missing}")
        check = getattr(self, "__check_init__", None)
        if check is not None:
            check()

    def __setattr__(self: Any, name: str, value: Any) -> None:
        if self.__dict__.get(_INITIALISING, False):
            object.__setattr__(self, name, value)
            return
        raise dataclasses.FrozenInstanceError(f"cannot assign to field {name!r}")

    setattr(cls, "__init__", __init__)  # noqa: B010 (the type checker rejects plain assignment)
    setattr(cls, "__setattr__", __setattr__)  # noqa: B010
