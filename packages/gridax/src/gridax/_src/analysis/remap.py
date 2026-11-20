"""Remap a field along one axis onto new coordinates, on any array backend.

Moved from ``xrops.transforms`` (#423), where the kernels looped over columns
with ``np.interp``. Here they gather the two bracketing levels of every
target at once, in the namespace of the values (NumPy, JAX, CuPy, …):

.. math::

    f(\\zeta) = f_{lo} + \\frac{f_{hi} - f_{lo}}{z_{hi} - z_{lo}}\\,(\\zeta - z_{lo})

which is ``np.interp``'s formula, so the NumPy results equal the old ones.

:func:`remap_axis` takes one 1-D source coordinate (pressure levels, a time
axis); :func:`remap_axis_columns` takes a source coordinate of the values'
shape, one set of levels per column (terrain-following or hybrid levels).

Dtypes follow the input: a floating or complex input keeps its dtype, and an
integer one is computed in the namespace's default float.
"""

from __future__ import annotations

from typing import Any, Literal

from jaxtyping import Float, Inexact
import numpy as np

from gridax._src.pytree import array_namespace

Extrapolate = Literal["nan", "nearest"]
_EXTRAPOLATE_MODES = ("nan", "nearest")
_METHODS = ("linear", "nearest")


def _as_inexact(xp: Any, values: Any) -> Any:
    """``values`` as a floating or complex array of its namespace."""
    arr = xp.asarray(values)
    if xp.isdtype(arr.dtype, ("real floating", "complex floating")):
        return arr
    return xp.astype(arr, xp.asarray(0.0).dtype)


def _real_dtype(xp: Any, arr: Any) -> Any:
    """The real floating dtype of ``arr`` (float32 for complex64, …)."""
    return xp.finfo(arr.dtype).dtype


def _concrete(x: Any) -> np.ndarray | None:
    """``x`` as a NumPy array, or ``None`` when it has no value yet (a tracer)."""
    try:
        return np.asarray(x)
    except Exception:  # JAX tracers refuse conversion
        return None


def _take(xp: Any, f: Any, index: Any) -> Any:
    """``f[..., index[..., k]]`` for every target ``k``; ``index`` broadcasts."""
    index = xp.broadcast_to(index, (*f.shape[:-1], index.shape[-1]))
    return xp.take_along_axis(f, index, axis=-1)


def _interpolate(
    xp: Any,
    f: Any,
    z: Any,
    tgt: Any,
    method: str,
    extrapolate: str,
) -> Any:
    """Remap ``f[..., n]`` on ascending levels ``z[..., n]`` to ``tgt[m]``."""
    n = z.shape[-1]
    zk = tgt[(None,) * (z.ndim - 1) + (slice(None),)]  # [..., m]
    z_first, z_last = z[..., :1], z[..., n - 1 :]
    if method == "linear":
        # hi = clip(#{j: z_j <= zeta}, 1, n - 1), lo = hi - 1
        count = xp.sum(
            xp.astype(z[..., None, :] <= zk[..., :, None], xp.int64), axis=-1
        )
        hi = xp.clip(count, 1, n - 1)
        lo = hi - 1
        z_lo, z_hi = _take(xp, z, lo), _take(xp, z, hi)
        f_lo, f_hi = _take(xp, f, lo), _take(xp, f, hi)
        # f_lo + (f_hi - f_lo) / (z_hi - z_lo) * (zeta - z_lo), np.interp's order
        out = (f_hi - f_lo) / (z_hi - z_lo) * (zk - z_lo) + f_lo
        # an exact hit returns that level's value; beyond the ends hold them
        out = xp.where(z_lo == zk, f_lo, out)
        out = xp.where(z_hi == zk, f_hi, out)
        out = xp.where(zk < z_first, f[..., :1], out)
        out = xp.where(zk > z_last, f[..., n - 1 :], out)
    else:
        safe = xp.where(xp.isnan(zk), z_first, zk)
        idx = xp.argmin(xp.abs(z[..., None, :] - safe[..., :, None]), axis=-1)
        out = _take(xp, f, idx)
    nan = xp.asarray(xp.nan, dtype=out.dtype)
    if extrapolate == "nan":
        out = xp.where((zk < z_first) | (zk > z_last), nan, out)
    return xp.where(xp.isnan(zk), nan, out)


def remap_axis(
    values: Inexact[np.ndarray, "..."],
    *,
    axis: int = -1,
    source_coords: Float[np.ndarray, "src"],
    target_coords: Float[np.ndarray, "tgt"],
    method: str = "linear",
) -> Inexact[np.ndarray, "..."]:
    """Interpolate ``values`` along ``axis`` from ``source_coords`` to ``target_coords``.

    Parameters
    ----------
    values : array [..., src, ...]
        Real or complex field; complex values are interpolated linearly, which
        is the same as interpolating the real and imaginary parts apart.
    axis : int, optional
        The axis to remap. Default ``-1``.
    source_coords : array [src]
        Strictly monotonic (ascending or descending) levels. They must be
        concrete (NumPy, or a JAX array outside ``jit``): the orientation
        decides the code path.
    target_coords : array [tgt]
        Target levels. Targets outside the source range, or ``NaN``, give
        ``NaN``.
    method : {"linear", "nearest"}, optional
        Default ``"linear"``.

    Returns
    -------
    array
        ``values``' shape with ``axis`` of length ``len(target_coords)``,
        in ``values``' namespace and dtype.

    Raises
    ------
    ValueError
        If a coordinate is not 1-D, the lengths disagree, the source is not
        strictly monotonic, or ``method`` is unknown.

    Examples
    --------
    >>> p = np.array([1000.0, 850.0, 500.0])  # descending pressure
    >>> t = np.array([[290.0, 280.0, 250.0]])
    >>> remap_axis(t, source_coords=p, target_coords=np.array([925.0, 675.0]))
    array([[285., 265.]])
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; expected 'linear' or 'nearest'")
    xp = array_namespace(values)
    arr = _as_inexact(xp, values)
    src_np = _concrete(source_coords)
    if src_np is None:
        raise ValueError("remap_axis needs concrete source_coords (not a tracer)")
    # coordinates take the values' precision: the caller decides the dtype
    tgt = xp.asarray(target_coords, dtype=_real_dtype(xp, arr))
    if src_np.ndim != 1 or tgt.ndim != 1:
        raise ValueError("source_coords and target_coords must be 1D")
    if arr.shape[axis] != src_np.size:
        raise ValueError(
            f"values.shape[axis]={arr.shape[axis]} but len(source_coords)={src_np.size}"
        )
    diffs = np.diff(src_np)
    if np.all(diffs > 0):
        flip = False
    elif np.all(diffs < 0):
        flip = True
    else:
        raise ValueError("source_coords must be strictly monotonic")
    moved = xp.moveaxis(arr, axis, -1)
    src = xp.asarray(src_np[::-1].copy() if flip else src_np, dtype=tgt.dtype)
    if flip:
        moved = xp.flip(moved, axis=-1)
    out = _interpolate(xp, moved, src, tgt, method, extrapolate="nan")
    return xp.moveaxis(out, -1, axis)


def remap_axis_columns(
    values: Inexact[np.ndarray, "..."],
    *,
    axis: int,
    source_coords: Float[np.ndarray, "..."],
    target_coords: Float[np.ndarray, "tgt"],
    method: str = "linear",
    extrapolate: Extrapolate = "nan",
) -> Inexact[np.ndarray, "..."]:
    """Interpolate every column of ``values`` onto ``target_coords``.

    Unlike :func:`remap_axis`, ``source_coords`` has ``values``' shape: each
    column (each position along the other axes) carries its own levels, and
    each column must be strictly monotone along ``axis`` (ascending or
    descending; columns may differ). A column that broadcasts one 1-D
    coordinate gives :func:`remap_axis`'s result exactly.

    Parameters
    ----------
    values : array
        At least two levels along ``axis``; real or complex.
    axis : int
        The axis to remap.
    source_coords : array
        Source levels, ``values``' shape. May be traced: the orientation of
        each column is chosen with ``where``. The monotonicity check runs
        only on concrete levels.
    target_coords : array [tgt]
        Target levels; ``NaN`` targets give ``NaN``.
    method : {"linear", "nearest"}, optional
        Default ``"linear"``.
    extrapolate : {"nan", "nearest"}, optional
        Targets outside a column's own range: ``NaN`` (default), or that
        column's end value.

    Returns
    -------
    array
        ``values``' shape with ``axis`` of length ``len(target_coords)``.

    Raises
    ------
    ValueError
        If shapes disagree, ``target_coords`` is not 1-D, ``axis`` has fewer
        than two levels, a concrete column is not strictly monotone, or
        ``method`` / ``extrapolate`` is unknown.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; expected 'linear' or 'nearest'")
    if extrapolate not in _EXTRAPOLATE_MODES:
        raise ValueError(
            f"unknown extrapolate {extrapolate!r}; expected one of "
            f"{_EXTRAPOLATE_MODES!r}"
        )
    xp = array_namespace(values, source_coords)
    arr = _as_inexact(xp, values)
    src = xp.asarray(source_coords)
    tgt = xp.asarray(target_coords)
    if tgt.ndim != 1:
        raise ValueError("target_coords must be 1D")
    if tuple(src.shape) != tuple(arr.shape):
        raise ValueError(
            f"source_coords shape {tuple(src.shape)} must equal values shape "
            f"{tuple(arr.shape)}"
        )
    n_lev = arr.shape[axis]
    if n_lev < 2:
        raise ValueError(
            f"remap_axis_columns needs at least two levels along axis, got {n_lev}"
        )
    z = xp.moveaxis(src, axis, -1)
    f = xp.moveaxis(arr, axis, -1)
    z = xp.astype(z, _real_dtype(xp, arr))
    tgt = xp.astype(tgt, z.dtype)
    z_np = _concrete(z)
    if z_np is not None:
        steps = np.diff(z_np, axis=-1)
        bad = ~(np.all(steps > 0, axis=-1) | np.all(steps < 0, axis=-1))
        if bad.any():
            raise ValueError(
                "source_coords must be strictly monotone along axis in every column; "
                f"{int(bad.sum())} of {bad.size} columns are not"
            )
    # flip descending columns so every column is ascending
    descending = z[..., :1] > z[..., n_lev - 1 :]
    z = xp.where(descending, xp.flip(z, axis=-1), z)
    f = xp.where(descending, xp.flip(f, axis=-1), f)
    # one target at a time: a [columns, levels] temporary, not [columns,
    # targets, levels], as the NumPy kernel did
    out = xp.concat(
        [
            _interpolate(xp, f, z, tgt[k : k + 1], method, extrapolate)
            for k in range(tgt.shape[0])
        ],
        axis=-1,
    )
    return xp.moveaxis(out, -1, axis)


__all__ = ["Extrapolate", "remap_axis", "remap_axis_columns"]
