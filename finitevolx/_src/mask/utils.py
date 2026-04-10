"""Internal mask construction helpers (numpy/scipy, not JIT-traced).

These primitives are dimension-agnostic — they accept arrays of any
``ndim`` and dispatch on shape.  They are used by ``Mask1D``, ``Mask2D``
and ``Mask3D`` factory methods to derive staggered masks, classify cells,
and build sponge layers.

All routines operate on ``np.ndarray`` inputs and return ``np.ndarray``
outputs.  Conversion to JAX is done by the caller after construction.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.ndimage import binary_dilation


def _pool_bool(
    arr: np.ndarray,
    kernel: tuple[int, ...],
    threshold: float,
) -> np.ndarray:
    """n-D average-pool of a mask with leading-side zero-padding.

    Pads ``(k - 1)`` cells on the leading side of each axis so the output
    shape equals the input shape::

        pool[idx] = mean(arr[idx - (kernel - 1) : idx + 1, ...]) > threshold

    Parameters
    ----------
    arr : np.ndarray
        Input mask (float values in {0, 1}).  Any number of dimensions.
    kernel : tuple of int
        Per-axis kernel sizes.  Length must equal ``arr.ndim``.
    threshold : float
        Wet/dry threshold applied to the local mean.

    Returns
    -------
    np.ndarray
        Boolean array with the same shape as ``arr``.
    """
    if len(kernel) != arr.ndim:
        raise ValueError(
            f"kernel length {len(kernel)} does not match arr.ndim {arr.ndim}"
        )
    arr_f = arr.astype(float)
    pad_widths = tuple((k - 1, 0) for k in kernel)
    arr_padded = np.pad(arr_f, pad_widths)
    total = np.zeros_like(arr_f)
    for offsets in itertools.product(*(range(k) for k in kernel)):
        slices = tuple(
            slice(o, o + s) for o, s in zip(offsets, arr.shape, strict=True)
        )
        total += arr_padded[slices]
    return total / float(np.prod(kernel)) > threshold


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Binary dilation by 1 cell with an n-D cross structuring element.

    Uses ``scipy.ndimage.binary_dilation`` with a (3,)*ndim cross — only
    the centre and the immediate axis-aligned neighbours are True
    (no diagonals).  Boundary cells are treated as zero (no wraparound).

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask of any number of dimensions.

    Returns
    -------
    np.ndarray
        Dilated boolean mask, same shape.
    """
    ndim = mask.ndim
    struct = np.zeros((3,) * ndim, dtype=bool)
    centre = (1,) * ndim
    struct[centre] = True
    for axis in range(ndim):
        for offset in (0, 2):
            idx = list(centre)
            idx[axis] = offset
            struct[tuple(idx)] = True
    return binary_dilation(mask.astype(bool), structure=struct, border_value=0)


def _count_contiguous(
    arr: np.ndarray,
    axis: int,
    forward: bool,
) -> np.ndarray:
    """Count contiguous wet cells from each point along one axis (n-D).

    For each cell, the result is the number of consecutive wet cells
    starting **at** that cell and moving in the chosen direction.  A wet
    cell at the start counts as 1; a dry cell returns 0.

    Parameters
    ----------
    arr : np.ndarray
        Wet/dry mask of any number of dimensions.
    axis : int
        Axis along which to scan.
    forward : bool
        ``True``  → positive-axis direction.
        ``False`` → negative-axis direction.

    Returns
    -------
    np.ndarray
        int32 array of the same shape as ``arr``.
    """
    arr_int = np.asarray(arr, dtype=np.int32)
    # Move the scan axis to position 0 so the loop iterates over scalars
    # along that axis while keeping the perpendicular axes vectorised.
    arr_moved = np.moveaxis(arr_int, axis, 0)
    n = arr_moved.shape[0]
    count = np.zeros_like(arr_moved)

    if forward:
        # Recurrence (positive direction means scanning from high indices down):
        #   count[i] = arr[i] * (1 + count[i+1])
        count[-1] = arr_moved[-1]
        for i in range(n - 2, -1, -1):
            count[i] = arr_moved[i] * (1 + count[i + 1])
    else:
        # Negative direction: scanning from low indices up:
        #   count[i] = arr[i] * (1 + count[i-1])
        count[0] = arr_moved[0]
        for i in range(1, n):
            count[i] = arr_moved[i] * (1 + count[i - 1])

    return np.moveaxis(count, 0, axis)


def _make_sponge(shape: tuple[int, ...], width: int) -> np.ndarray:
    """n-D linear sponge ramp: 0 at every wall, 1 inside the interior.

    Per axis, builds a ramp that rises linearly from 0 at the wall to 1
    at distance ``width`` cells inside.  The full sponge is the
    elementwise product of all per-axis ramps via broadcasting.

    Parameters
    ----------
    shape : tuple of int
        Output array shape.
    width : int
        Number of cells over which the ramp rises from 0 to 1.  A value
        of 0 (or any non-positive value) returns an all-ones array.

    Returns
    -------
    np.ndarray
        float32 sponge array with the requested shape.
    """
    if width is None or width <= 0:
        return np.ones(shape, dtype=np.float32)

    width_f = float(width)
    out = np.ones(shape, dtype=np.float32)
    for axis, n in enumerate(shape):
        idx = np.arange(n, dtype=np.float32)
        ramp = np.clip(np.minimum(idx, (n - 1) - idx) / width_f, 0.0, 1.0)
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = n
        out = out * ramp.reshape(broadcast_shape)
    return out
